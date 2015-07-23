"""RandomForestClassifier"""

import numpy
import multiprocessing as mp
from copy import deepcopy
from joblib import Parallel, delayed
import pickle
import random

try:
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
except ImportError:
    pass

from .FeatureFunction import FeatureFunction
from .DecisionTree import myDecisionTreeClassifier
from .Criterion import Entropy, Gini


MAX_INT = numpy.iinfo(numpy.int32).max

def _get_n_jobs(n_jobs):
	"""Get number of jobs for the computation."""
	if n_jobs < 0:
		return max(mp.cpu_count() + 1 + n_jobs, 1)
	elif n_jobs == 0:
		raise ValueError('Parameter n_jobs == 0 has no meaning.')
	else:
		return n_jobs

def _partition_estimators(n_estimators, n_jobs):
	"""Private function used to partition estimators between jobs."""
	# Compute the number of jobs
	n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

	# Partition estimators between jobs
	n_estimators_per_job = (n_estimators // n_jobs) * numpy.ones(n_jobs,
															  dtype=numpy.int)
	n_estimators_per_job[:n_estimators % n_jobs] += 1
	starts = numpy.cumsum(n_estimators_per_job)

	return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

def _parallel_build_trees(tree, forest, X, y, tree_idx, n_trees,
						  verbose=0, class_weight=None):
	"""Private function used to fit a single tree in parallel."""
	if verbose > 1:
		print("building tree %d of %d" % (tree_idx + 1, n_trees))

	if forest.bootstrap:
		n_samples = X.shape[0]
		curr_sample_weight = numpy.ones((n_samples,), dtype=numpy.float64)

		indices = numpy.random.randint(0, n_samples, n_samples)
		sample_counts = numpy.bincount(indices, minlength=n_samples)
		curr_sample_weight *= sample_counts

		if class_weight == 'subsample':
			curr_sample_weight *= compute_sample_weight('auto', y, indices)
		tree.fit(X, y, curr_sample_weight)

		tree.indices_ = sample_counts > 0.

	else:
		tree.fit(X, y)

	return tree

def _parallel_build_trees_image(tree, forest, raster_data, sample_index, y, tree_idx, n_trees,
						  verbose=0, class_weight=None):
	"""Private function used to fit a single tree in parallel."""
	if verbose > 1:
		print("building tree %d of %d" % (tree_idx + 1, n_trees))

	if forest.bootstrap:
		n_samples = sample_index.shape[0]
		curr_sample_weight = numpy.ones((n_samples,), dtype=numpy.float64)

		indices = numpy.random.randint(0, n_samples, n_samples)
		sample_counts = numpy.bincount(indices, minlength=n_samples)
		curr_sample_weight *= sample_counts

		if class_weight == 'subsample':
			curr_sample_weight *= compute_sample_weight('auto', y, indices)
		tree.fit_image(raster_data, sample_index, y, curr_sample_weight)

		tree.indices_ = sample_counts > 0.

	else:
		tree.fit_image(raster_data, sample_index, y)

	return tree

def _parallel_helper(obj, methodname, *args, **kwargs):
	"""Private helper to workaround Python 2 pickle limitations"""
	return getattr(obj, methodname)(*args, **kwargs)


class myRandomForestClassifier():
	"""myRandomForestClassifier"""

	def __init__(self,
					n_estimators = 10,
					max_depth = None,
					min_samples_split = 2,
					max_features = 10,
					min_samples_leaf = 1,
					max_leaf_nodes = -1,
					featureFunction = FeatureFunction(),
					criterion = Gini(),
					verbose=0,
					bootstrap=True,
					oob_score=False,
					n_jobs=4):

		self.n_estimators=n_estimators
		
		self.estimator=myDecisionTreeClassifier(max_depth,
				min_samples_split,
				max_features,
				min_samples_leaf,
				max_leaf_nodes,
				criterion,
				featureFunction)
		self.featureFunction = featureFunction
		self.verbose = verbose
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = n_jobs
		self.estimators_ = []

	def fit(self, X, y):
		"""Build a forest of trees from the training set (X, y)"""

		n_samples = X.shape[0]

		#Get classes
		tmp = list(set(y))
		self.classes = numpy.array(tmp)
		self.n_classes = len(tmp)

		self.y = y

		trees = []

		for i in range(self.n_estimators):
			tree = deepcopy(self.estimator)
			trees.append(tree)

		# Parallel loop: we use the threading backend as the Cython code
		# for fitting the trees is internally releasing the Python GIL
		# making threading always more efficient than multiprocessing in
		# that case.


		if (dview is None):
			trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
							 backend="threading")(
				delayed(_parallel_build_trees)(
					t, self, X, y, i, len(trees),
					verbose=self.verbose)
				for i, t in enumerate(trees))
		else:
			tasks = []
			for i, t in enumerate(trees):
				ar = dview.apply_async(_parallel_build_trees,
					t, self, X, y, i, len(trees),
					verbose=self.verbose)
				tasks.append(ar)

			# wait for computation to end
			trees = [ar.get() for ar in tasks]

		# Collect newly grown trees
		self.estimators_.extend(trees)

		if self.oob_score:
			self._set_oob_score(X, y)

		# Decapsulate classes_ attributes
		if hasattr(self, "classes_") and self.n_outputs_ == 1:
			self.n_classes_ = self.n_classes_[0]
			self.classes_ = self.classes_[0]
		
		self.samples = []
		self.y = []

		return self

	def fit_image(self, raster_data, sample_index, y, dview = None):
		"""Build a forest of trees from the training set (X, y)"""

		n_samples = sample_index.shape[0]

		#Get classes
		tmp = list(set(y))
		self.classes = numpy.array(tmp)
		self.n_classes = len(tmp)

		self.y = y

		trees = []

		for i in range(self.n_estimators):
			tree = deepcopy(self.estimator)
			trees.append(tree)

		n_jobs = self.n_jobs

		if (dview is None):
			trees = Parallel(n_jobs=n_jobs, verbose=self.verbose,
							 backend="threading")(
				delayed(_parallel_build_trees_image)(
					t, self, raster_data, sample_index, y, i, len(trees),
					verbose = self.verbose)
				for i, t in enumerate(trees))
		else:
			tasks = []
			for i, t in enumerate(trees):
				ar = dview.apply_async(_parallel_build_trees_image,
					t, self, imarray, sample_index, y, i, len(trees),
					verbose = self.verbose, gpu = gpu
					)
				tasks.append(ar)

			# wait for computation to end
			trees = [ar.get() for ar in tasks]


		# Collect newly grown trees
		self.estimators_.extend(trees)

		#TODO ???
		#if self.oob_score:
		#	self._set_oob_score(X, y)

		# Decapsulate classes_ attributes
		if hasattr(self, "classes_") and self.n_outputs_ == 1:
			self.n_classes_ = self.n_classes_[0]
			self.classes_ = self.classes_[0]
		
		self.samples = []
		self.y = []

		return self

	def _set_oob_score(self, X, y):
		"""Compute out-of-bag score"""
		n_classes_ = self.n_classes_
		n_samples = y.shape[0]

		predictions = []

		for k in range(self.n_outputs_):
			predictions.append(numpy.zeros((n_samples, n_classes_[k])))

		sample_indices = numpy.arange(n_samples)
		for estimator in self.estimators_:
			mask = numpy.ones(n_samples, dtype=numpy.bool)
			mask[estimator.indices_] = False
			mask_indices = sample_indices[mask]
			p_estimator = estimator.predict_proba(X[mask_indices, :])

			predictions[mask_indices, :] += p_estimator

		self.oob_decision_function_ = (predictions / predictions.sum(axis=1)[:, numpy.newaxis])
		self.oob_score_ = numpy.mean(y == numpy.argmax(predictions, axis=1), axis=0)

	def predict(self, X):
		"""Predict class for X."""

		proba = numpy.array(self.predict_proba(X))
		return self.classes.take(numpy.argmax(proba, axis=1), axis=0)

	def predict_proba(self, X):
		"""Predict class probabilities for X"""

		# Assign chunk of trees to jobs
		n_jobs, n_trees, starts = _partition_estimators(self.n_estimators,
														self.n_jobs)

		# Parallel loop

		all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
							 backend="threading")(
			delayed(_parallel_helper)(e, 'predict_proba', X)
			for e in self.estimators_)

		# Reduce
		proba = all_proba[0]

		for j in range(1, len(all_proba)):
			proba += all_proba[j]

		return proba / len(self.estimators_)

	def predict_log_proba(self, X):
		"""Predict class log-probabilities for X"""
		proba = self.predict_proba(X)

		return numpy.log(proba)

	def predict_image(self, imarray, w_x, w_y):
		"""Predict class for X."""
		proba = numpy.array(self.predict_proba_image(imarray, w_x, w_y))
		return self.classes.take(numpy.argmax(proba, axis=0))

	def predict_proba_image(self, imarray, w_x, w_y):
		"""Predict class probabilities for X"""

		x_gpu = gpuarray.to_gpu(imarray.astype(numpy.float32))

		proba_gpu = gpuarray.zeros((self.n_classes,imarray.shape[1],imarray.shape[2]),numpy.float32)
		
		for e in self.estimators_:
			e.predict_proba_image(x_gpu,proba_gpu, w_x, w_y)

		proba = proba_gpu.get()

		return proba / len(self.estimators_)

	def getFeatureImportance(self):

		acc = self.featureFunction.getEmptyAccumulator();
		for e in self.estimators_:
			e.getFeatureImportance(acc)
		
		return acc

	def getFeatureImportanceByClass(self):

		acc = self.featureFunction.getEmptyAccumulatorByClass(self.n_classes);
		for e in self.estimators_:
			e.getFeatureImportanceByClass(acc)
		
		return acc		

	def __repr__(self):
		out = "RandomForest : \r\n"
		for t in self.estimators_:
			out +=t.__repr__() + "\r\n"
		return out

	def save(self, filename):
		return pickle.dump(self, open(filename, "wb"))

	@staticmethod
	def load(filename):
		return pickle.load(open(filename,"rb"))
