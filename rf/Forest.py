"""RandomForestClassifier"""

import numpy
import multiprocessing as mp
from copy import deepcopy
from joblib import Parallel, delayed
try:
   import cPickle as pickle
except:
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

def _parallel_build_trees(tree, bootstrap, X, y, tree_idx, n_trees,
						  verbose=0, class_weight=None):
	"""Private function used to fit a single tree in parallel."""
	if verbose > 1:
		print("building tree %d of %d" % (tree_idx + 1, n_trees))

	if bootstrap:
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

def _parallel_build_trees_image(tree, bootstrap, raster_data, sample_index, y, tree_idx, n_trees,
						  verbose=0, class_weight=None):
	"""Private function used to fit a single tree in parallel."""
	if verbose > 1:
		print("building tree %d of %d" % (tree_idx + 1, n_trees))

	if bootstrap:
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
					entangled=None,
					n_jobs=4):

		self.n_estimators=n_estimators

		self.estimator=myDecisionTreeClassifier(max_depth,
				min_samples_split,
				max_features,
				min_samples_leaf,
				max_leaf_nodes,
				criterion,
				featureFunction,
				entangled)
		self.featureFunction = featureFunction
		self.verbose = verbose
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = n_jobs
		self.estimators_ = []

	def fit(self, X, y, dview = None):
		"""Build a forest of trees from the training set (X, y)"""

		n_samples = X.shape[0]

		#Get classes
		classes, y[:] = numpy.unique(y[:], return_inverse=True)
		self.classes_ = classes
		self.n_classes_ = classes.shape[0]

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
					t, self.bootstrap, X, y, i, len(trees),
					verbose=self.verbose)
				for i, t in enumerate(trees))
		else:
			tasks = []
			for i, t in enumerate(trees):
				ar = dview.apply_async(_parallel_build_trees,
					t, self.bootstrap, X, y, i, len(trees),
					verbose=self.verbose)
				tasks.append(ar)

			# wait for computation to end
			trees = [ar.get() for ar in tasks]

		# Collect newly grown trees
		self.estimators_.extend(trees)

		if self.oob_score:
			self._set_oob_score(X, y)

		return self

	def fit_image(self, raster_data, sample_index, y, dview = None):
		"""Build a forest of trees from the raster training set"""

		n_samples = sample_index.shape[0]

		if (type(raster_data) is numpy.ndarray):
			raster_data=raster_data.cumsum(2).cumsum(1)

		#Get classes
		classes, y[:] = numpy.unique(y[:], return_inverse=True)
		self.classes_ = classes
		self.n_classes_ = classes.shape[0]

		trees = []

		for i in range(self.n_estimators):
			tree = deepcopy(self.estimator)
			trees.append(tree)

		n_jobs = self.n_jobs
		if(self.estimator.entangled is not None):
			trees = [ _parallel_build_trees_image(
										t, self.bootstrap, raster_data,
										sample_index, y, i, len(trees), verbose = self.verbose)
									for i, t in enumerate(trees)]

		elif (dview is None):
			trees = Parallel(n_jobs=n_jobs, verbose=self.verbose,
							 backend="threading")(
				delayed(_parallel_build_trees_image)(
					t, self.bootstrap, raster_data, sample_index, y, i, len(trees),
					verbose = self.verbose)
				for i, t in enumerate(trees))
		else:
			tasks = []
			for i, t in enumerate(trees):
				ar = dview.apply_async(_parallel_build_trees_image,
					t, self.bootstrap, raster_data, sample_index, y, i, len(trees),
					verbose = self.verbose)
				tasks.append(ar)

			# wait for computation to end
			trees = [ar.get() for ar in tasks]


		# Collect newly grown trees
		self.estimators_.extend(trees)

		#TODO ???
		#if self.oob_score:
		#	self._set_oob_score(X, y)


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
		return self.classes_.take(numpy.argmax(proba, axis=1), axis=0)

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

	def predict_image(self, input_data, w_x, w_y):
		"""Predict class for X."""
		proba = numpy.array(self.predict_proba_image(input_data, w_x, w_y))
		return self.classes_.take(numpy.argmax(proba, axis=0))

	def predict_proba_image(self, array_image, w_x, w_y):
		"""Predict class probabilities for X"""

		s_c, s_x, s_y = array_image.shape
		array_image_cum = array_image.cumsum(2).cumsum(1)
		if( self.estimator.entangled is not None):
			tmp = numpy.zeros((self.n_classes_,s_x,s_y))
			array_image_cum = numpy.concatenate((array_image_cum,tmp))
			
		x_gpu = gpuarray.to_gpu(array_image_cum.astype(numpy.float32))
		
		proba_gpu = gpuarray.zeros((self.n_classes_,s_x,s_y),numpy.float32)

		for e in self.estimators_:
			e.predict_proba_image(x_gpu,proba_gpu, w_x, w_y)

		proba = proba_gpu.get()
		del x_gpu
		del proba_gpu

		return proba / len(self.estimators_)

	def getFeatureImportance(self):

		acc = self.featureFunction.getEmptyAccumulator();
		for e in self.estimators_:
			e.getFeatureImportance(acc)

		return acc

	def getFeatureImportanceByClass(self):

		acc = self.featureFunction.getEmptyAccumulatorByClass(self.n_classes_);
		for e in self.estimators_:
			e.getFeatureImportanceByClass(acc)

		return acc

	def __repr__(self):
		out = "RandomForest : \r\n"
		for t in self.estimators_:
			out +=t.__repr__() + "\r\n"
		return out

	def save(self, filename):
		fichier = open(filename, "wb")
		out = pickle.dump(self, fichier, protocol = 4)
		fichier.close()
		return out

	@staticmethod
	def load(filename):
		fichier = open(filename, "rb")
		out = pickle.load(fichier)
		fichier.close()
		return out
