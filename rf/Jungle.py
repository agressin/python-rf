import numpy
import multiprocessing as mp
from joblib import Parallel, delayed
from copy import deepcopy
import scipy.ndimage as nd

try:
   import cPickle as pickle
except:
   import pickle

from .FeatureFunction import FeatureFunction
from .Criterion import Entropy, Gini
from .Forest import myRandomForestClassifier
from .Geodesic import GDT

class myJungleClassifier():
	"""myJungleClassifier"""

	def __init__(self,
					n_estimators = 10,
					max_depth = None,
					min_samples_split = 2,
					max_features = 10,
					min_samples_leaf = 1,
					max_leaf_nodes = -1,
					entangled = None,
					featureFunction = FeatureFunction(),
					criterion = Gini(),
					verbose=0,
					bootstrap=True,
					oob_score=False,
					n_forests = 2,
					n_steps_simple = None,
					n_steps_proba  = None,
					specialisation = None, # "global" , "per_class", ... ?
					add_previous_prob = False,
					fusion = None, # last_only, mean, mean_last_simple
					use_geodesic = False,
					geodesic_a = 10,
					geodesic_b = 1,
					geodesic_sigma = 5,
					n_jobs=4):

		self.n_forests=n_forests
		self.n_steps_simple = n_steps_simple
		self.n_steps_proba = n_steps_proba
		self.geodesic_cost = None

		self.forest=myRandomForestClassifier(n_estimators,
				max_depth,
				min_samples_split,
				max_features,
				min_samples_leaf,
				max_leaf_nodes,
				featureFunction,
				criterion,
				verbose,
				bootstrap,
				oob_score,
				entangled,
				n_jobs)
		self.featureFunction = featureFunction
		self.verbose = verbose
		self.specialisation = specialisation
		self.fusion = fusion
		self.add_previous_prob = add_previous_prob
		self.use_geodesic = use_geodesic
		self.geodesic_a = geodesic_a
		self.geodesic_b = geodesic_b
		self.geodesic_sigma = geodesic_sigma
		self.forests_ = []

	def geodesic(self,proba):
		a = self.geodesic_a
		b = self.geodesic_b
		for i in range(proba.shape[0]):
			proba[i] = GDT(a*(1-proba[i]), b*self.geodesic_cost)
			proba[i] = 1 - proba[i]/a
		return proba

	def fit(self, X, y, dview = None):
		"""Build a jungle of trees from the training set (X, y)"""
		#Get classes
		classes, y[:] = numpy.unique(y[:], return_inverse=True)
		self.classes_ = classes
		self.n_classes_ = classes.shape[0]
		forests = []

		featureFunction = self.featureFunction
		for i in range(self.n_forests):
			print("forest : ",i+1," / ",self.n_forests)
			if (i != 0):
				if(self.specialisation == 'global'):
					acc = forest.getFeatureImportance()
					featureFunction.random_weight = acc
				elif(self.specialisation =='per_class'):
					acc_per_class = forest.getFeatureImportanceByClass()
					featureFunction.random_weight_per_class = acc_per_class

			forest = deepcopy(self.forest)
			forest.featureFunction = featureFunction
			forest.fit(X, y, dview)
			forests.append(forest)

		# Collect newly grown Forests
		self.forests_.extend(forests)

	def fit_image(self, array_data, sample_index, y, dview = None):
		"""Build a jungle of trees from the training set (X, y)"""

		#Get classes
		classes, y[:] = numpy.unique(y[:], return_inverse=True)
		self.classes_ = classes
		self.n_classes_ = classes.shape[0]

		forests = []
		featureFunction = self.featureFunction
		SWx,SWy = featureFunction.width, featureFunction.height

		if(self.use_geodesic):
			pan = array_data[0]
			self.geodesic_cost = nd.gaussian_gradient_magnitude(pan, self.geodesic_sigma)

		if(self.n_steps_simple is None and self.n_steps_proba is None):
			for i in range(self.n_forests):
				print("forest : ",i+1," / ",self.n_forests)
				if (i != 0):
					if(self.specialisation == 'global'):
						acc = forest.getFeatureImportance()
						featureFunction.random_weight = acc
					elif(self.specialisation =='per_class'):
						acc_per_class = forest.getFeatureImportanceByClass()
						featureFunction.random_weight_per_class = acc_per_class
					if(self.add_previous_prob):
						proba = forest.predict_proba_image(array_data,SWx,SWy)
						if(self.use_geodesic):
							proba = self.geodesic(proba)
						array_data = numpy.concatenate((array_data,proba))
						featureFunction.nb_channels = array_data.shape[0]

				forest = deepcopy(self.forest)
				forest.featureFunction = featureFunction
				forest.fit_image(array_data, sample_index, y, dview)
				forests.append(forest)
		else:
			n_forests = 0
			for step_proba in range(self.n_steps_proba):
				for step_simple in range(self.n_steps_simple):
					print("step_proba : " ,step_proba +1," / ",self.n_steps_proba)
					print("step_simple : ",step_simple+1," / ",self.n_steps_simple)
					if (step_simple != 0):
						if(self.specialisation == 'global'):
							acc = forest.getFeatureImportance()
							featureFunction.random_weight = acc
						elif(self.specialisation =='per_class'):
							acc_per_class = forest.getFeatureImportanceByClass()
							featureFunction.random_weight_per_class = acc_per_class
						#if specialisation
					else :
						featureFunction.random_weight = None
						featureFunction.random_weight_per_class = None
					#if step_simple !=0
					if (step_proba != 0) and (step_simple == 0):
						proba = forest.predict_proba_image(array_data,SWx,SWy)
						if(self.use_geodesic):
							proba = self.geodesic(proba)
						#if use_geodesic
						array_data = numpy.concatenate((array_data,proba))
						featureFunction.nb_channels = array_data.shape[0]
					#if (step_proba != 0) and (step_simple=0):
					forest = deepcopy(self.forest)
					forest.featureFunction = featureFunction
					forest.fit_image(array_data, sample_index, y, dview)
					forests.append(forest)
					n_forests +=1
				#for step_simple
			#for step_proba
			self.n_forests = n_forests

		# Collect newly grown Forests
		self.forests_.extend(forests)

	def predict(self, X):
		"""Predict class for X."""
		proba = numpy.array(self.predict_proba(X))
		return self.classes_.take(numpy.argmax(proba, axis=0))

	def predict_proba(self, X):
		"""Predict class probabilities for X"""
		all_proba = []


		for i in range(self.n_forests):
			proba = self.forests_[i].predict_proba(X)
			all_proba.append(proba)

		if(self.fusion == "mean"):
			for j in range(1, len(all_proba)):
				proba += all_proba[j]
			return proba / self.n_forests
		else: # (fusion =="last"):
			return proba


	def predict_image(self, array_data, w_x, w_y):
		"""Predict class for X."""
		proba = numpy.array(self.predict_proba_image(array_data, w_x, w_y))
		return self.classes_.take(numpy.argmax(proba, axis=0))

	def predict_proba_image(self, array_data, w_x, w_y):
		"""Predict class probabilities for X"""
		all_proba = []

		if(self.use_geodesic):
			pan = array_data[0]
			self.geodesic_cost = nd.gaussian_gradient_magnitude(pan, self.geodesic_sigma)


		if(self.n_steps_simple is None and self.n_steps_proba is None):
			for i in range(self.n_forests):
				if ((i != 0) and self.add_previous_prob):
					if(self.use_geodesic):
						proba = self.geodesic(proba)
					array_data = numpy.concatenate((array_data,proba))
				proba = self.forests_[i].predict_proba_image(array_data, w_x, w_y)
				all_proba.append(proba)
		else:
			i = 0
			done = True
			for step_proba in range(self.n_steps_proba):
				for step_simple in range(self.n_steps_simple):
					if (step_proba != 0) and (step_simple == 0):
						proba = self.forests_[i].predict_proba_image(array_data, w_x, w_y)
						if(self.use_geodesic):
							proba = self.geodesic(proba)
						#if use_geodesic
						array_data = numpy.concatenate((array_data,proba))
					#if (step_proba != 0) and (step_simple=0):
					proba = self.forests_[i].predict_proba_image(array_data, w_x, w_y)
					i +=1
				#for step_simple
			#for step_proba

		if(self.fusion == "mean"):
			for j in range(1, len(all_proba)):
				proba += all_proba[j]
			return proba / self.n_forests
		else: # (fusion =="last"):
			return proba

	def __repr__(self):
		out = "Jungle : \r\n"
		for f in self.forests_:
			out +=F.__repr__() + "\r\n"
		return out

	def save(self, filename):
		return pickle.dump(self, open(filename, "wb"))

	@staticmethod
	def load(filename):
		return pickle.load(open(filename,"rb"))
