import numpy
import multiprocessing as mp
from joblib import Parallel, delayed

from .FeatureFunction import FeatureFunction


class myJungleClassifier():
	"""myJungleClassifier"""

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
					n_forests = 2,
					specialisation = None, # "global" , "per_class", ... ?
					add_previous_prob = False,
					fusion = None, # last_only, mean, weithed_mean ??, ... ?
					n_jobs=4):

		self.n_forests=n_forests
		
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
				n_jobs)
		self.featureFunction = featureFunction
		self.verbose = verbose
		self.specialisation = specialisation
		self.fusion = fusion
		self.add_previous_prob = add_previous_prob
		self.forests_ = []


	def fit(self, X, y):
		"""Build a jungle of trees from the training set (X, y)"""
		#TODO : add previous proba
		forests = []
		featureFunction = self.featureFunction
		for i in range(self.n_forests):
			if (i != 0):
				if(specialisation == 'global'):
					acc = forest.getFeatureImportance()
					featureFunction.random_weight = acc
				elif(specilisation =='per_class'):
					acc_per_class = forest.getFeatureImportanceByClass()
					featureFunction.random_weight_per_class = acc_per_class
				if(add_previous_prob):
					#TODO
					X=X
			forest = deepcopy(self.forest)
			forest.featureFunction = featureFunction
			forest.fit(X,y)
			forests.append(forest)

		# Collect newly grown Forests
		self.forests_.extend(forests)
	
	def fit_image(self, raster_data, sample_index, y, dview = None):
		"""Build a jungle of trees from the training set (X, y)"""
		#TODO  : add previous proba
		forests = []
		featureFunction = self.featureFunction
		for i in range(self.n_forests):
			if (i != 0):
				if(specialisation == 'global'):
					acc = forest.getFeatureImportance()
					featureFunction.random_weight = acc
				elif(specilisation =='per_class'):
					acc_per_class = forest.getFeatureImportanceByClass()
					featureFunction.random_weight_per_class = acc_per_class
				if(add_previous_prob):
					#TODO
					X=X
			forest = deepcopy(self.forest)
			forest.featureFunction = featureFunction
			forest.fit_image(raster_data, sample_index, y, dview)
			forests.append(forest)
		
		# Collect newly grown Forests
		self.forests_.extend(forests)

	def predict(self, X):
		"""Predict class for X."""
		
		proba = numpy.array(self.predict_proba(X))
		return self.classes.take(numpy.argmax(proba, axis=1), axis=0)


	def predict_proba(self, X):
		"""Predict class probabilities for X"""
		#TODO
		all_proba = []
		for i in range(self.n_forests):
			if ((i != 0) and add_previous_prob):
				#TODO
				X=X # + proba
			proba = self.forests_[i].predict_proba(X)
			all_proba.append(proba)
		
		if(fusion == "mean"):
			for j in range(1, len(all_proba)):
				proba += all_proba[j]
			return proba / self.n_forests
			
		else: # (fusion =="last"):
			return proba
		
		return proba

	def predict_image(self, imarray, w_x, w_y):
		"""Predict class for X."""
		proba = numpy.array(self.predict_proba_image(imarray, w_x, w_y))
		return self.classes.take(numpy.argmax(proba, axis=0))

	def predict_proba_image(self, imarray, w_x, w_y):
		"""Predict class probabilities for X"""
		#TODO
		all_proba = []
		for i in range(self.n_forests):
			if ((i != 0) and add_previous_prob):
				#TODO
				X=X # + proba
			proba = self.forests_[i].predict_proba_image(imarray)
			all_proba.append(proba)
		
		if(fusion == "mean"):
			for j in range(1, len(all_proba)):
				proba += all_proba[j]
			return proba / self.n_forests
			
		else: # (fusion =="last"):
			return proba
		
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
