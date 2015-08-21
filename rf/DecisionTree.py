import numpy as np

from .tree import Tree
from .Splitter import SplitRecord, Splitter
from .FeatureFunction import FeatureFunction
from .Criterion import Entropy, Gini

# ======================================================================
# Class myDecisionTreeClassifier
# ======================================================================
class myDecisionTreeClassifier:
	"""Class Tree """

	def __init__(self, max_depth = None,
				min_samples_split = 2,
				max_features = 10,
				min_samples_leaf = 1,
				max_leaf_nodes = -1,
				criterion = Gini(),
				featureFunction = FeatureFunction(),
				entangled = None):
		""" init """

		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.max_features = max_features
		self.min_samples_leaf = min_samples_leaf
		self.max_leaf_nodes = max_leaf_nodes
		self.featureFunction = featureFunction
		self.criterion = criterion
		self.entangled = entangled

	def fit(self, X, y, sample_weight=None):
		"""Build a decision tree from the training set (X, y)."""

		#Get classes
		tmp = list(set(y))
		self.classes = np.array(tmp)
		self.n_classes = len(tmp)
		
		max_depth = ((2 ** 31) - 1 if self.max_depth is None else self.max_depth)

		#self.criterion = Entropy(self.n_classes)
		self.splitter = Splitter(self.criterion, self.max_features, self.min_samples_leaf, self.featureFunction)


		# build tree
		# Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
		self.tree = Tree(self.featureFunction,self.splitter, self.min_samples_split,
										   self.min_samples_leaf,
										   max_depth,
										   self.n_classes,
										   self.max_leaf_nodes)

		if self.max_leaf_nodes < 0:
			self.tree.depthFirstTreeBuilder(X,y,sample_weight)
		else:
			self.tree.bestFirstTreeBuilder(X,y,sample_weight)

		self.tree.release()
		return self

	def fit_image(self, raster_data, sample_index, y, sample_weight=None):
		"""Build a decision tree from one image."""

		#Get classes
		tmp = list(set(y))
		self.classes = np.array(tmp)
		self.n_classes = len(tmp)
		
		max_depth = ((2 ** 31) - 1 if self.max_depth is None else self.max_depth)

		#self.criterion = Entropy(self.n_classes)
		self.splitter = Splitter(self.criterion, self.max_features, self.min_samples_leaf, self.featureFunction)

		# build tree
		# Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
		self.tree = Tree(self.featureFunction,self.splitter, self.min_samples_split,
										   self.min_samples_leaf,
										   max_depth,
										   self.n_classes,
										   self.max_leaf_nodes)
		
		if (type(raster_data) is np.ndarray):
			if self.entangled is not None :
				self.tree.entangledTreeBuilderImage(raster_data, sample_index,y,sample_weight,self.entangled)
			elif self.max_leaf_nodes < 0:
				self.tree.depthFirstTreeBuilderImage(raster_data, sample_index,y,sample_weight)
			else:
				self.tree.bestFirstTreeBuilderImage(raster_data, sample_index,y,sample_weight)
		else:
			X = self.featureFunction.getSamplesDataFromImage(raster_data, sample_index)
			if self.max_leaf_nodes < 0:
				self.tree.depthFirstTreeBuilder(X,y,sample_weight)
			else:
				self.tree.bestFirstTreeBuilder(X,y,sample_weight)

		self.tree.release()
		return self

	def predict(self, X):
		proba = np.array(self.tree.predict(X))
		return self.classes.take(np.argmax(proba, axis=1), axis=0)

	def predict_proba(self, X):
		proba = self.tree.predict(X)
		proba = proba[:, :self.n_classes]
		normalizer = proba.sum(axis=1)[:, np.newaxis]
		normalizer[normalizer == 0.0] = 1.0
		proba /= normalizer

		return proba

	def predict_image(self, imarray, w_x, w_y):
		proba = np.array(self.tree.predict_image(imarray, w_x, w_y))
		return self.classes.take(np.argmax(proba, axis=2))

	def predict_proba_image(self,x_gpu,proba_gpu, w_x, w_y):
		self.tree.predict_image(x_gpu,proba_gpu, w_x, w_y, entangled = self.entangled)

	def getFeatureImportance(self, acc = None):
		return self.tree.getFeatureImportance(acc)

	def getFeatureImportanceByClass(self, acc = None):
		return self.tree.getFeatureImportanceByClass(acc)

	def __repr__(self):
		return "DecisionTree, tree : \r\n"+self.tree.__repr__()
