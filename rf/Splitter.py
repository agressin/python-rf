import numpy as np
from copy import deepcopy, copy

FEATURE_THRESHOLD = 1e-7

# ======================================================================
#  Splitter
# ======================================================================
class SplitRecord:
	"""Class SplitRecord"""
	
	def __init__(self,pos=0):
		""" init """
		# Data to track sample split
		self.feature = 0		 		# Which feature to split on.
		self.pos = pos					# Split samples array at the given position,
										# i.e. count of samples below threshold for feature.
										# pos is >= end if the node is a leaf.
		self.threshold = 0.	  			# Threshold to split at.
		self.improvement = -np.inf 		# Impurity improvement given parent node.
		self.impurity_left = np.inf  	# Impurity of the left split.
		self.impurity_right = np.inf	# Impurity of the right split.
		self.values = []
		self.stats = []
		self.stats_left = []
		self.stats_right = []


class Splitter:
	"""Class Splitter"""

	def __init__(self, criterion, max_features, min_samples_leaf, featureFunction):
		""" init """
		self.criterion = criterion
		self.featureFunction = featureFunction

		self.samples = []
		self.n_samples = 0
		self.feature_values = []

		self.X = []
		self.y = []
		self.max_features = max_features
		self.min_samples_leaf = min_samples_leaf

		self.start = 0
		self.end = 0

		self.gpu_mode = False
		self.image_mode = False

	def init(self, X, y, sample_weight=None):
		"""Initialize the splitter."""

		# Initialize samples and features structures
		self.n_samples = X.shape[0]
		self.end = self.n_samples
		
		#self.samples = list(range(self.n_samples))
		self.samples = []

		weighted_n_samples = 0.0
		for i in range(self.n_samples):
			# Only work with positively weighted samples
			if (sample_weight is None) or (sample_weight[i] != 0.0):
				self.samples.append(i)

			if (sample_weight is not None):
				weighted_n_samples += sample_weight[i]
			else:
				weighted_n_samples += 1.0

		self.n_samples = len(self.samples)
		self.weighted_n_samples = weighted_n_samples
		
		# Initialize X, y, sample_weight
		self.X = X
		self.y = y
		self.sample_weight = sample_weight
		
		#if(len(X.shape) == 3):
		#	self.featureFunction.init(X.shape[1], X.shape[2], X.shape[3])
		#else:
		#	self.featureFunction.init(X.shape[1])

	def init_image(self, X, sample_index, y, sample_weight=None):
		"""Initialize the splitter."""

		# Initialize samples and features structures
		self.n_samples = sample_index.shape[0]
		self.end = self.n_samples
		
		#self.samples = list(range(self.n_samples))
		self.samples = []

		weighted_n_samples = 0.0
		for i in range(self.n_samples):
			# Only work with positively weighted samples
			if (sample_weight is None) or (sample_weight[i] != 0.0):
				self.samples.append(i)

			if (sample_weight is not None):
				weighted_n_samples += sample_weight[i]
			else:
				weighted_n_samples += 1.0

		self.n_samples = len(self.samples)
		self.weighted_n_samples = weighted_n_samples
		
		# Initialize X, y, sample_weight
		self.X = X
		self.sample_index = sample_index
		self.y = y
		self.sample_weight = sample_weight

		self.image_mode = True


	def node_reset(self, start, end):
		""" node_reset """
		self.start = start
		self.end = end
		self.criterion.init(self.y,
				self.sample_weight,
				self.weighted_n_samples,
				self.samples,
				start,
				end)


	def node_split(self, impurity):
		"""Find the best split on node samples[start:end]. (from BestSplitter)"""
		# Find the best split
		samples = self.samples
		start = self.start
		end = self.end


		X = self.X
		if(self.image_mode):
			sample_index = self.sample_index

		max_features = self.max_features
		min_samples_leaf = self.min_samples_leaf

		n_visited_features = 0
		n_rerun = 0
		max_rerun = 1

		best =  SplitRecord(end)
		current = deepcopy(best)

		# Sample up to max_features
		while (n_visited_features < max_features):

			n_visited_features += 1
			
			if ( (n_visited_features == max_features)
					and (best.improvement < 0)
					and (n_rerun < max_rerun) ):
				n_visited_features = 0
				n_rerun += 1
				if( n_rerun == max_rerun):
					print("Splitter : max_rerun", n_rerun)
			
			# Draw a feature at random
			self.featureFunction.random()

			current.feature = self.featureFunction.copy()

			if(self.image_mode):
				Xf = current.feature.evaluate_image(X, sample_index[samples[start:end]])
			else:
				# Compute feature on each sample
				Xtemp = deepcopy(X[start:end])
				for p in range(start, end):
					Xtemp[p-start]= (X[samples[p]])
				Xf = current.feature.evaluate(Xtemp)

			# Sort feature and mv X element at the same time in samples array
			samples[start:end] = [x for (y,x) in sorted(zip(Xf,samples[start:end]))]
			Xf.sort()

			# Evaluate all splits
			self.criterion.reset()
			p = start

			while p < end:
				while (p + 1 < end and Xf[p + 1 - start] <= Xf[p- start] + FEATURE_THRESHOLD):
					p += 1

				# (p + 1 >= end) or (X[samples[p + 1], current.feature] >
				#					X[samples[p], current.feature])
				p += 1
				# (p >= end) or (X[samples[p], current.feature] >
				#				X[samples[p - 1], current.feature])

				if p < end:
					current.pos = p

					# Reject if min_samples_leaf is not guaranteed
					if (((current.pos - start) < min_samples_leaf) or
							((end - current.pos) < min_samples_leaf)):
						continue

					self.criterion.update(current.pos)

					current.improvement = self.criterion.impurity_improvement(impurity)
					
					if current.improvement > best.improvement:
						current.values = Xf
						current.stats = self.criterion.get_stats()
						current.stats_right = self.criterion.get_stats_right()
						current.stats_left = self.criterion.get_stats_left()
						
						current.impurity_left,current.impurity_right = self.criterion.children_impurity()
						current.threshold = (Xf[p - 1 - start] + Xf[p - start]) / 2.0

						if current.threshold == Xf[p - start]:
							current.threshold = Xf[p - 1 - start]

						best = deepcopy(current)  # copy

		# Reorganize into samples[start:best.pos] + samples[best.pos:end]
		if best.pos < end:
			partition_end = end
			p = start

			Xf = best.values

			while p < partition_end:
				if Xf[p-start] <= best.threshold:
					p += 1

				else:
					partition_end -= 1

					tmp = samples[partition_end]
					samples[partition_end] = samples[p]
					samples[p] = tmp
		self.samples = samples
		# Return values
		return best

	def node_impurity(self):
		""" node_impurity """
		return self.criterion.node_impurity()
