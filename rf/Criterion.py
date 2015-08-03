import numpy as np
from math import log

# =============================================================================
# Criterion
# =============================================================================

class Criterion:
	"""Interface for impurity criteria."""

	def init(self, y, sample_weight, weighted_n_samples,  samples, start, end):
		"""Initialize the criterion at node samples[start:end] and
		   children samples[start:start] and samples[start:end]."""
		pass

	def reset(self):
		"""Reset the criterion at pos=start."""
		pass

	def update(self, new_pos):
		"""Update the collected statistics by moving samples[pos:new_pos] from
		   the right child to the left child."""
		pass

	def node_impurity(self):
		"""Evaluate the impurity of the current node, i.e. the impurity of
		   samples[start:end]."""
		pass

	def children_impurity(self):
		"""Evaluate the impurity in children nodes, i.e. the impurity of
		   samples[start:pos] + the impurity of samples[pos:end]."""
		pass

	def impurity_improvement(self, impurity):
		"""Weighted impurity improvement, i.e.

		   N_t / N * (impurity - N_t_L / N_t * left impurity
							   - N_t_L / N_t * right impurity),

		   where N is the total number of samples, N_t is the number of samples
		   in the current node, N_t_L is the number of samples in the left
		   child and N_t_R is the number of samples in the right child."""

		impurity_left, impurity_right = self.children_impurity()
		#return (impurity -  impurity_right -  impurity_left)
		return ((self.weighted_n_node_samples / self.weighted_n_samples) *
					(impurity - self.weighted_n_right / self.weighted_n_node_samples * impurity_right
					- self.weighted_n_left / self.weighted_n_node_samples * impurity_left))


class ClassificationCriterion(Criterion):
	"""Abstract criterion for classification."""

	def __init__(self):
		# Default values
		self.y = []
		self.sample_weight = None
		self.samples = []
		self.start = 0
		self.pos = 0
		self.end = 0

		self.n_node_samples = 0
		self.weighted_n_node_samples = 0.0
		self.weighted_n_left = 0.0
		self.weighted_n_right = 0.0

		self.label_count_total 	= []
		self.label_count_left 	= []
		self.label_count_right 	= []

		# Count labels
		self.n_classes = 0


	def init(self, y, sample_weight, weighted_n_samples, samples, start, end):
		"""Initialize the criterion at node samples[start:end] and
		   children samples[start:start] and samples[start:end]."""
		# Initialize fields
		self.y = y
		self.sample_weight = sample_weight
		self.samples = samples
		self.start = start
		self.end = end
		self.n_node_samples = end - start
		
		self.weighted_n_samples = weighted_n_samples
		
		
		#self.n_classes = len(set(y))
		self.n_classes = max(y)+1

		self.label_count_total 	= np.zeros(self.n_classes)
		self.label_count_left 	= np.zeros(self.n_classes)
		self.label_count_right 	= np.zeros(self.n_classes)

		# Initialize label_count_total
		weighted_n_node_samples = 0.0
		w = 1.0
		for p in range(start, end):
			i = samples[p]
			c = y[i]
			if sample_weight is not None:
				w = sample_weight[i]
			self.label_count_total[c] += w
			weighted_n_node_samples += w
 
		self.weighted_n_node_samples = weighted_n_node_samples
		
		# Reset to pos=start
		self.reset()

	def reset(self):
		"""Reset the criterion at pos=start."""
		self.pos = self.start

		self.weighted_n_left = 0.0
		self.weighted_n_right = self.weighted_n_node_samples

		self.label_count_left 	= np.zeros(self.n_classes)
		self.label_count_right 	= np.copy(self.label_count_total)


	def update(self, new_pos):
		"""Update the collected statistics by moving samples[pos:new_pos]
			from the right child to the left child."""

		w = 1.0
		diff_w = 0.0

		# Note: We assume start <= pos < new_pos <= end
		for p in range(self.pos, new_pos):
			i = self.samples[p]

			if self.sample_weight is not None:
				w = self.sample_weight[i]

			self.label_count_left[self.y[i]]  += w
			self.label_count_right[self.y[i]] -= w

			diff_w += w

		self.weighted_n_left += diff_w
		self.weighted_n_right -= diff_w

		self.pos = new_pos

	def node_impurity(self):
		pass

	def children_impurity(self):
		pass
	
	def get_stats(self):
		return self.label_count_total/ self.weighted_n_node_samples

	def get_stats_left(self):
		return self.label_count_left/self.weighted_n_left

	def get_stats_right(self):
		return self.label_count_right/self.weighted_n_right

class Entropy(ClassificationCriterion):
	"""Cross Entropy impurity criteria.

	Let the target be a classification outcome taking values in 0, 1, ..., K-1.
	If node m represents a region Rm with Nm observations, then let

		pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

	be the proportion of class k observations in node m.

	The cross-entropy is then defined as

		cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
	"""
	def node_impurity(self):
		"""Evaluate the impurity of the current node, i.e. the impurity of samples[start:end]."""

		entropy = 0.0
		for c in range(self.n_classes):
			tmp = self.label_count_total[c]
			if tmp > 0.0:
				tmp /= self.weighted_n_node_samples
				entropy -= tmp * log(tmp)

		return entropy

	def children_impurity(self):
		"""Evaluate the impurity in children nodes, i.e. the impurity of the
		   left child (samples[start:pos]) and the impurity the right child
		   (samples[pos:end])."""

		entropy_left = 0.0
		entropy_right = 0.0

		for c in range(self.n_classes):
			tmp = self.label_count_left[c]
			if tmp > 0.0:
				tmp /= self.weighted_n_left
				entropy_left -= tmp * log(tmp)

			tmp = self.label_count_right[c]
			if tmp > 0.0:
				tmp /= self.weighted_n_right
				entropy_right -= tmp * log(tmp)

		return entropy_left, entropy_right


class Gini(ClassificationCriterion):
	"""Gini Index impurity criteria.

	Let the target be a classification outcome taking values in 0, 1, ..., K-1.
	If node m represents a region Rm with Nm observations, then let

		pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

	be the proportion of class k observations in node m.

	The Gini Index is then defined as:

		index = \sum_{k=0}^{K-1} pmk (1 - pmk)
			  = 1 - \sum_{k=0}^{K-1} pmk ** 2
	"""
	def node_impurity(self):
		"""Evaluate the impurity of the current node, i.e. the impurity of
		   samples[start:end]."""
		
		gini = 0.0
		total = 0.0

		for c in range(self.n_classes):
			tmp = self.label_count_total[c]
			gini += tmp * tmp

		gini = 1.0 - gini / (self.weighted_n_node_samples *
							 self.weighted_n_node_samples)

		return gini

	def children_impurity(self):
		"""Evaluate the impurity in children nodes, i.e. the impurity of the
		   left child (samples[start:pos]) and the impurity the right child
		   (samples[pos:end])."""

		gini_left = 0.0
		gini_right = 0.0
		total = 0.0
		total_left = 0.0
		total_right = 0.0

		for c in range(self.n_classes):
			tmp = self.label_count_left[c]
			gini_left += tmp * tmp
			tmp = self.label_count_right[c]
			gini_right += tmp * tmp

		gini_left = 1.0 - gini_left / (self.weighted_n_left *
									   self.weighted_n_left)
		gini_right = 1.0 - gini_right / (self.weighted_n_right *
										 self.weighted_n_right)

		return gini_left, gini_right
