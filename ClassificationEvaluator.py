import numpy as np

class ClassificationEvaluator():
	""" ClassificationEvaluator """
	def __init__(self, classif, ground_truth):
		""" init"""

		self.is_init = False
		self.w_d, self.h_d = classif.shape
		self.w_l, self.h_l = ground_truth.shape
		self.classes_freq=[]
		self.classes_labels=[]
		self.mat_conf = None

		if(not ((self.w_d == self.w_l) and (self.h_d == self.h_l))):
			print("classif and gt must have the same size")
		else:
			classes_freq = np.bincount(ground_truth.ravel())

			#We assume that 0 is the no data classe
			self.nb_classes = len(classes_freq)-1
			self.classes_labels = list(range(1,self.nb_classes))
			self.mat_conf = np.zeros((self.nb_classes,self.nb_classes),dtype=np.int)
			#We assume that 0 is the no data classe
			self.classes_freq = np.asarray(classes_freq[1:])
			print("Classes labels :",self.classes_labels)
			print("Classes frequencies :",self.classes_freq)
			self.classif = classif
			self.gt = ground_truth
			self.is_init = True



	def computeStat(self, downscale = 1):
		for i in range(0,self.w_d,downscale):
			for j in range(0,self.h_d,downscale):
				c = self.classif[i,j]
				g = self.gt[i,j] -1
				if(g != 0):
					self.mat_conf[c,g] +=1
		print(self.mat_conf);
				
