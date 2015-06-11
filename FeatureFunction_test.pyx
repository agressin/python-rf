import numpy as np
cimport numpy as np
import random

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# ======================================================================
# Class FeatureFunction
#
# TODO :
#  ou est-ce qu'on fixe les param : self.nb_samples , self.nb_channels, self.width, self.height  = X.shape
#  supprimer les bandes inutiles avant de les passer aux fonctions one, diff, ...
#  stocker X dans une variable GPU ?
#  faire les calculs sur le GPU directement ?
# ======================================================================
cdef class FeatureFunction:
	"""Class FeatureFunction
		- option
		- nb_channels
		- width
		- height
	"""
	cdef int nb_channels, width, height
	cdef dict option

	def __init__(self, nb_channels =0, width=0, height=0, option = {}):
		self.nb_channels = nb_channels
		self.width	 = width
		self.height  = height
		self.option  = option

	def copy(self):
		return FeatureFunction(self.nb_channels, self.width, self.height, self.option )
	
	def init(self, int nb_channels, int width, int height):
		self.nb_channels = nb_channels
		self.width	 = width
		self.height  = height

	cpdef np.ndarray evaluate(self, np.ndarray X):
		""" Evalute the feature on each sample of the dataset X """
		
		# get width, height, nb channels and nb samples from dataset X
		cdef int nb_samples 	= X.shape[0]
		cdef int nb_channels 	= X.shape[1]
		cdef int width 			= X.shape[2]
		cdef int height 		= X.shape[3]
		if( self.nb_channels == nb_channels and  self.width == width and  self.height == height ):
			if self.option['type'] == 'RQE':
				return self.RQE(X)
		else:
			print("ERROR : X shape is not corresponding to the featureFunction shape ")

	cpdef np.ndarray RQE(self, np.ndarray X):
		""" Compute the RQE feature
			option.type = RQE
			option.RQE.windows = []
				-windows[i]
				- Xmin, Ymin, Xmax, Ymax, Channel
		"""
		cdef dict w1,w2
		# get RQE param from option
		param = self.option['RQE']
		w1 = param['windows'][0]
		if param['type'] != "one":
			w2 = param['windows'][1]
			if param['type'] == 'sum':
				return self._sum(X,w1,w2)
			elif  param['type'] == 'diff':
				return self._diff(X,w1,w2)
			elif  param['type'] == 'ratio':
				return self._ratio(X,w1,w2)
		else:
			return self._one(X, w1)

	cpdef np.ndarray _one(self,np.ndarray X, dict w):
		""" Compute the mean value on windows w """
		return X[:,w['Channel'],w['Xmin']:w['Xmax'],w['Ymin']:w['Ymax']].mean(axis=1).mean(axis=1)

	cpdef np.ndarray _diff(self,np.ndarray X, dict w1, dict w2):
		""" Compute the difference of the mean value on two windows """
		return self._one(X,w1) - self._one(X,w2)

	cpdef np.ndarray _sum(self,np.ndarray X, dict w1, dict w2):
		""" Compute the sum of the mean value on two windows """
		return self._one(X,w1) + self._one(X,w2)

	cdef np.ndarray _ratio(self,np.ndarray X, dict w1, dict w2):
		""" Compute the sum of the mean value on two windows """
		s = self._sum(X,w1,w2)
		s[s == 0.0] = 1.0
		return self._diff(X,w1,w2) / s 

	def random(self):
		# get width, height, nb channels and nb samples from dataset X
		#nb_samples , self.nb_channels, self.width, self.height  = X.shape
		types=["one","diff","sum","ratio"]
		Xmin1 = random.randint(0, self.width-1)
		Ymin1 = random.randint(0, self.height-1)
		Xmax1 = random.randint(Xmin1+1, self.width)
		Ymax1 = random.randint(Ymin1+1, self.height)
		Channel1 = random.randint(0,self.nb_channels-1)
		Xmin2 = random.randint(0, self.width-1)
		Ymin2 = random.randint(0, self.height-1)
		Xmax2 = random.randint(Xmin2+1, self.width)
		Ymax2 = random.randint(Ymin2+1, self.height)
		Channel2 = random.randint(0,self.nb_channels-1)
		t= random.choice(types)

		option={ 'type' : 'RQE', 'RQE' : { 'type' :  t,
					'windows' : [
						{ 'Xmin' : Xmin1,'Ymin' : Ymin1,'Xmax' : Xmax1,'Ymax' : Ymax1,'Channel' : Channel1},
						{ 'Xmin' : Xmin2,'Ymin' : Ymin2,'Xmax' : Xmax2,'Ymax' : Ymax2,'Channel' : Channel2},
					]
				}
			}
		self.option = option

	def __repr__(self):
		return "FeatureFunction " +  self.option.__repr__()
