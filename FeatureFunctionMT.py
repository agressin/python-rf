import numpy
import random
import math
import itertools
import bisect

from string import Template
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# ======================================================================
# Class FeatureFunctionMT
# ======================================================================
class FeatureFunctionMT:
	"""Class FeatureFunctionMT
		- option
		- nb_channels
	"""
 
	def __init__(self, nb_channels = 0, option = {}, random_weight=None):
		
		self.nb_channels = nb_channels
		self.option	 = option
		self.random_weight = random_weight

	def init(self, nb_channels):
		self.nb_channels = nb_channels

	def copy(self):
		return FeatureFunctionMT(self.nb_channels,self.option)

	def evaluate(self,X):
		""" Evalute the feature on each sample of the dataset X """
		
		# get width, height, nb channels and nb samples from dataset X
		nb_samples , nb_channels  = X.shape

		if( self.nb_channels == nb_channels):
			if self.option['type'] == 'RQE':
				return self.RQE(X)
		else:
			print(self.nb_channels, nb_channels)
			print("ERROR : X shape is not corresponding to the FeatureFunctionMT shape ")

	def RQE(self,X):
		""" Compute the RQE feature
			option.type = RQE
			option.RQE.windows = []
				-windows[i]
				- X, Y, Channel
		"""
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
			return self._one(X,w1)

	def _one(self,X,w):
		""" Compute the mean value on windows w """

		return X[:,w['Channel']]

	def _diff(self,X,w1,w2):
		""" Compute the difference of the mean value on two windows """
		return self._one(X,w1) - self._one(X,w2)

	def _sum(self,X,w1,w2):
		""" Compute the sum of the mean value on two windows """
		return self._one(X,w1) + self._one(X,w2)

	def _ratio(self,X,w1,w2):
		""" Compute the sum of the mean value on two windows """
		s = self._sum(X,w1,w2)
		s[s == 0.0] = 1.0
		return self._diff(X,w1,w2) / s 

	def toGPU(self, f_gpu, f_size):
		shift=0
		param = self.option['RQE']
		w = param['windows'][0]
		c1 = w['Channel']
		
		type = 0
		c2 = 0

		if param['type'] != "one":
			w = param['windows'][1]
			c2 = w['Channel']
			if param['type'] == 'sum':
				type = 1
			elif  param['type'] == 'diff':
				type = 2
			elif  param['type'] == 'ratio':
				type = 3
		b = memoryview(numpy.uint32(type)); 		cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(c1)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(c2)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		assert(shift == f_size)

	def random(self):

		types=["one","diff","sum","ratio"]

		if(self.random_weight == None):

			t= random.choice(types)
			Channel1 = random.randint(0,self.nb_channels-1)
			Channel2 = random.randint(0,self.nb_channels-1)

		else:

			acc = self.random_weight
			weights = acc['RQE']['type']
			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			t = types[bisect.bisect(cumdist, x)]

			weights = acc['RQE']['channel']
			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			Channel1 = bisect.bisect(cumdist, x)
			x = random.random() * cumdist[-1]
			Channel2 = bisect.bisect(cumdist, x)


		option={ 'type' : 'RQE', 'RQE' : { 'type' :  t,
					'windows' : [
						{ 'Channel' : Channel1},
						{ 'Channel' : Channel2},
					]
				}
			}
		self.option = option

	def getEmptyAccumulator(self):
		acc = {}
		#if self.option['type'] == 'RQE':
		acc['type'] = 'RQE'
		req = {}
		req['type'] = numpy.zeros(4)
		req['channel'] = numpy.zeros(self.nb_channels)
		acc['RQE'] = req

		return acc

	def addFeatureImportance(self, improvement, acc):

		if self.option['type'] == 'RQE':
			param = self.option['RQE']
			w = param['windows'][0]
			c1 = w['Channel']
			
			type = 0
			if param['type'] == 'sum':
				type = 1
			elif  param['type'] == 'diff':
				type = 2
			elif  param['type'] == 'ratio':
				type = 3

			acc['RQE']['type'][type] += improvement
	
			if type == 0:
				acc['RQE']['channel'][c1] += improvement
			else:
				w = param['windows'][1]
				c2 = w['Channel']
				acc['RQE']['channel'][c1] += improvement/2
				acc['RQE']['channel'][c2] += improvement/2

	def __repr__(self):
		return "FeatureFunctionMT " + str(self.nb_channels) + " " + " " + self.option.__repr__()
