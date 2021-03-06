import numpy
import random
import math
import itertools
import bisect
from osgeo import gdal

try:
	from string import Template
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
except ImportError:
    pass


# ======================================================================
# Class FeatureFunction
# ======================================================================
class FeatureFunction:
	"""Class FeatureFunction
		- option
		- nb_channels
		- width
		- height
	"""
 
	def __init__(self, nb_channels = 0, width = 0, height =0, option = {}, random_weight=None, random_weight_per_class=None, epsilon = 0.1):
		self.nb_channels = nb_channels
		self.width	 = (width//2)  * 2
		self.height  = (height//2) * 2
		self.option	 = option
		self.random_weight = random_weight
		self.random_weight_per_class = random_weight_per_class
		self.epsilon = epsilon

	def init(self, nb_channels,width,height):
		self.nb_channels = nb_channels
		self.width	 = (width//2)  * 2
		self.height  = (height//2) * 2

	def copy(self):
		return FeatureFunction(self.nb_channels,self.width,self.height,self.option)

	def evaluate(self,X):
		""" Evalute the feature on each sample of the dataset X """
		
		# get width, height, nb channels and nb samples from dataset X
		nb_samples , nb_channels, width, height  = X.shape

		if( self.nb_channels == nb_channels and  self.width == width and  self.height == height ):
			if self.option['type'] == 'RQE':
				return self.RQE(X)
		else:
			print(self.nb_channels, nb_channels)
			print(self.width, width)
			print(self.height, height)
			print("ERROR : X shape is not corresponding to the featureFunction shape ")

	def RQE(self,X):
		""" Compute the RQE feature
			option.type = RQE
			option.RQE.windows = []
				-windows[i]
				- Xmin, Ymin, Xmax, Ymax, Channel
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
		#return X[:,w['Channel'],w['Xmin']:w['Xmax'],w['Ymin']:w['Ymax']].mean(axis=1).mean(axis=1)
		dX = self.width//2
		dY = self.height//2

		a = X[:,w['Channel'],w['Xmin']+dX,w['Ymin']+dY]
		b = X[:,w['Channel'],w['Xmin']+dX,w['Ymax']+dY]
		c = X[:,w['Channel'],w['Xmax']+dX,w['Ymin']+dY]
		d = X[:,w['Channel'],w['Xmax']+dX,w['Ymax']+dY]
		return a - b - c + d

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

	def evaluate_image_cpu(self, X, sample_index):
		n_channels, w_d, h_d = X.shape
		w = self.width //2
		h = self.height //2

		Xt = numpy.zeros((1,n_channels,self.width,self.height))
		Xf = []
		for id in sample_index:
			i, j = id
			imin = max(0,i-w)
			if imin == 0:
				imax = 2*w
			else:
				imax = min(i+w,w_d)
				if imax == w_d:
					imin = w_d-2*w
			jmin = max(0,j-h)
			if jmin == 0:
				jmax = 2*h
			else:
				jmax = min(j+h,h_d)
				if jmax == h_d:
					jmin = h_d-2*h
			Xt[0] = X[:,imin:imax,jmin:jmax]
			r = self.evaluate(Xt)
			Xf.append(r)
		return Xf

	def evaluate_image_gpu(self, x_gpu, sample_index):
		func_mod_template = Template("""

			// Macro for converting subscripts to linear index:
			// A = nb_channels
			// B,C = image X,Y
			#define INDEX3D(a, b, c) a*${sizeX}*${sizeY}+b*${sizeY}+c

			struct Feature {
				unsigned int type;
				int c1,xm1,xM1,ym1,yM1;
				int c2,xm2,xM2,ym2,yM2;
			};

		__device__ float one(float *X, unsigned int idx, unsigned int idy , int c1,
								int dxm1, int dxM1,
								int dym1, int dyM1)
			{
				int xm1 = idx+dxm1;
				int xM1 = idx+dxM1;
				int ym1 = idy+dym1;
				int yM1 = idy+dyM1;
				
				xm1 = (xm1>=0)?xm1:0;
				ym1 = (ym1>=0)?ym1:0;
				
				xM1 = (xM1< ${sizeX})?xM1:(${sizeX}-1);
				yM1 = (yM1< ${sizeY})?yM1:(${sizeY}-1);
				
				float a = X[INDEX3D(c1,xm1,ym1)];
				float b = X[INDEX3D(c1,xM1,ym1)];
				float c = X[INDEX3D(c1,xm1,yM1)];
				float d = X[INDEX3D(c1,xM1,yM1)];
				return a - b - c + d;
			}

		__global__ void apply_feature(float *X, unsigned int* indexTab, Feature *f, float *Xf)
			{
				// Obtain the linear index corresponding to the current thread:
				unsigned int i = blockIdx.x*${n_threads_x}+threadIdx.x;
				if( i < ${n_samples} )
				{
					unsigned int idx = indexTab[2*i];
					unsigned int idy = indexTab[2*i+1];
					float r = 0;
					if(f->type == 0) // ONE
						r = one(X,idx, idy,f->c1,f->xm1,f->xM1,f->ym1,f->yM1);
					elif (f->type == 1) // SUM
						r = one(X,idx, idy,f->c1,f->xm1,f->xM1,f->ym1,f->yM1) + one(X,idx, idy,f->c2,f->xm2,f->xM2,f->ym2,f->yM2);
					elif (f->type == 2) // DIFF
						r = one(X,idx, idy,f->c1,f->xm1,f->xM1,f->ym1,f->yM1) - one(X,idx, idy,f->c2,f->xm2,f->xM2,f->ym2,f->yM2);
					elif (f->type == 3) // RATIO
					{
						r = one(X,idx, idy,f->c1,f->xm1,f->xM1,f->ym1,f->yM1) + one(X,idx, idy,f->c2,f->xm2,f->xM2,f->ym2,f->yM2);
						if(abs(r)>=0.00001)
							r = one(X,idx, idy,f->c1,f->xm1,f->xM1,f->ym1,f->yM1) - one(X,idx, idy,f->c2,f->xm2,f->xM2,f->ym2,f->yM2) / r;
						else
							r = 0;
					}
					Xf[i] = r;
				}
			}

			""")

		nbChannels, sizeX, sizeY = x_gpu.shape
		n_samples = len(sample_index)

		max_threads_per_block = pycuda.autoinit.device.MAX_THREADS_PER_BLOCK

		n_threads_x = min(max_threads_per_block,n_samples)
		n_blocks_x = sizeX // n_threads_x +1;

		block_dim=(n_threads_x,1,1)
		grid_dim=(n_blocks_x,1)
		print(block_dim,grid_dim)
		func_mod = SourceModule(
					func_mod_template.substitute(
						n_threads_x=n_threads_x,
						sizeX=sizeX, sizeY=sizeY,
						n_samples = n_samples)
					)

		apply_feature = func_mod.get_function('apply_feature')


		#Copy data to GPU
		id_gpu = gpuarray.to_gpu(numpy.asarray(sample_index).astype(numpy.uint32))
		xf_gpu = gpuarray.zeros(n_samples,numpy.float32)

		f_size = 44
		f_gpu = cuda.mem_alloc(f_size)
		self.toGPU(f_gpu,f_size)

		#apply_feature
		apply_feature(x_gpu, id_gpu, f_gpu, xf_gpu, block=block_dim, grid=grid_dim)

		xf = xf_gpu.get()
		del id_gpu
		del xf_gpu
		del f_gpu
		return xf

	def evaluate_image(self, X, sample_index, gpu = False):
		if(gpu):
			return self.evaluate_image_gpu(X, sample_index)
		else:
			return self.evaluate_image_cpu(X, sample_index)

	def toGPU(self, f_gpu, f_size):
		shift=0
		param = self.option['RQE']
		w = param['windows'][0]
		c1 = w['Channel']
		xm1 = w['Xmin']
		xM1 = w['Xmax']
		ym1 = w['Ymin']
		yM1 = w['Ymax']
		
		type = 0
		c2 = 0
		xm2 = 0
		xM2 = 0
		ym2 = 0
		yM2 = 0
		if param['type'] != "one":
			w = param['windows'][1]
			c2 = w['Channel']
			xm2 = w['Xmin']
			xM2 = w['Xmax']
			ym2 = w['Ymin']
			yM2 = w['Ymax']
			if param['type'] == 'sum':
				type = 1
			elif  param['type'] == 'diff':
				type = 2
			elif  param['type'] == 'ratio':
				type = 3
		b = memoryview(numpy.uint32(type)); 		cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(c1)); 				cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(xm1)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(xM1)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(ym1)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(yM1)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(c2)); 				cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(xm2)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(xM2)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(ym2)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		b = memoryview(numpy.int32(yM2)); 			cuda.memcpy_htod(int(f_gpu)+shift, b); shift+= len(b)
		assert(shift == f_size)

	def random(self, stats=None):

		w = self.width //2
		h = self.height //2
		epsilon = self.epsilon
		
		types=["one","diff","sum","ratio"]
		isDefault = True
		isPerClass = False
		
		if( (self.random_weight_per_class != None) and (stats != None) ):
			isPerClass = True
			isDefault = False
		elif( self.random_weight != None):
			acc = self.random_weight
			isDefault = False

		if (not isDefault):
			weights = acc['RQE']['type']
			if isPerClass:
				weights = weights*stats
				
			cum = numpy.sum(weights)
			if(cum!= 0 ):
				weights += epsilon * cum
			else:
				weights += epsilon

			cumdist = list(itertools.accumulate(weights))
				
			x = random.random() * cumdist[-1]
			t = types[bisect.bisect(cumdist, x)]
			
			weights = acc['RQE']['channel']
			if isPerClass:
				weights = weights*stats

			cum = numpy.sum(weights)
			if(cum!= 0 ):
				weights += epsilon * cum
			else:
				weights += epsilon
				
			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			Channel1 = bisect.bisect(cumdist, x)
			x = random.random() * cumdist[-1]
			Channel2 = bisect.bisect(cumdist, x)

			weights = acc['RQE']['windows_width']
			if isPerClass:
				weights = weights*stats

			cum = numpy.sum(weights)
			if(cum!= 0 ):
				weights += epsilon * cum
			else:
				weights += epsilon				

			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			w1 = bisect.bisect(cumdist, x) //2
			x = random.random() * cumdist[-1]
			w2 = bisect.bisect(cumdist, x) //2

			weights = acc['RQE']['windows_height']
			if isPerClass:
				weights = weights*stats

			cum = numpy.sum(weights)
			if(cum!= 0 ):
				weights += epsilon * cum
			else:
				weights += epsilon

			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			h1 = bisect.bisect(cumdist, x) //2
			x = random.random() * cumdist[-1]
			h2 = bisect.bisect(cumdist, x) //2

			weights = acc['RQE']['windows_dist']
			if isPerClass:
				weights = weights*stats

			cum = numpy.sum(weights)
			if(cum!= 0 ):
				weights += epsilon * cum
			else:
				weights += epsilon

			cumdist = list(itertools.accumulate(weights))
			x = random.random() * cumdist[-1]
			d1 = bisect.bisect(cumdist, x)
			x = random.random() * cumdist[-1]
			d2 = bisect.bisect(cumdist, x)

			theta1 = random.random() * 2*math.pi
			theta2 = random.random() * 2*math.pi

			x = math.floor(d1*math.cos(theta1))
			y = math.floor(d1*math.sin(theta1))
			
			Xmin1 = x - w1
			Xmax1 = Xmin1 + w1
			if(Xmin1 <= -w):
				Xmin1 = -w+1
				Xmax1 = Xmin1 + w1
			if(Xmax1 >= w):
				Xmax1 = w-1
				Xmin1 = Xmax1 - w1

			Ymin1 = y - h1
			Ymax1 = Ymin1 + h1
			if(Ymin1 <= -h):
				Ymin1 = -h+1
				Ymax1 = Ymin1 + h1
			if(Ymax1 >= h):
				Ymax1 = h-1
				Ymin1 = Ymax1 - h1

			x = math.floor(d1*math.cos(theta1))
			y = math.floor(d1*math.sin(theta1))
			
			Xmin2 = x - w2
			Xmax2 = Xmin2 + w2
			if(Xmin2 <= -w):
				Xmin2 = -w+1
				Xmax2 = Xmin2 + w2
			if(Xmax2 >= w):
				Xmax2 = w-1
				Xmin2 = Xmax2 - w2

			Ymin2 = y - h2
			Ymax2 = Ymin2 + h2
			if(Ymin2 <= -h):
				Ymin2 = -h+1
				Ymax2 = Ymin2 + h2
			if(Ymax2 >= h):
				Ymax2 = h-1
				Ymin2 = Ymax2 - h2

		else:
			t = random.choice(types)

			Xmin1 = random.randint(-w,w-2)
			Ymin1 = random.randint(-h,h-2)
			Xmax1 = random.randint(Xmin1+1, w-1)
			Ymax1 = random.randint(Ymin1+1, h-1)
			Channel1 = random.randint(0,self.nb_channels-1)

			Xmin2 = random.randint(-w,w-2)
			Ymin2 = random.randint(-h,h-2)
			Xmax2 = random.randint(Xmin2+1, w-1)
			Ymax2 = random.randint(Ymin2+1, h-1)
			Channel2 = random.randint(0,self.nb_channels-1)

		option={ 'type' : 'RQE', 'RQE' : { 'type' :  t,
				'windows' : [
					{ 'Xmin' : Xmin1,'Ymin' : Ymin1,'Xmax' : Xmax1,'Ymax' : Ymax1,'Channel' : Channel1},
					{ 'Xmin' : Xmin2,'Ymin' : Ymin2,'Xmax' : Xmax2,'Ymax' : Ymax2,'Channel' : Channel2},
				] } }
		self.option = option

	def getEmptyAccumulator(self):
		acc = {}
		#if self.option['type'] == 'RQE':
		acc['type'] = 'RQE'
		req = {}
		req['type'] = numpy.zeros(4)
		req['channel'] = numpy.zeros(self.nb_channels)
		req['windows_width'] = numpy.zeros(self.width)
		req['windows_height'] = numpy.zeros(self.height)
		dMax = int(math.ceil(math.sqrt(math.pow(self.width,2) + math.pow(self.height,2))))
		req['windows_dist'] = numpy.zeros(dMax)
		req['img'] = numpy.zeros((self.width,self.height,self.nb_channels))
		acc['RQE'] = req

		return acc		

	def getEmptyAccumulatorByClass(self,nbClasses):
		acc = {}
		#if self.option['type'] == 'RQE':
		acc['type'] = 'RQE'
		req = {}
		req['type'] = numpy.zeros((4,nbClasses))
		req['channel'] = numpy.zeros((self.nb_channels,nbClasses))
		req['windows_width'] = numpy.zeros((self.width,nbClasses))
		req['windows_height'] = numpy.zeros((self.height,nbClasses))
		dMax = int(math.ceil(math.sqrt(math.pow(self.width,2) + math.pow(self.height,2))))
		req['windows_dist'] = numpy.zeros((dMax,nbClasses))
		req['img'] = numpy.zeros((self.width,self.height,self.nb_channels,nbClasses))
		acc['RQE'] = req

		return acc

	def addFeatureImportance(self, improvement, acc):

		if self.option['type'] == 'RQE':
			param = self.option['RQE']
			w = param['windows'][0]
			c1 = w['Channel']
			xm1 = w['Xmin']
			xM1 = w['Xmax']
			ym1 = w['Ymin']
			yM1 = w['Ymax']

			dX = self.width//2
			dY = self.height//2
				
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
				acc['RQE']['windows_width'][xM1-xm1] += improvement
				acc['RQE']['windows_height'][yM1-ym1] += improvement
				d = int(math.floor(math.sqrt(math.pow(xM1+xm1,2) + math.pow(yM1+ym1,2))))
				acc['RQE']['windows_dist'][d] += improvement
				acc['RQE']['img'][xm1+dX:xM1+dX,ym1+dY:yM1+dY,c1] += improvement
			else:
				w = param['windows'][1]
				c2 = w['Channel']
				xm2 = w['Xmin']
				xM2 = w['Xmax']
				ym2 = w['Ymin']
				yM2 = w['Ymax']
				acc['RQE']['channel'][c1] += improvement/2
				acc['RQE']['channel'][c2] += improvement/2
				acc['RQE']['windows_width'][xM1-xm1] += improvement/2
				acc['RQE']['windows_width'][xM2-xm2] += improvement/2
				acc['RQE']['windows_height'][yM2-ym2] += improvement/2
				acc['RQE']['windows_height'][yM2-ym2] += improvement/2
				d = int(math.floor(math.sqrt(math.pow(xM1+xm1,2) + math.pow(yM1+ym1,2))))
				acc['RQE']['windows_dist'][d] += improvement/2
				d = int(math.floor(math.sqrt(math.pow(xM2+xm2,2) + math.pow(yM2+ym2,2))))
				acc['RQE']['windows_dist'][d] += improvement/2
				
				acc['RQE']['img'][xm1+dX:xM1+dX,ym1+dY:yM1+dY,c1] += improvement/2
				acc['RQE']['img'][xm2+dX:xM2+dX,ym2+dY:yM2+dY,c2] += improvement/2

	def addFeatureImportanceByClass(self, improvement, stats, acc):
		if self.option['type'] == 'RQE':
			param = self.option['RQE']
			w = param['windows'][0]
			c1 = w['Channel']
			xm1 = w['Xmin']
			xM1 = w['Xmax']
			ym1 = w['Ymin']
			yM1 = w['Ymax']
			
			dX = self.width//2
			dY = self.height//2
			
			type = 0
			if param['type'] == 'sum':
				type = 1
			elif  param['type'] == 'diff':
				type = 2
			elif  param['type'] == 'ratio':
				type = 3

			acc['RQE']['type'][type,:] += improvement*stats
	
			if type == 0:
				acc['RQE']['channel'][c1,:] += improvement*stats
				acc['RQE']['windows_width'][xM1-xm1,:] += improvement*stats
				acc['RQE']['windows_height'][yM1-ym1,:] += improvement*stats
				d = int(math.floor(math.sqrt(math.pow(xM1+xm1,2) + math.pow(yM1+ym1,2))))
				acc['RQE']['windows_dist'][d,:] += improvement*stats
				acc['RQE']['img'][xm1+dX:xM1+dX,ym1+dY:yM1+dY,c1,:] += improvement*stats
			else:
				w = param['windows'][1]
				c2 = w['Channel']
				xm2 = w['Xmin']
				xM2 = w['Xmax']
				ym2 = w['Ymin']
				yM2 = w['Ymax']
				acc['RQE']['channel'][c1,:] += improvement/2*stats
				acc['RQE']['channel'][c2,:] += improvement/2*stats
				acc['RQE']['windows_width'][xM1-xm1,:] += improvement/2*stats
				acc['RQE']['windows_width'][xM2-xm2,:] += improvement/2*stats
				acc['RQE']['windows_height'][yM2-ym2,:] += improvement/2*stats
				acc['RQE']['windows_height'][yM2-ym2,:] += improvement/2*stats
				d = int(math.floor(math.sqrt(math.pow(xM1+xm1,2) + math.pow(yM1+ym1,2))))
				acc['RQE']['windows_dist'][d,:] += improvement/2*stats
				d = int(math.floor(math.sqrt(math.pow(xM2+xm2,2) + math.pow(yM2+ym2,2))))
				acc['RQE']['windows_dist'][d,:] += improvement/2*stats
				acc['RQE']['img'][xm1+dX:xM1+dX,ym1+dY:yM1+dY,c1] += improvement/2
				acc['RQE']['img'][xm2+dX:xM2+dX,ym2+dY:yM2+dY,c2] += improvement/2

	def getSamplesDataFromImage(self, input_data, sample_index, cumul = True):
		if(type(input_data) is str):
			print("raster_data from filename")
			raster_data = gdal.Open(input_data)
		elif (type(input_data) is gdal.Dataset):
			print("raster_data from gdal raster")
			raster_data = input_data
		elif (type(input_data) is numpy.ndarray):
			print("Error format ndarray is no more support")
			return False
		else:
			print("Error input_data format unknown")
			return False

		block_sizes = raster_data.GetRasterBand(1).GetBlockSize()  
		x_block_size = block_sizes[0]  
		y_block_size = block_sizes[1]  
			
		xsize = raster_data.RasterXSize  
		ysize = raster_data.RasterYSize
		
		nb_samples = len(sample_index)
		X = numpy.zeros((nb_samples,self.nb_channels,self.width,self.height))

		w = self.width //2
		h = self.height //2
		for i_b in range(0, xsize, x_block_size):  
			if i_b + x_block_size < xsize:  
				x_block_size_tmp = x_block_size
			else:  
				x_block_size_tmp = xsize - i_b  
			for j_b in range(0, ysize, y_block_size):  
				if j_b + y_block_size < ysize:  
						y_block_size_tmp = y_block_size  
				else:
						y_block_size_tmp = ysize - j_b
				
				#Get sample in chunk
				list_id_in_chunk = numpy.where(
							(j_b - w <= sample_index[:,0]) &
							(sample_index[:,0] < j_b + y_block_size_tmp + w) &
							(i_b - h <= sample_index[:,1]) &
							(sample_index[:,1] < i_b + x_block_size_tmp + h)
						)[0]
				#
				if(len(list_id_in_chunk)):
					data_chunk = raster_data.ReadAsArray(i_b, j_b, x_block_size_tmp, y_block_size_tmp)
					for id in list_id_in_chunk:
						j, i = sample_index[id]
						
						imin_image = max(0,i-w)
						if imin_image == 0:
							imax_image = 2*w
						else:
							imax_image = min(i+w,xsize)
							if imax_image == xsize:
								imin_image = xsize-2*w

						imin_chunk = max(0, imin_image - i_b)
						imin_out   = max(0, i_b - imin_image)
						
						imax_chunk = min(x_block_size_tmp, max(1,imax_image - i_b))
						imax_out   = imin_out + imax_chunk - imin_chunk
						
						jmin_image = max(0,j-h)
						if jmin_image == 0:
							jmax_image = 2*h
						else:
							jmax_image = min(j+h,ysize)
							if jmax_image == ysize:
								jmin_image = ysize-2*h
						
						jmin_chunk = max(0, jmin_image - j_b)
						jmin_out   = max(0, j_b - jmin_image)
						
						jmax_chunk = min(y_block_size_tmp, max(1,jmax_image - j_b))
						jmax_out   = jmin_out + jmax_chunk - jmin_chunk

						X[id,:,jmin_out:jmax_out,imin_out:imax_out] = data_chunk[:,jmin_chunk:jmax_chunk,imin_chunk:imax_chunk]
		if cumul :
			return X.cumsum(3).cumsum(2)
		else :
			return X
					

	def __repr__(self):
		return "FeatureFunction " + str(self.nb_channels) + " " +str(self.width) + " " +str(self.height) + " " + self.option.__repr__()
