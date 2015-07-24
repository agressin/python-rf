import numpy as np
import random
from osgeo import gdal

class TrainSamplesGenerator():
	""" TrainSampleGenerator """
	def __init__(self, input_data, input_label):
		""" init"""
		self.is_init = False
		
		if(type(input_data) is str):
			print("raster_data from filename")
			raster_data = gdal.Open(input_data)
		elif (type(input_data) is gdal.Dataset):
			print("raster_data from gdal raster")
			raster_data = input_data
		elif (type(input_data) is np.ndarray):
			print("Error format ndarray is no more support")
			return False
	
		if(type(input_label) is str):
			print("raster_label from filename")
			raster_label = gdal.Open(input_label)
		elif (type(input_label) is gdal.Dataset):
			print("raster_label from gdal raster")
			raster_label = input_label
		elif (type(input_label) is np.ndarray):
			print("Error format ndarray is no more support")
			return False
		
		self.raster_data = raster_data
		self.raster_label = raster_label
		
		self.c_d, self.w_d, self.h_d = self.raster_data.RasterCount, self.raster_data.RasterXSize, self.raster_data.RasterYSize
		self.w_l, self.h_l = self.raster_label.RasterXSize, self.raster_label.RasterYSize
		self.data_cumul = None
		self.classes_freq=[]
		self.classes_labels=[]

		if(not ((self.w_d == self.w_l) and (self.h_d == self.h_l))):
			print("data and label must have the same size")
		else:
			#Get number of pixel per classes (in the label image)
			self.label_band = self.raster_label.GetRasterBand(1)
			hist = np.array(self.label_band.GetHistogram(approx_ok = 0))
			nz = np.nonzero(hist)[0].tolist()
			classes_freq = hist[nz]

			#We assume that 0 is the no data classe
			self.classes_labels = list(range(1,len(classes_freq)))
			#We assume that 0 is the no data classe
			self.classes_freq = np.asarray(classes_freq[1:])
			print("Classes labels :",self.classes_labels)
			print("Classes frequencies :",self.classes_freq)
			self.is_init = True

	def getSamples(self, nb_samples=100, windows_w = 10, windows_h = 10, integral = True,  downscale = 1):
		#TODO adapt to raster (gdal)
		if(self.is_init and (self.data is not None) and (self.label is not None) ):
			nb_samples = min(min(self.classes_freq),nb_samples)
			print("nb_samples : ", nb_samples)
			prob_classes = nb_samples*downscale*downscale / self.classes_freq

			Xtmp=[]
			ytmp=[]
			for i in range(0,self.w_d,downscale):
				for j in range(0,self.h_d,downscale):
					c = self.label[i,j] 
					c_index = self.classes_labels.index(c) if c in self.classes_labels else -1
					if (c_index !=-1):
						if( random.random() <= prob_classes[c_index]):
							ytmp.append(c_index)
							if( (windows_w == 1) and (windows_h == 1) ):
								Xtmp.append(self.data[:,i,j])
							else:
								imin = max(0,i-windows_w)
								if imin == 0:
									imax = 2*windows_w
								else:
									imax = min(i+windows_w,self.w_d)
									if imax == self.w_d:
										imin = self.w_d-2*windows_w
								jmin = max(0,j-windows_h)
								if jmin == 0:
									jmax = 2*windows_h
								else:
									jmax = min(j+windows_h,self.h_d)
									if jmax == self.h_d:
										jmin = self.h_d-2*windows_h
								
						
								tt = self.data[:,imin:imax,jmin:jmax]
								if(integral):
									Xtmp.append(tt.cumsum(2).cumsum(1))
								else:
									Xtmp.append(tt)

			print(np.bincount(ytmp))

			nb_samples = len(ytmp)
			if( (windows_w == 1) and (windows_h == 1) ):
				X = np.zeros(nb_samples*self.c_d).reshape(nb_samples,self.c_d)			
			else:
				X = np.zeros(nb_samples*self.c_d*windows_w*2*windows_h*2).reshape(nb_samples,self.c_d,windows_w*2,windows_h*2)

			for i in range(0,nb_samples):
				X[i] = Xtmp[i]
			y = np.array(ytmp)
			
			return X,y
		else:
			print("An error occure during TrainSamplesGenerator initialisation")

	def getSamplesImages (self, nb_samples=100,  downscale = 1):
		if(self.is_init):
			nb_samples = min(min(self.classes_freq),nb_samples)
			print("nb_samples : ", nb_samples)
			prob_classes = nb_samples*downscale*downscale / self.classes_freq

			index_tmp=[]
			ytmp=[]

			block_sizes = self.label_band.GetBlockSize()  
			x_block_size = block_sizes[0]  
			y_block_size = block_sizes[1]  
				
			xsize = self.label_band.XSize  
			ysize = self.label_band.YSize

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
			
					label_chunk = self.label_band.ReadAsArray(i_b, j_b, x_block_size_tmp, y_block_size_tmp)
					for i in range(0,x_block_size_tmp,downscale):
						for j in range(0,y_block_size_tmp,downscale):
							c = label_chunk[j,i]
							c_index = self.classes_labels.index(c) if c in self.classes_labels else -1
							if (c_index != -1):
								if( random.random() <= prob_classes[c_index]):
									ytmp.append(c_index)
									index_tmp.append([j_b+j,i_b+i])

			print(np.bincount(ytmp))

			y = np.array(ytmp)
			index = np.array(index_tmp)

			return index,y
		else:
			print("An error occure during TrainSamplesGenerator initialisation")

	def getAllData(self, windows_w = 10, windows_h = 10, integral = True, downscale = 1):

		if(self.is_init):
			nb_samples = (self.w_d // downscale) * (self.h_d // downscale)
			if( (windows_w == 1) and (windows_h == 1) ):
				X = np.zeros(nb_samples*self.c_d).reshape(nb_samples,self.c_d)
			else:				
				X = np.zeros(nb_samples*self.c_d*windows_w*2*windows_h*2).reshape(nb_samples,self.c_d,windows_w*2,windows_h*2)
			print(nb_samples)
			for i in range(0,self.w_d,downscale):
				for j in range(0,self.h_d,downscale):
					if( (windows_w == 1) and (windows_h == 1) ):
						X[i] = self.data[:,i,j]					
					else:
						imin = max(0,i-windows_w)
						if imin == 0:
							imax = 2*windows_w
						else:
							imax = min(i+windows_w,self.w_d)
							if imax == self.w_d:
								imin = self.w_d-2*windows_w
						jmin = max(0,j-windows_h)
						if jmin == 0:
							jmax = 2*windows_h
						else:
							jmax = min(j+windows_h,self.h_d)
							if jmax == self.h_d:
								jmin = self.h_d-2*windows_h

						tt = self.data[:,imin:imax,jmin:jmax]
						if(integral):
							X[i] = tt.cumsum(2).cumsum(1)
						else:
							X[i] = tt

			return X
		else:
			print("An error occure during TrainSamplesGenerator initialisation")

	def predict_image_all(self, predictor , w_x = 10, w_y = 10):
		if(self.is_init):

			if(self.data_cumul == None ):
				self.data_cumul = self.data.cumsum(2).cumsum(1)

			return predictor.predict_image(self.data_cumul, w_x, w_y)
		else:
			print("An error occure during TrainSamplesGenerator initialisation")
