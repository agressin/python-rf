import numpy as np
import random
from osgeo import gdal

class TrainSamplesGenerator():
	""" TrainSampleGenerator """
	def __init__(self, input_data, input_label, no_data = 0):
		""" init"""
		self.is_init = False
		self.no_data = no_data
		
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
			classes_labels = np.nonzero(hist>10)[0].tolist()
			
			#Remove no data classe
			if no_data in classes_labels:
				classes_labels.remove(no_data)
			
			self.classes_labels = classes_labels
			self.classes_freq = hist[classes_labels]
			
			print("Classes labels :",self.classes_labels)
			print("Classes frequencies :",self.classes_freq)
			self.is_init = True

	def getSamples(self, nb_samples=100, windows_w = 1, windows_h = 1, integral = True,  downscale = 1):
		sample_index,y = self.getSamplesImages(nb_samples,  downscale)
		X = self.getSamplesData(sample_index, windows_w,  windows_h, integral)
		return X,y

	def getSamplesImages (self, nb_samples=100,  downscale = 1):
		if(self.is_init):
			nb_samples = min(min(self.classes_freq),nb_samples)
			print("nb_samples : ", nb_samples)
			prob_classes = nb_samples*downscale*downscale / self.classes_freq

			index_tmp=[]
			ytmp=[]
			nb_samples_take = 0 * self.classes_freq
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
								if (random.random() <= prob_classes[c_index]) and (nb_samples_take[c_index] <= nb_samples):
									ytmp.append(c)
									index_tmp.append([j_b+j,i_b+i])
									nb_samples_take[c_index] += 1

			#print(np.bincount(ytmp))
			print("nb_samples_take",nb_samples_take)

			y = np.array(ytmp)
			index = np.array(index_tmp)

			return index,y
		else:
			print("An error occure during TrainSamplesGenerator initialisation")

	def getSamplesData(self, sample_index, windows_w = 1,  windows_h = 1, integral = True):
		if(self.is_init):
			raster_data = self.raster_data
			block_sizes = raster_data.GetRasterBand(1).GetBlockSize()  
			x_block_size = block_sizes[0]  
			y_block_size = block_sizes[1]  
			
			xsize = raster_data.RasterXSize  
			ysize = raster_data.RasterYSize
		
			nb_samples = len(sample_index)
			is_single_pixel = ( windows_h == 1 ) and ( windows_w == 1)
			if is_single_pixel :
				X = np.zeros((nb_samples,self.c_d))
			else:
				X = np.zeros((nb_samples,self.c_d,windows_w, windows_h))
			print("X.shape ",X.shape)
			w = windows_w //2
			h = windows_h //2
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
					list_id_in_chunk = np.where(
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
							if(is_single_pixel) :
								X[id,:] = data_chunk[:,jmin_chunk,imin_chunk]
							else :
								X[id,:,jmin_out:jmax_out,imin_out:imax_out] = data_chunk[:,jmin_chunk:jmax_chunk,imin_chunk:imax_chunk]
			if(is_single_pixel or (not integral) ) :
				return X
			else :
				return X.cumsum(3).cumsum(2)
