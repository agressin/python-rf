# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:07:30 2015

@author: agressin
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt

from python-rf.FeatureFunction import FeatureFunction
from python-rf.DecisionTree import myDecisionTreeClassifier
from python-rf.Forest import myRandomForestClassifier
from python-rf.TrainSamples import TrainSamplesGenerator
from python-rf.ClassificationEvaluator import ClassificationEvaluator

from osgeo import gdal

########################################################################
# Parameters
########################################################################

input_image="data/all_crop_2.tif"
input_label="learning/OCS_TEST.tif"

nb_samples = 1000

downscale=2
train_or_load = True
vector_or_image = False
classify = True
get_proba = False
get_FI = False
train_gpu = False

for SW in range(15, 0, -5):
	print(SW)
	SWx=SW
	SWy=SW
	for max_depth in range(5, 15, 5):
		print(max_depth)
		n_estimators = 20
		max_features = 10
		min_samples_split = 10
		min_samples_leaf = 10
		verbose=2
		n_jobs=-1

		output="out/rf_"+str(nb_samples)+"-"+str(n_estimators)+"-"+str(max_features)+"-"+str(max_depth)+"-"+str(SW)
		output_rf = output + ".p"
		output_classif= output
		if get_proba:
			output_classif += "_p"
		output_classif += ".tif"

		########################################################################
		# Read data and labels
		########################################################################
		raster = gdal.Open(input_image)
		imarray = np.array(raster.ReadAsArray())

		raster_label = gdal.Open(input_label)
		imarray_label = np.array(raster_label.ReadAsArray())

		nbChannels, sizeX, sizeY = imarray.shape

		########################################################################
		# Compute some stats and get random learning data
		########################################################################

		samplor = TrainSamplesGenerator(imarray,imarray_label)
		if train_or_load:
			print("Get samples")
			if vector_or_image:
				X,y = samplor.getSamples(nb_samples,SWx,SWy,downscale)
			else:
				sample_index, y = samplor.getSamplesImages(nb_samples,downscale)


		########################################################################
		# Train RF ...
		########################################################################
		if train_or_load:
			print("Train RF")

			#To get FI from a previous Forest
			#d1 = myRandomForestClassifier.load("out/rf_1500-10-10-5_1.p")
			#acc = d1.getFeatureImportance()
			f = FeatureFunction(nbChannels,SWx,SWy)
			#f.random_weight = acc

			d = myRandomForestClassifier(n_estimators = n_estimators,
							max_features = max_features,
							max_depth = max_depth,
							min_samples_split = min_samples_split,
							min_samples_leaf = min_samples_leaf,
							verbose=verbose,
							n_jobs=n_jobs,
							featureFunction = f)

			start = timeit.default_timer()

			if vector_or_image:
				d.fit(X,y)
			else:
				d.fit_image(imarray.cumsum(2).cumsum(1),sample_index,y, gpu=train_gpu)

			stop = timeit.default_timer()
			print(stop-start)

			print("save")
			d.save(output_rf)

		########################################################################
		# ... or load RF
		########################################################################
		if not train_or_load:
			print("load")
			d = myRandomForestClassifier.load(output_rf)

		########################################################################
		# GetFI
		########################################################################

		if get_FI:
			acc = d.getFeatureImportance()

			print('type',acc['RQE']['type'])
			print('channel',acc['RQE']['channel'])
			print('height',acc['RQE']['windows_height'])
			print('width',acc['RQE']['windows_width'])
			print('dist',acc['RQE']['windows_dist'])


		########################################################################
		# Classify
		########################################################################
		if classify:
			print("Classify")
			start = timeit.default_timer()

			print(sizeX, sizeY)

			if get_proba:
				out = d.predict_proba_image(imarray,SWx,SWy)
			else:
				out = d.predict_image(imarray,SWx,SWy)

			stop = timeit.default_timer()
			print(stop-start)

		########################################################################
		# ClassificationEvaluator
		########################################################################
			#cf = ClassificationEvaluator(out, imarray_label)
			#cf.computeStat()

		########################################################################
		# Save image
		########################################################################

			driver = gdal.GetDriverByName('GTiff')

			if get_proba:
				dst_ds = driver.Create( output_classif, out.shape[2], out.shape[1], out.shape[0], gdal.GDT_Float32 )
			else:
				dst_ds = driver.Create( output_classif, out.shape[1], out.shape[0], 1, gdal.GDT_Byte )

			dst_ds.SetGeoTransform( raster.GetGeoTransform() )
			dst_ds.SetProjection( raster.GetProjection() )

			if get_proba:
				for i in range(out.shape[0]):
					dst_ds.GetRasterBand(i+1).WriteArray( out[i] )
			else:
				dst_ds.GetRasterBand(1).WriteArray( out )
