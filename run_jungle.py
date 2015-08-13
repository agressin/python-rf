#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: agressin
"""

import numpy as np
import os.path
import os,gc,sys
import sys, getopt
import argparse

from rf.FeatureFunction import FeatureFunction
from rf.DecisionTree import myDecisionTreeClassifier
from rf.Forest import myRandomForestClassifier
from rf.TrainSamples import TrainSamplesGenerator
from rf.Jungle import myJungleClassifier

from osgeo import gdal



########################################################################
# Parameters
########################################################################

def main(argv):

	parser = argparse.ArgumentParser(description="Run Jungle Random Forest.")

	parser.add_argument('-i', '--image', help='input image', type=str, default="data/all_crop_2.tif")
	parser.add_argument('-l', '--label', help='input label', type=str, default="learning/OCS_TEST.tif")
	parser.add_argument('-o', '--output', help='output dir', type=str, default="out/test_jungle")

	parser.add_argument('-t', '--train', help='do train', action='store_true', default=False)
	parser.add_argument('-p', '--predict', help='do predict', action='store_true', default=False)
	
	parser.add_argument('-ns', '--nb_samples', help='nb_samples', type=int, default=500)
	parser.add_argument('-ws', '--windows_size', help='windows_size', type=int, default=50)
	parser.add_argument('-ne', '--nb_estimators', help='nb_estimators', type=int, default=50)
	parser.add_argument('-mf', '--max_features', help='max_features', type=int, default=20)
	parser.add_argument('-md', '--max_depth', help='max_depth', type=int, default=5)
	parser.add_argument('-mss', '--min_samples_split', help='min_samples_split', type=int, default=10)
	parser.add_argument('-msl', '--min_samples_leaf', help='min_samples_leaf', type=int, default=5)
	parser.add_argument('-nf', '--nb_forests', help='nb_forests', type=int, default=2)
	parser.add_argument('-app', '--add_previous_prob', help='add previous proba', action='store_true', default=False)
	parser.add_argument('-ug', '--use_geodesic', help='use geodesic proba', action='store_true', default=False)
	parser.add_argument('-fu', '--fusion', help='fusion [None (last only), mean]', type=str, default=None)
	
	parser.add_argument('-pid', '--pid', help='to save pid in a file', action='store_true', default=False)
	
	args = parser.parse_args()

	input_image 			= args.image
	input_label 			= args.label
	output 						= args.output

	n_samples 				= args.nb_samples
	SW 								= args.windows_size
	downscale=1

	n_estimators 			= args.nb_estimators
	max_features 			= args.max_features
	max_depth 				= args.max_depth
	min_samples_split = args.min_samples_split
	min_samples_leaf 	= args.min_samples_leaf

	n_forests 				= args.nb_forests
	fusion 						= args.fusion # last_only, mean, weithed_mean ??, ... ?
	specialisation 		= None, # "global" , "per_class", ... ?
	add_previous_prob = args.add_previous_prob
	use_geodesic 			= args.use_geodesic


	verbose=2
	n_jobs=-1

	output+="-"+str(n_samples)+"-"+str(n_estimators)+"-"+str(max_features)+"-"+str(max_depth)+"-"+str(SW)+"-"+str(n_forests)
	if(use_geodesic):
		  output += "geoF"
	print(output)
	output_classif = output + ".tif"
	output_jungle  = output + ".j"

	########################################################################
	# To save pid in a tmp file
	########################################################################
	if args.pid:
		pid = str(os.getpid())
		pidfile = "/home/prof/iPython/tmp.pid"

		if os.path.isfile(pidfile):
				print (pidfile," already exists, exiting")
				sys.exit()
		else:
				open(pidfile, 'w').write(pid)

	########################################################################
	# Read data and labels
	########################################################################
	print("Read Images")
	print(input_image)
	raster_image = gdal.Open(input_image)
	print(input_label)
	raster_label = gdal.Open(input_label)

	nbChannels = raster_image.RasterCount

	array_image = np.array(raster_image.ReadAsArray())
	array_label = np.array(raster_label.ReadAsArray())

	nbChannels, sizeX, sizeY = array_image.shape

	j = None
	if args.train :
		########################################################################
		# Compute some stats and get random learning data
		########################################################################

		samplor = TrainSamplesGenerator(raster_image,raster_label)

		print("Get random samples")
		sample_index, y = samplor.getSamplesImages(n_samples,downscale)


		########################################################################
		# Train Jungle
		########################################################################
		print("Train Jungle")
		f = FeatureFunction(nbChannels,SW,SW)
		j = myJungleClassifier(n_estimators = n_estimators,
							max_features = max_features,
							max_depth = max_depth,
							min_samples_split = min_samples_split,
							min_samples_leaf = min_samples_leaf,
							verbose=verbose,
							n_jobs=n_jobs,
							featureFunction = f,
							n_forests = n_forests,
							specialisation = specialisation,
							add_previous_prob = add_previous_prob,
				      use_geodesic = use_geodesic,
							fusion = fusion # last_only, mean, weithed_mean ??, ... ?
							)
		j.fit_image(array_image,sample_index, y)

		j.save(output_jungle)

	if args.predict :
		if(j is None):
			print("Load Jungle")
			j = myJungleClassifier.load(output_jungle)
		########################################################################
		# Predict Jungle
		########################################################################
		print("Predict Jungle")
		out = j.predict_image(array_image,SW,SW)

		########################################################################
		# Save ouput
		########################################################################
		print("Save Predict")
		driver = gdal.GetDriverByName('GTiff')

		dst_ds = driver.Create( output_classif, out.shape[1], out.shape[0], 1, gdal.GDT_Byte )
		dst_ds.SetGeoTransform( raster_image.GetGeoTransform() )
		dst_ds.SetProjection( raster_image.GetProjection() )
		dst_ds.GetRasterBand(1).WriteArray( out )
	
	if args.pid:
		os.unlink(pidfile)

########################################################################
# Main
########################################################################
if __name__ == "__main__" :
	main(sys.argv)

