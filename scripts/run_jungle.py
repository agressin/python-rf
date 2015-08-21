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
try:
   from sklearn.metrics import classification_report
except ImportError:
    pass


#TODO ??
#######################################################################
# Use iPython Parallel
# Needs to load cluster before :
# ipcluster start -n 28 --profile=nbserver
#######################################################################
#dview = None
#if(use_ipython_parallel):
#	print("iPython parallel")
#	from IPython.parallel import Client
#	c = Client(profile='nbserver')
#	print(c.ids)
#	dview = c.load_balanced_view()
#else:
#	print("no iPython parallel")
#
#To use it : j.fit_image(array_image,sample_index, y, dview=dview)



########################################################################
# classification_report
########################################################################
def my_classification_report(array_true, array_pred, labels = None, no_data = 0):

		y_true = array_true.ravel()
		y_pred = array_pred.ravel()
		y_pred = y_pred[y_true != no_data]
		y_true = y_true[y_true != no_data]

		return classification_report(y_true, y_pred, labels)

########################################################################
# Parameters
########################################################################

def main(argv):

	parser = argparse.ArgumentParser(description="Run Jungle Random Forest.")

	parser.add_argument('-i', '--images', help='input images list', type=str, nargs='+')
	parser.add_argument('-l', '--labels', help='input labels list', type=str, nargs='+', default="")
	parser.add_argument('-nd', '--no_data', help='no data value for label (default is 0)', type=int, default=0)
	parser.add_argument('-ds', '--downscale', help='downscaling factor for label search (default is 1)', type=int, default=1)
	parser.add_argument('-o', '--output', help='output dir', type=str, default="")
	parser.add_argument('-oj', '--output_jungle', help='output jungle', type=str, default="")

	parser.add_argument('-t', '--train', help='do train', action='store_true', default=False)
	parser.add_argument('-p', '--predict', help='do predict', action='store_true', default=False)
	parser.add_argument('-pp', '--predict_proba', help='do predict proba', action='store_true', default=False)

	parser.add_argument('-ns', '--nb_samples', help='nb samples per class (default is 500)', type=int, default=500)
	parser.add_argument('-ws', '--windows_size', help='windows_size (default is 50)', type=int, default=50)
	parser.add_argument('-ne', '--nb_estimators', help='nb estimators (trees) (default is 50)', type=int, default=50)
	parser.add_argument('-mf', '--max_features', help='max features tested by split (default is 20)', type=int, default=20)
	parser.add_argument('-md', '--max_depth', help='max depth (default is 5)', type=int, default=5)
	parser.add_argument('-mss', '--min_samples_split', help='min samples split (default is 10)', type=int, default=10)
	parser.add_argument('-msl', '--min_samples_leaf', help='min samples leaf (default is 5)', type=int, default=5)
	parser.add_argument('-en', '--entangled', help='entangled section depth', type=int, nargs='*')

	parser.add_argument('-nf', '--nb_forests', help='nb_forests (default is 1)', type=int, default=1)
	parser.add_argument('-nss', '--nb_steps_simple', help='steps_simple (default is 0)', type=int, default=0)
	parser.add_argument('-nsp', '--nb_steps_proba', help='steps_proba (default is 0)', type=int, default=0)
	parser.add_argument('-sp', '--specialisation', help='specialisation [none, global, per_class] (default is none)', choices=['none', 'global', 'per_class'], default='none')
	parser.add_argument('-app', '--add_previous_prob', help='add previous proba (default is False)', action='store_true', default=False)
	parser.add_argument('-ug', '--use_geodesic', help='use geodesic proba (default is False)', action='store_true', default=False)
	parser.add_argument('-fu', '--fusion', help='fusion [last, mean] (default is last)', choices=['last', 'mean'], default='last')

	parser.add_argument('-ram', '--ram', help='available ram in Mo (default is 128)', type=int, default=128)
	parser.add_argument('-pid', '--pid', help='to save pid in a file', action='store_true', default=False)

	args = parser.parse_args()


	if len(sys.argv) <= 1 :
		parser.print_help()
		exit()

	input_images 			= args.images
	input_labels 			= args.labels
	if input_images is None:
		print("Error : no input image, at least one image is needed")
		exit()

	get_labels = True
	if (len(input_images) != len(input_labels)):
		if args.train :
			print("Error for train: len(input_images) != len(input_labels)")
			exit()
		else :
			get_labels = False
		
	n_images = len(input_images)

	output 						= args.output
	no_data 					= args.no_data

	n_samples 				= args.nb_samples
	SW 								= args.windows_size
	downscale					= args.downscale

	n_estimators 			= args.nb_estimators
	max_features 			= args.max_features
	max_depth 				= args.max_depth
	min_samples_split = args.min_samples_split
	min_samples_leaf 	= args.min_samples_leaf
	
	entangled = None
	if  args.entangled:
		entangled = sorted(args.entangled)
		max_depth = max(max_depth, entangled[-1])
		print("entangled",entangled)
	
	n_forests 				= args.nb_forests
	n_steps_simple		= args.nb_steps_simple
	n_steps_proba			= args.nb_steps_proba
	fusion 						= args.fusion
	specialisation 		= args.specialisation,
	add_previous_prob = args.add_previous_prob
	use_geodesic 			= args.use_geodesic

	verbose=2
	n_jobs=-1

	output += "jungle"
	output += "-ns-" + str(n_samples)
	output += "-ws-" + str(SW)
	output += "-ne-" + str(n_estimators)
	output += "-mf-" + str(max_features)
	output += "-md-" + str(max_depth)
	if entangled:
		output += "-en"
		for en in entangled:
			output += "-"+str(en)

	if n_steps_simple and n_steps_proba :
		output += "-sts-" + str(n_steps_simple)
		output += "-stp-" + str(n_steps_proba)
	else:
		output += "-nf-" + str(n_forests)
		n_steps_simple = None
		n_steps_proba  = None

	if add_previous_prob :
		output += "-app"
		if use_geodesic:
			output += "-ug"

	output += "-fu-" + fusion

	if(use_geodesic):
		output += "geoF"

	if args.output_jungle:
		output_jungle = args.output_jungle
		output_classif = output_jungle
	else:
		output_jungle  = output + ".j"
		output_classif = output

	if args.predict_proba:
		output_classif += "_p"

	print(output_jungle)

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

	raster_images = []
	nbChannels = 0
	XSizeMax = 0
	YSizeMax = 0
	
	for image in input_images:
		raster = gdal.Open(image)
		raster_images.append(raster)
		XSizeMax = max(XSizeMax,raster.RasterXSize)
		YSizeMax = max(YSizeMax,raster.RasterYSize)
		nbChannels_tmp = raster.RasterCount
		if nbChannels == 0:
			nbChannels = nbChannels_tmp
		else :
			if nbChannels != nbChannels_tmp:
				print('Inputs images have different numbers of band')
				exit()

	size_in_Mo = XSizeMax*YSizeMax*nbChannels*8/1000000
	
	if get_labels :
		raster_labels = []
		for label in input_labels:
			raster = gdal.Open(label)
			raster_labels.append(raster)


	########################################################################
	# Get samplors
	########################################################################
	if get_labels :
		samplors = []
		for i in range(n_images):
			sa = TrainSamplesGenerator(raster_images[i],raster_labels[i],no_data=no_data)
			samplors.append(sa)

	########################################################################
	# Training
	########################################################################
	j = None
	if args.train :

		########################################################################
		# Create Jungle
		########################################################################

		f = FeatureFunction(nbChannels,SW,SW)
		j = myJungleClassifier(n_estimators = n_estimators,
							max_features = max_features,
							max_depth = max_depth,
							min_samples_split = min_samples_split,
							min_samples_leaf = min_samples_leaf,
							entangled = entangled,
							verbose=verbose,
							n_jobs=n_jobs,
							featureFunction = f,
							n_forests = n_forests,
							n_steps_simple = n_steps_simple,
							n_steps_proba = n_steps_proba,
							specialisation = specialisation,
							add_previous_prob = add_previous_prob,
				      use_geodesic = use_geodesic,
							fusion = fusion
							)

		########################################################################
		# Train Jungle
		########################################################################
		#Si l'image est trop grosse, on utiliser la 2e méthode
		if(n_images == 1) and (size_in_Mo < args.ram) :
			print("Get samples from one image")
			samplor = samplors[0]
			sample_index, y = samplor.getSamplesImages(n_samples,downscale)
			array_image = np.array(raster_images[0].ReadAsArray())
			print("Train Jungle")
			j.fit_image(array_image,sample_index, y)

		else:
			X = None
			#TODO repartir le nb de samples par images avec la repartition des classes ?
			n_samples_per_image = n_samples // n_images
			print("Get samples from ",n_images," images")
			for samplor in samplors:
				X_tmp, y_tmp = samplor.getSamples(n_samples_per_image, SW,SW, downscale = downscale)
				if X is None:
					X = X_tmp
					y = y_tmp
				else:
					X = np.append(X,X_tmp, axis=0)
					y = np.append(y,y_tmp, axis=0)
			print("X.shape", X.shape)
			print("y.shape", y.shape)
			print("Train Jungle")
			j.fit(X,y)
		########################################################################
		# Save Jungle
		########################################################################
		j.save(output_jungle)

	if args.predict or  args.predict_proba:
		if(j is None):
			print("Load Jungle")
			j = myJungleClassifier.load(output_jungle)

		for i in range(n_images) :
			postfix = ""
			if n_images >=1:
				postfix = "_"+os.path.splitext(os.path.basename(input_images[i]))[0]
			########################################################################
			# Predict Jungle
			########################################################################
			print("Predict Jungle image ", i+1, "/", n_images)
			#TODO predict sur grosse image ?
			array_image = np.array(raster_images[i].ReadAsArray())
			if args.predict :
				out = j.predict_image(array_image,SW,SW)

			if args.predict_proba:
				out = j.predict_proba_image(array_image,SW,SW)

			########################################################################
			# classification_report
			########################################################################
			if args.predict and get_labels :
				try:
					array_label = np.array(raster_labels[i].ReadAsArray())
					report = my_classification_report(array_label,out,samplor.classes_labels,no_data)
					print(report)
					with open(output_jungle+postfix+".txt","a+") as fin:
						fin.write(report)
				except ValueError as e:
					print("error with report :", e)

			########################################################################
			# Save ouput
			########################################################################
			print("Save Predict")
			driver = gdal.GetDriverByName('GTiff')

			if args.predict :
				dst_ds = driver.Create( output_classif+postfix+".tif", out.shape[1], out.shape[0], 1, gdal.GDT_Byte )
				dst_ds.SetGeoTransform( raster_images[i].GetGeoTransform() )
				dst_ds.SetProjection( raster_images[i].GetProjection() )
				dst_ds.GetRasterBand(1).WriteArray( out )

			if args.predict_proba:
				dst_ds = driver.Create( output_classif+postfix+".tif", out.shape[2], out.shape[1], out.shape[0], gdal.GDT_Float32 )
				dst_ds.SetGeoTransform( raster_images[i].GetGeoTransform() )
				dst_ds.SetProjection( raster_images[i].GetProjection() )
				for i in range(out.shape[0]):
					dst_ds.GetRasterBand(i+1).WriteArray( out[i] )

	if args.pid:
		os.unlink(pidfile)

########################################################################
# Main
########################################################################
if __name__ == "__main__" :
	main(sys.argv)

