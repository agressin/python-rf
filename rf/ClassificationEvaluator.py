import numpy as np
import sklearn.metrics as metrics
#TODO :
# use from sklearn.metrics import confusion_matrix
# confusion_matrix(array_label.ravel, out.ravel, samplor.classes_labels)
#
class ClassificationEvaluator():
	""" ClassificationEvaluator """

	def classification_report(array_true, array_pred, labels = None, no_data = 0):

		y_true = array_true.ravel()
		y_pred = array_pred.ravel()
		y_pred = y_pred[y_true != no_data]
		y_true = y_true[y_true != no_data]

		return metrics.classification_report(y_true, y_pred, labels)				
