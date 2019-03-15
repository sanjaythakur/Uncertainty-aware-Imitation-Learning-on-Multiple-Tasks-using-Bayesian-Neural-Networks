import tensorflow as tf
import _pickle as pickle, sys

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
from Housekeeping import *

class Load_BBB():
	def __init__(self, context_code, window_size, partial_observability):
		self.sess = tf.Session()
		configuration_identity = str(context_code) + '_' + str(window_size) + '_' + str(partial_observability) + '_' + 'BBB'
		meta_information_directory_copycat = SAVED_FINAL_MODEL_DIRECTORY + configuration_identity + '/'
		best_model_directory_copycat = SAVED_MODELS_DURING_ITERATIONS_DIRECTORY + configuration_identity + '/'
		imported_meta = tf.train.import_meta_graph(meta_information_directory_copycat + 'final.meta')
		imported_meta.restore(self.sess, tf.train.latest_checkpoint(best_model_directory_copycat))
		graph = tf.get_default_graph()
		self.x_input = graph.get_tensor_by_name('inputs/x_input:0')
		self.y_input = graph.get_tensor_by_name('inputs/y_input:0')
		self.mean_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_mean:0')
		self.deviation_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_standard_deviation:0')
		self.maximum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_maximum:0')
		self.minimum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_minimum:0')
		self.getInputManipulationInformation(configuration_identity)

	def getInputManipulationInformation(self, configuration_identity):
		relevant_file_name = INPUT_MANIPULATION_DIRECTORY + configuration_identity + '.pkl'
		with open(relevant_file_name, 'rb') as f:
			stored_dataset_manipulation_data = pickle.load(f)
		self.mean_x = stored_dataset_manipulation_data[MEAN_KEY_X]
		self.deviation_x = stored_dataset_manipulation_data[DEVIATION_KEY_X]
		self.mean_y = stored_dataset_manipulation_data[MEAN_KEY_Y]
		self.deviation_y = stored_dataset_manipulation_data[DEVIATION_KEY_Y]
		self.drift_per_time_step = stored_dataset_manipulation_data[DRIFT_PER_TIME_STEP_KEY]
		self.moving_windows_x_size = stored_dataset_manipulation_data[MOVING_WINDOWS_X_SIZE_KEY]