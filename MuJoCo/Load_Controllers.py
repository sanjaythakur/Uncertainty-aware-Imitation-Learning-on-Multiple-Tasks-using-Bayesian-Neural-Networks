import tensorflow as tf
import _pickle as pickle, sys

import sys
sys.path.insert(0,'./../')
from Housekeeping import *


class Load_BBB():
	def __init__(self, controller_identity):
		self.sess = tf.Session()
		meta_information_directory_copycat = controller_identity + 'training/' + SAVED_FINAL_MODEL_DIRECTORY
		best_model_directory_copycat = controller_identity + 'training/' + SAVED_MODELS_DURING_ITERATIONS_DIRECTORY
		imported_meta = tf.train.import_meta_graph(meta_information_directory_copycat + 'final.meta')
		imported_meta.restore(self.sess, tf.train.latest_checkpoint(best_model_directory_copycat))
		graph = tf.get_default_graph()
		self.x_input = graph.get_tensor_by_name('inputs/x_input:0')
		self.y_input = graph.get_tensor_by_name('inputs/y_input:0')
		self.mean_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_mean:0')
		self.deviation_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_standard_deviation:0')
		self.maximum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_maximum:0')
		self.minimum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_minimum:0')
		self.getMetaData(controller_identity)

	def getMetaData(self, controller_identity):
		relevant_file_name = controller_identity + 'training/' + 'training_meta_data.pkl'
		with open(relevant_file_name, 'rb') as f:
			stored_meta_data = pickle.load(f)
		self.mean_x = stored_meta_data[MEAN_KEY_X]
		self.deviation_x = stored_meta_data[DEVIATION_KEY_X]
		self.mean_y = stored_meta_data[MEAN_KEY_Y]
		self.deviation_y = stored_meta_data[DEVIATION_KEY_Y]
		self.drift_per_time_step = stored_meta_data[DRIFT_PER_TIME_STEP_KEY]
		self.moving_windows_x_size = stored_meta_data[MOVING_WINDOWS_X_SIZE_KEY]
		self.window_size = stored_meta_data[WINDOW_SIZE_KEY]


class Load_Demonstrator():
	def __init__(self, domain_name, task_identity):
		self.getScaleAndOffset(domain_name, task_identity)
		saved_final_model_demonstrator = SAVED_DEMONSTRATOR_MODELS_DIRECTORY + domain_name + '/' + task_identity + '/'
		imported_meta = tf.train.import_meta_graph(saved_final_model_demonstrator + 'final.meta')
		self.sess = tf.Session()
		imported_meta.restore(self.sess, tf.train.latest_checkpoint(saved_final_model_demonstrator))
		graph = tf.get_default_graph()
		self.scaled_observation_node = graph.get_tensor_by_name('obs:0')
		self.output_action_node = graph.get_tensor_by_name('output_action:0')

	def getScaleAndOffset(self, domain_name, task_identity):
		file_name = DEMONSTRATOR_TRAJECTORIES_DIRECTORY + domain_name + '_' + task_identity + '.pkl'
		with open(file_name, "rb") as f:
			data_stored = pickle.load(f)
		self.scale = data_stored[SCALE_KEY]
		self.offset = data_stored[OFFSET_KEY]