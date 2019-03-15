import numpy as np
import copy

import sys
sys.path.insert(0, './../')
from Housekeeping import *


class Detector():
	def __init__(self, domain_name, start_monitoring_at_time_step=200, initial_threshold=0.3, detector_m=50, detector_c=1.5):
		self.domain_name = domain_name
		self.start_monitoring_at_time_step = start_monitoring_at_time_step
		self.threshold = initial_threshold
		self.detector_m = detector_m
		self.detector_c = detector_c
		self.reset()

	def isSafeToContinue(self, current_uncertainty):
		self.current_time_step += 1
		self.running_detector_window[:-1, :] = copy.deepcopy(self.running_detector_window[1:, :])
		self.running_detector_window[-1] = copy.deepcopy(current_uncertainty[0])
		if self.current_time_step < self.detector_m:
			self.current_uncertainty = np.sum(self.running_detector_window)/(self.current_time_step * current_uncertainty.shape[1])
		else:
			self.current_uncertainty = np.mean(self.running_detector_window)

		if self.current_time_step < self.start_monitoring_at_time_step:
			isSafe = 'True'
		else:
			if self.current_uncertainty < (self.detector_c * self.threshold):
				isSafe = 'True'
			else:
				isSafe = 'False'
		return isSafe, self.getStats()

	def reset(self):
		self.current_time_step = 0
		self.current_uncertainty = 0.
		if self.domain_name == 'HalfCheetah':
			self.running_detector_window = np.zeros((self.detector_m, 6))
		elif self.domain_name == 'Swimmer':
			self.running_detector_window = np.zeros((self.detector_m, 2))

	def getStats(self):
		return {CURRENT_DETECTOR_UNCERTAINTY_KEY: self.current_uncertainty, CURRENT_DETECTOR_UNCERTAINTY_THRESHOLD_KEY: self.threshold, RUNNING_DETECTOR_WINDOW_KEY: self.running_detector_window, CURRENT_TIME_STEP_KEY: self.current_time_step}	


def get_smoothed_transform(input_array, domain_name, smoothening_m, scaling_c):
	detector = Detector(domain_name=domain_name, detector_m=smoothening_m, detector_c=scaling_c)
	smoothed_transform = []
	for input_item in input_array:
		isSafe, stats = detector.isSafeToContinue(input_item[None])
		smoothed_transform.append(stats[CURRENT_DETECTOR_UNCERTAINTY_KEY])
	return smoothed_transform			 