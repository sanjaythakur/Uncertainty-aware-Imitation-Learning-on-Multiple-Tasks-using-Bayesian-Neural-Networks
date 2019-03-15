import numpy as np


#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)


# Hyper-parameters
NUMBER_SAMPLES_FOR_VARIANCE_REDUCTION = 5
NUMBER_SAMPLES_FOR_VALIDATION = 5

POLE_MASS = 'pole_mass'
EXPERT_CONTROLLER_LOCATION = 'expert_controller_location'
COLOR = 'color'

#All environments
PHYSICAL_PENDULUM = 'physical_pendulum'
SIMULATOR_PENDULUM = 'simulator_pendulum'

#All contexts
CONTEXT_0 = 'context_0'
CONTEXT_1 = 'context_1'
CONTEXT_2 = 'context_2'
CONTEXT_3 = 'context_3'
CONTEXT_4 = 'context_4'
CONTEXT_5 = 'context_5'
CONTEXT_6 = 'context_6'
CONTEXT_7 = 'context_7'
CONTEXT_8 = 'context_8'
CONTEXT_9 = 'context_9'
CONTEXT_10 = 'context_10'
CONTEXT_11 = 'context_11'
CONTEXT_12 = 'context_12'
CONTEXT_13 = 'context_13'
CONTEXT_14 = 'context_14'
CONTEXT_15 = 'context_15'
CONTEXT_16 = 'context_16'
CONTEXT_17 = 'context_17'
CONTEXT_18 = 'context_18'
CONTEXT_19 = 'context_19'
CONTEXT_20 = 'context_20'
CONTEXT_21 = 'context_21'
CONTEXT_22 = 'context_22'
CONTEXT_23 = 'context_23'
CONTEXT_24 = 'context_24'
CONTEXT_25 = 'context_25'
CONTEXT_26 = 'context_26'
CONTEXT_27 = 'context_27'
CONTEXT_28 = 'context_28'
CONTEXT_29 = 'context_29'
CONTEXT_30 = 'context_30'


multiple_contexts_1 = ['context_0', 'context_3', 'context_6']


ALL_CONTEXTS = {
				PHYSICAL_PENDULUM:
								{
								 CONTEXT_0: {EXPERT_CONTROLLER_LOCATION: '/media/diskstation/juan/arduino_cartpole_expert/policy_24', COLOR: 'black'},
								 CONTEXT_1: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_1/policy_52', COLOR: 'red'},
								 CONTEXT_2: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_2/policy_20', COLOR: 'gold'},
								 CONTEXT_3: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_3/policy_21', COLOR: 'chartreuse'},
								 CONTEXT_4: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_4/policy_20', COLOR: 'deepskyblue'},
								 CONTEXT_5: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_5/policy_22', COLOR: 'navy'},
								 CONTEXT_6: {EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/physical_pendulum/mcpilco_cdropoutd_dropoutp_8_context_6/policy_25', COLOR: 'm'},
								},
				SIMULATOR_PENDULUM:
								{
								 CONTEXT_0: {POLE_MASS: 0.2, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.2/policy_50'},
								 CONTEXT_1: {POLE_MASS: 0.3, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.3/policy_50'},
								 CONTEXT_2: {POLE_MASS: 0.5, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.5/policy_50'},
								 CONTEXT_3: {POLE_MASS: 0.62, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.62/policy_50'},
								 CONTEXT_4: {POLE_MASS: 0.75, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.75/policy_50'},
								 CONTEXT_5: {POLE_MASS: 0.87, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_0.87/policy_50'},
								 CONTEXT_6: {POLE_MASS: 1.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.0/policy_50'},
								 CONTEXT_7: {POLE_MASS: 1.12, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.12/policy_50'},
								 CONTEXT_8: {POLE_MASS: 1.25, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.25/policy_50'},
								 CONTEXT_9: {POLE_MASS: 1.37, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.37/policy_50'},
								 CONTEXT_10: {POLE_MASS: 1.5, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.5/policy_50'},
								 CONTEXT_11: {POLE_MASS: 1.62, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.62/policy_50'},
								 CONTEXT_12: {POLE_MASS: 1.75, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.75/policy_50'},
								 CONTEXT_13: {POLE_MASS: 1.87, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_1.87/policy_50'},
								 CONTEXT_14: {POLE_MASS: 2.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.0/policy_50'},
								 CONTEXT_15: {POLE_MASS: 2.12, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.12/policy_50'},
								 CONTEXT_16: {POLE_MASS: 2.25, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.25/policy_50'},
								 CONTEXT_17: {POLE_MASS: 2.37, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.37/policy_50'},
								 CONTEXT_18: {POLE_MASS: 2.5, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.5/policy_50'},
								 CONTEXT_19: {POLE_MASS: 2.62, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.62/policy_50'},
								 CONTEXT_20: {POLE_MASS: 2.75, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.75/policy_50'},
								 CONTEXT_21: {POLE_MASS: 2.87, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_2.87/policy_50'},
								 CONTEXT_22: {POLE_MASS: 3.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_3.0/policy_50'},
								 CONTEXT_23: {POLE_MASS: 5.5, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_5.5/policy_100'},
								 CONTEXT_24: {POLE_MASS: 8.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_8.0/policy_100'},
								 CONTEXT_25: {POLE_MASS: 10.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_10.0/policy_100'},
								 CONTEXT_26: {POLE_MASS: 13.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_13.0/policy_100'},
								 CONTEXT_27: {POLE_MASS: 17.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_17.0/policy_100'},
								 CONTEXT_28: {POLE_MASS: 20.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_20.0/policy_100'},
								 CONTEXT_29: {POLE_MASS: 30.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_30.0/policy_100'},
								 CONTEXT_30: {POLE_MASS: 40.0, EXPERT_CONTROLLER_LOCATION: '/home/adaptation/sanjay/.kusanagi/cp_results/simulator_pendulum/mcpilco_cdropoutd_dropoutp_9_pole_mass_40.0/policy_100'},
								}
		 		}


DEMONSTRATOR_STATES_KEY = 'demonstrator_states'
DEMONSTRATOR_ACTIONS_KEY = 'demonstrator_actions'
DEMONSTRATOR_COSTS_KEY = 'demonstrator_costs'

LEARNER_INDUCED_STATES_KEY = 'learner_induced_states_key'
LEARNER_INDUCED_ACTIONS_KEY = 'learner_induced_actions_key'
LEARNER_INDUCED_COST_Ts_KEY = 'learner_induced_cost_ts_key'
LEARNER_INDUCED_SIGMA_Ts_KEY = 'learner_induced_sigma_ts_key'

MEAN_KEY_X = 'mean_key_x'
DEVIATION_KEY_X = 'deviation_key_x'
MEAN_KEY_Y = 'mean_key_y'
DEVIATION_KEY_Y = 'deviation_key_y'

OBSERVATION_DIMENSIONS_PER_TIME_STEP_KEY = 'observation_dimensions_per_time_step_key'
OBSERVATION_WINDOW_SIZE_KEY = 'observation_window_size_key'
MOVING_WINDOW_KEY = 'moving_window_key'

DEMONSTRATIONS_DIRECTORY = './demonstrations_directory/'
TENSORBOARD_DIRECTORY = './tensorboard_data_copycat/'

SAVED_MODELS_DURING_ITERATIONS_DIRECTORY_COPYCAT = './saved_models_during_iterations_copycat/'
SAVED_FINAL_MODEL_DIRECTORY_COPYCAT = './saved_final_model_copycat/'
SAVED_PARAMS_FOR_TRAINING_COPYCAT = './saved_params_for_training_copycat/'

INPUT_REPRESENTATION_AND_MANIPULATION_DATA_DIRECTORY_COPYCAT = './data_for_input_representation_and_manipulation/'
EXPERT_CONTROLLER_PERFORMANCE_LOG_DIRECTORY = './logs/expert_controller_performance/'
COPYCAT_CONTROLLER_PERFORMANCE_LOG_DIRECTORY = './logs/copycat_controller_performance/'


def standard_normalize(data):
	mean = np.mean(data, axis = 0)
	deviation = np.std(data, axis = 0)
	for some_iterator in range(deviation.shape[0]):
		if deviation[some_iterator] == 0:
			deviation[some_iterator] = 0.00001
	data = np.divide(np.subtract(data, mean), deviation)
	return data, mean, deviation


NORMALIZE = lambda data, mean, deviation: np.divide(np.subtract(data, mean), deviation)
REVERSE_NORMALIZE = lambda data, mean, deviation: np.add((data * deviation), mean)


def randomize(a, b):
	# Generate the permutation index array.
	permutation = np.random.permutation(a.shape[0])
	# Shuffle the arrays by giving the permutation in the square brackets.
	shuffled_a = a[permutation]
	shuffled_b = b[permutation]
	return shuffled_a, shuffled_b


'''
def getDemonstratorPolicyLocation(env_name, context):
	if env_name == PHYSICAL_PENDULUM:
		if context == CONTEXT_DEFAULT:
			return '/media/diskstation/juan/arduino_cartpole_expert/policy_24'
	elif env_name == SIMULATOR_PENDULUM:
		if context == CONTEXT_DEFAULT:
			return '/home/adaptation/sanjay/.kusanagi/output/Cartpole_8/policy_41'
	else:
		return ''
'''