import numpy as np
import _pickle as pickle
import gpflow
import os

from Dataset import getDemonstrationDataset
from Sliding_Block import *

import sys
sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')

from Housekeeping import *

def validate_GP_controller(context_code, window_size, partial_observability, drift_per_time_step, moving_windows_x_size, behavior_controller, mean_x, deviation_x, mean_y, deviation_y):
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)

    file_to_save_logs = LOGS_DIRECTORY + str(context_code) + '_' + str(window_size) + '_' + str(partial_observability) + '_' + 'GP.pkl'

    logs_for_all_blocks = {}
    for block_mass in ALL_BLOCK_MASSES_TO_VALIDATE:
        logs_for_a_block_and_initial_state = {}
        for initial_state in INITIALIZATION_STATES_TO_VALIDATE:
            all_observations = []
            all_behavior_control_means = []
            all_behavior_control_deviations = []
            all_behavior_costs = []
            all_target_control_means, all_target_control_deviations = [], []
            env = Sliding_Block(mass=block_mass, initial_state=initial_state)
            total_cost = total_variance = 0.
            observation = env.state

            from LQR import dlqr
            K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)            

            target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])
            
            if not partial_observability:
                observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
            else:
                observation = observation.T
            
            moving_window_x = np.zeros((1, moving_windows_x_size))
            moving_window_x[0, -observation.shape[1]:] = observation[0]
            
            behavior_mean_control, behavior_var_control = behavior_controller.predict_y(NORMALIZE(moving_window_x, mean_x, deviation_x))
            behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, mean_y, deviation_y)
            behavior_var_control = behavior_var_control * deviation_y
            
            step_limit = 0
            while (step_limit < MAXIMUM_NUMBER_OF_STEPS):      
                step_limit += 1
                all_observations.append(observation)

                all_behavior_control_means.append(behavior_mean_control)
                all_behavior_control_deviations.append(np.sqrt(behavior_var_control))

                all_target_control_means.append(target_mean_control)
                all_target_control_deviations.append(target_var_control)

                observation, cost, finish = env.step(behavior_mean_control)
                all_behavior_costs.append(cost)

                target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

                if not window_size == 1:
                    moving_window_x[0, :-drift_per_time_step] = moving_window_x[0, drift_per_time_step:]
                    moving_window_x[0, -drift_per_time_step:-(drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                    moving_window_x[0, -(drift_per_time_step-behavior_mean_control.shape[0])] = -cost      
                if not partial_observability:
                    observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
                else:
                    observation = observation.T
                moving_window_x[0, -observation.shape[1]:] = observation[0]
                
                behavior_mean_control, behavior_var_control = behavior_controller.predict_y(NORMALIZE(moving_window_x, mean_x, deviation_x))
                behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, mean_y, deviation_y)
                behavior_var_control = behavior_var_control * deviation_y

            logs_for_a_block_and_initial_state[str(initial_state)] = {OBSERVATIONS_LOG_KEY: np.concatenate(all_observations), BEHAVIORAL_CONTROL_MEANS_LOG_KEY: np.concatenate(all_behavior_control_means),
                                                                     BEHAVIORAL_CONTROL_COSTS_LOG_KEY: np.concatenate(all_behavior_costs), BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY: np.concatenate(all_behavior_control_deviations), 
                                                                     TARGET_CONTROL_MEANS_LOG_KEY: np.concatenate(all_target_control_means), TARGET_CONTROL_DEVIATIONS_LOG_KEY: np.concatenate(all_target_control_deviations)}
        logs_for_all_blocks[str(block_mass)] = logs_for_a_block_and_initial_state
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_blocks, f, protocol=-1)