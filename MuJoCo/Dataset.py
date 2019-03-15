import copy, os
import numpy as np
import _pickle as pickle
import sys
sys.path.insert(0,'./../')

from Housekeeping import *

def getDemonstrationsFromTask(domain_name, task_identity, window_size, number_demonstrations):
    file_name = DEMONSTRATOR_TRAJECTORIES_DIRECTORY + domain_name + '_' + str(task_identity) + '.pkl'
    with open(file_name, "rb") as f:
        data_stored = pickle.load(f)
    demonstrator_trajectories = data_stored[DEMONSTRATOR_TRAJECTORY_KEY]
    moving_windows_x_task, moving_windows_y_task = None, None
    for episode in demonstrator_trajectories[:number_demonstrations]:
        observations = episode[DEMONSTRATOR_OBSERVATIONS_KEY]
        actions = episode[DEMONSTRATOR_ACTIONS_KEY]
        rewards = episode[DEMONSTRATOR_REWARDS_KEY]
        unscaled_observations = episode[DEMONSTRATOR_UNSCALED_OBSERVATIONS_KEY]

        drift_per_time_step, moving_windows_x_size, moving_windows_y_size = get_moving_window_size(observation_sample=unscaled_observations[:1], action_sample=actions[:1], window_size=window_size)

        moving_window_x = np.zeros((1, moving_windows_x_size))
        current_observation = unscaled_observations[0]
        current_action = np.reshape(actions[0], (1, -1))
        current_reward = rewards[0]

        for time_step_iterator in range(1, unscaled_observations.shape[0]):
            moving_window_x[0, -unscaled_observations.shape[1]:] = current_observation
            if moving_windows_x_task is None:
                moving_windows_x_task = copy.deepcopy(moving_window_x)
                moving_windows_y_task = copy.deepcopy(current_action)
            else:
                moving_windows_x_task =  np.append(moving_windows_x_task, moving_window_x, axis=0)
                moving_windows_y_task =  np.append(moving_windows_y_task, current_action, axis=0)      
            
            if not window_size == 1:
                moving_window_x[0, :-drift_per_time_step] = moving_window_x[0, drift_per_time_step:]
                moving_window_x[0, -drift_per_time_step:-(drift_per_time_step-current_action.shape[1])] = current_action[0]
                moving_window_x[0, -(drift_per_time_step-current_action.shape[1])] = current_reward

            current_observation = unscaled_observations[time_step_iterator]
            current_action = np.reshape(actions[time_step_iterator], (1, -1))
            current_reward = rewards[time_step_iterator]            

        moving_window_x[0, -unscaled_observations.shape[1]:] = current_observation
        moving_windows_x_task = np.append(moving_windows_x_task, moving_window_x, axis=0)
        moving_windows_y_task = np.append(moving_windows_y_task, current_action, axis=0)

    return moving_windows_x_task, moving_windows_y_task, drift_per_time_step, moving_windows_x_size