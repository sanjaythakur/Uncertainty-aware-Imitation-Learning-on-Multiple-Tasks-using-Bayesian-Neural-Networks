import matplotlib.pyplot as plt
import copy, os
import numpy as np
from Sliding_Block import *
from LQR import *

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')

from Housekeeping import *


def getDemonstrationsFromBlocknMass(block_mass, window_size, initial_state, partial_observability):
    env = Sliding_Block(mass=block_mass, initial_state=initial_state)
    observation = env.state
    K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)
    u = -1. * np.dot(K, observation)
    moving_windows_x_block_n_state, moving_windows_y_block_n_state = None, None
    if not partial_observability:
        observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
    else:
        observation = observation.T
    drift_per_time_step, moving_windows_x_size, moving_windows_y_size = get_moving_window_size(observation_sample=observation, action_sample=u, window_size=window_size)
    moving_window_x = np.zeros((1, moving_windows_x_size))
    moving_window_x[0, -observation.shape[1]:] = observation[0]
    step_limit = 0
    while (step_limit < MAXIMUM_NUMBER_OF_STEPS):
        if moving_windows_x_block_n_state is None:
            moving_windows_x_block_n_state = copy.deepcopy(moving_window_x)
            moving_windows_y_block_n_state = copy.deepcopy(u)
        else:
            moving_windows_x_block_n_state =  np.append(moving_windows_x_block_n_state, moving_window_x, axis=0)
            moving_windows_y_block_n_state =  np.append(moving_windows_y_block_n_state, u, axis=0)      
        step_limit += 1
        observation, cost, finish = env.step(u)
        if not window_size == 1:
            moving_window_x[0, :-drift_per_time_step] = moving_window_x[0, drift_per_time_step:]
            moving_window_x[0, -drift_per_time_step:-(drift_per_time_step-u.shape[1])] = u[0]
            moving_window_x[0, -(drift_per_time_step-u.shape[0])] = -cost
        u = -1. * np.dot(K, observation)        
        if not partial_observability:
            observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
        else:
            observation = observation.T
        moving_window_x[0, -observation.shape[1]:] = observation[0]
    return moving_windows_x_block_n_state, moving_windows_y_block_n_state, drift_per_time_step, moving_windows_x_size


def getDemonstrationsFromBlock(block_mass, window_size, partial_observability):
    ## Initial-States Grid
    all_states, all_velocities = np.meshgrid(np.linspace(-5, 5, 6), np.linspace(-5, 5, 6))
    all_states = np.reshape(all_states, (-1, 1))
    all_velocities = np.reshape(all_velocities, (-1, 1))
    all_initial_states = np.append(all_states, all_velocities, axis=1)
    moving_windows_x_block, moving_windows_y_block = None, None
    for initial_state in all_initial_states:
        moving_windows_x_block_n_state, moving_windows_y_block_n_state, drift_per_time_step, moving_windows_x_size = getDemonstrationsFromBlocknMass(block_mass=block_mass,
                                                                                                         window_size=window_size,
                                                                                                         initial_state=initial_state,
                                                                                                         partial_observability=partial_observability)
        if moving_windows_x_block is None:
            moving_windows_x_block = copy.deepcopy(moving_windows_x_block_n_state)
            moving_windows_y_block = copy.deepcopy(moving_windows_y_block_n_state)
        else:
            moving_windows_x_block = np.append(moving_windows_x_block, moving_windows_x_block_n_state, axis=0)
            moving_windows_y_block = np.append(moving_windows_y_block, moving_windows_y_block_n_state, axis=0)    
    return moving_windows_x_block, moving_windows_y_block, drift_per_time_step, moving_windows_x_size


def getDemonstrationDataset(all_block_masses, window_size, partial_observability):
    moving_windows_x, moving_windows_y = None, None
    for block_mass in all_block_masses:
        moving_windows_x_block, moving_windows_y_block, drift_per_time_step, moving_windows_x_size = getDemonstrationsFromBlock(block_mass=block_mass,
                                                                                    window_size=window_size,
                                                                                    partial_observability=partial_observability)
        if moving_windows_x is None:
            moving_windows_x = copy.deepcopy(moving_windows_x_block)
            moving_windows_y = copy.deepcopy(moving_windows_y_block)
        else:
            moving_windows_x = np.append(moving_windows_x, moving_windows_x_block, axis=0)
            moving_windows_y = np.append(moving_windows_y, moving_windows_y_block, axis=0)

    return moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size