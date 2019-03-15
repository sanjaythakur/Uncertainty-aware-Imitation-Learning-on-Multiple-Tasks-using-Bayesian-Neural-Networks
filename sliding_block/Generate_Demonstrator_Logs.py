import numpy as np
import _pickle as pickle
import os

from Sliding_Block import *
from LQR import dlqr

import sys
sys.path.insert(0,'./../')

from Housekeeping import *

def generate_demonstrator_logs():
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)

    file_to_save_logs = LOGS_DIRECTORY + 'demonstrator_logs.pkl'

    logs_for_all_blocks = {}
    for block_mass in ALL_BLOCK_MASSES_TO_VALIDATE:
        logs_for_a_block_and_initial_state = {}
        for initial_state in INITIALIZATION_STATES_TO_VALIDATE:
            all_observations = []
            all_controls = []
            all_costs = []
            env = Sliding_Block(mass=block_mass, initial_state=initial_state)
            K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)

            observation = env.state        

            control = -1. * np.dot(K, observation)
            
            step_limit = 0
            while (step_limit < MAXIMUM_NUMBER_OF_STEPS):      
                step_limit += 1
                all_observations.append(observation)
                all_controls.append(control)
                observation, cost, finish = env.step(control)
                all_costs.append(cost)
                control = -1. * np.dot(K, observation)

            logs_for_a_block_and_initial_state[str(initial_state)] = {OBSERVATIONS_LOG_KEY: np.concatenate(all_observations), DEMONSTRATOR_CONTROLS_LOG_KEY: np.concatenate(all_controls),
                                                                     DEMONSTRATOR_COSTS_LOG_KEY: np.concatenate(all_costs)}
        logs_for_all_blocks[str(block_mass)] = logs_for_a_block_and_initial_state
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_blocks, f, protocol=-1)


if __name__ == '__main__':
    generate_demonstrator_logs()