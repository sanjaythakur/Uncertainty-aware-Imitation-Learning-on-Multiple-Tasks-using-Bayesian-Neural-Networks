import numpy as np
import _pickle as pickle
import gpflow
import argparse
import os
from datetime import datetime
from scipy.spatial.distance import cdist

from Dataset import getDemonstrationDataset
from Sliding_Block import *
from LQR import *
#from Validate_GP_Controller import validate_GP_controller

import sys
sys.path.insert(0,'./../')

from Housekeeping import *

def GP_sanity_check(block_mass, should_normalize, lengthscales_code):

    #configuration = str(context_code) + '_' + str(window_size) + '_' + str(partial_observability) + '_' + 'GP_sanity_check_' +  str(should_normalize) + '_' + str(lengthscales_code)

    all_block_masses = [block_mass]
    
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationDataset(all_block_masses=all_block_masses,
                                                         window_size=1,
                                                         partial_observability=True)

    if should_normalize:
        mean_x, deviation_x = get_mean_and_deviation(data = moving_windows_x)
        moving_windows_x = NORMALIZE(moving_windows_x, mean_x, deviation_x)

        mean_y, deviation_y = get_mean_and_deviation(data = moving_windows_y)
        moving_windows_y = NORMALIZE(moving_windows_y, mean_y, deviation_y)
    

    '''
    print(GREEN('Heuristic values of the parameters'))
    kernel_variance = np.var(moving_windows_y)
    kernel_lengthscales = np.median(cdist(moving_windows_x, moving_windows_x, 'sqeuclidean').flatten())
    print(BLUE('Kernel Variance is ' + str(kernel_variance)))
    print(BLUE('Kernel lengthscales is ' + str(kernel_lengthscales)))
    print(BLUE('Likelihood variance is 1/%-10/% /of ' + str(kernel_variance)))
    '''

    if lengthscales_code == 0:
        lengthscales_dimensions = 1
        lengthscales_initial_values = 0.3
    else:
        lengthscales_dimensions = moving_windows_x.shape[1]
        lengthscales_initial_values = 0.1*np.std(moving_windows_x, axis=0)

    print(RED('Block mass is ' + str(block_mass) + ', normalization is ' + str(should_normalize) + ', lengthscale dimensions are ' + str(lengthscales_dimensions) + ', and lengthscale initial values are ' + str(lengthscales_initial_values)))

    states_grid = get_states_grid(resolution=21)
    env = Sliding_Block(mass=block_mass)
    K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)


    k = gpflow.kernels.RBF(lengthscales_dimensions, lengthscales=lengthscales_initial_values)
    #k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    #meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    #m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k, meanf)
    m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k)
    m.likelihood.variance = 0.01
    
    #print(m.read_trainables())
    #print(m.as_pandas_table())

    exhaustive_predictive_errors = []
    exhaustive_standard_deviations = []
    for state in states_grid:
        #env.state = np.reshape(state, (-1, 1))
        demonstrator_action = -1. * np.dot(K, np.reshape(state, (-1,1)))
        if should_normalize:
            state = NORMALIZE(state, mean_x, deviation_x)
        GP_action, GP_variance = m.predict_y(state[None])
        if should_normalize:
            GP_action = REVERSE_NORMALIZE(GP_action, mean_y, deviation_y)
            GP_std = np.std(GP_variance) * deviation_y
        else:
            GP_std = np.std(GP_variance)
        exhaustive_predictive_errors.append(np.mean(np.square(GP_action - demonstrator_action)))
        exhaustive_standard_deviations.append(GP_std)
    print(GREEN('Exhaustive mean squared predictive error is ' + str(np.mean(exhaustive_predictive_errors))))
    print(GREEN('Exhaustive total standard deviations is ' + str(np.sum(exhaustive_standard_deviations))))

    mean_control, var_control = m.predict_y(moving_windows_x)
    mean_squared_predictive_error = np.mean(np.square(mean_control - moving_windows_y))
    average_var_control = np.mean(var_control)
    print(GREEN('Mean squared predictive error is ' + str(mean_squared_predictive_error)))
    print(GREEN('Average control variance is ' + str(average_var_control)))

    start_time = datetime.now()
    gpflow.train.ScipyOptimizer().minimize(m)
    print(RED('Time taken to optimize the parameters is ' + str(datetime.now()-start_time)))

    #plot(m)
    #print(m.read_trainables())
    #print(m.as_pandas_table())

    #print(m.kern.lengthscales.read_value())

    exhaustive_predictive_errors = []
    exhaustive_standard_deviations = []
    for state in states_grid:
        #env.state = np.reshape(state, (-1, 1))
        demonstrator_action = -1. * np.dot(K, np.reshape(state, (-1,1)))
        if should_normalize:
            state = NORMALIZE(state, mean_x, deviation_x)
        GP_action, GP_variance = m.predict_y(state[None])
        if should_normalize:
            GP_action = REVERSE_NORMALIZE(GP_action, mean_y, deviation_y)
            GP_std = np.std(GP_variance) * deviation_y
        else:
            GP_std = np.std(GP_variance)
        exhaustive_predictive_errors.append(np.mean(np.square(GP_action - demonstrator_action)))
        exhaustive_standard_deviations.append(GP_std)
    print(GREEN('Exhaustive mean squared predictive error is ' + str(np.mean(exhaustive_predictive_errors))))
    print(GREEN('Exhaustive total standard deviations is ' + str(np.sum(exhaustive_standard_deviations))))

    mean_control, var_control = m.predict_y(moving_windows_x)
    mean_squared_predictive_error = np.mean(np.square(mean_control - moving_windows_y))
    average_var_control = np.mean(var_control)
    print(GREEN('Mean squared predictive error is ' + str(mean_squared_predictive_error)))
    print(GREEN('Average control variance is ' + str(average_var_control)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bm', '--block_mass', type=float, help='Block Mass to train on', default=10.)
    parser.add_argument('-sn', '--should_normalize', type=str, help='Should Normalize?', default='True')
    parser.add_argument('-lc', '--lengthscales_code', type=int, help='Lengthscales code', default=0)
    args = parser.parse_args()

    GP_sanity_check(block_mass=args.block_mass, should_normalize=str_to_bool(args.should_normalize), lengthscales_code=args.lengthscales_code)
