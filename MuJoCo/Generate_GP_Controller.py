import numpy as np
import _pickle as pickle
import gpflow
import argparse
import os
from datetime import datetime
import tensorflow as tf
from scipy.spatial.distance import cdist

from Dataset import getDemonstrationsFromTask
from Validate_GP_Controller import validate_GP_controller
import sys
sys.path.insert(0,'./../')
from Housekeeping import *


def generate_GP_controller(domain_name, task_identity, window_size, number_demonstrations):
    #if not os.path.exists(LOGS_DIRECTORY):
    #    os.makedirs(LOGS_DIRECTORY)

    GP_LOGS_DIRECTORY = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/GP_controller/window_size_' + str(window_size) + '/'
    if not os.path.exists(GP_LOGS_DIRECTORY):
        os.makedirs(GP_LOGS_DIRECTORY)
    file_to_save_gp_fit_logs = GP_LOGS_DIRECTORY  + str(task_identity) + '_fit.pkl'
    gp_fit_logs = {}

    start_time = datetime.now()
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationsFromTask(domain_name=domain_name, task_identity=task_identity, window_size=window_size, number_demonstrations=number_demonstrations)
    print(RED('Time taken to generate dataset is ' + str(datetime.now()-start_time)))
    
    '''
    print(GREEN('Heuristic values of the parameters'))
    kernel_variance = np.var(moving_windows_y)
    kernel_lengthscales = np.median(cdist(moving_windows_x, moving_windows_x, 'sqeuclidean').flatten())
    print(BLUE('Kernel Variance is ' + str(kernel_variance)))
    print(BLUE('Kernel lengthscales is ' + str(kernel_lengthscales)))
    print(BLUE('Likelihood variance is 1/%-10/% /of ' + str(kernel_variance)))
    '''

    mean_x, deviation_x = get_mean_and_deviation(data = moving_windows_x)
    moving_windows_x = NORMALIZE(moving_windows_x, mean_x, deviation_x)

    mean_y, deviation_y = get_mean_and_deviation(data = moving_windows_y)
    moving_windows_y = NORMALIZE(moving_windows_y, mean_y, deviation_y) 

    k = gpflow.kernels.RBF(moving_windows_x.shape[1], lengthscales=0.01*np.std(moving_windows_x, axis=0))

    moving_windows_x = np.float64(moving_windows_x)
    moving_windows_y = np.float64(moving_windows_y)

    m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k)
    m.likelihood.variance = 1e-4
 
    mean_control, var_control = m.predict_y(moving_windows_x)
    mean_squared_predictive_error = np.mean(np.square(mean_control - moving_windows_y))
    average_dev_control = np.mean(np.sqrt(var_control))
    gp_fit_logs[UNOPTIMIZED_GP_FIT_KEY] = {MEAN_GP_FIT_PREDICTIVE_ERROR_KEY: mean_squared_predictive_error, MEAN_GP_FIT_DEVIATION_KEY: average_dev_control}
    gp_fit_logs[UNOPTIMIZED_GP_TRAINABLES_KEY] = m.as_pandas_table()

    start_time = datetime.now()
    gpflow.train.ScipyOptimizer().minimize(m)
    print(RED('Time taken to optimize the parameters is ' + str(datetime.now()-start_time)))

    mean_control, var_control = m.predict_y(moving_windows_x)
    mean_squared_predictive_error = np.mean(np.square(mean_control - moving_windows_y))
    average_dev_control = np.mean(np.sqrt(var_control))
    gp_fit_logs[OPTIMIZED_GP_FIT_KEY] = {MEAN_GP_FIT_PREDICTIVE_ERROR_KEY: mean_squared_predictive_error, MEAN_GP_FIT_DEVIATION_KEY: average_dev_control}
    gp_fit_logs[OPTIMIZED_GP_TRAINABLES_KEY] = m.as_pandas_table()

    with open(file_to_save_gp_fit_logs, 'wb') as f:
        pickle.dump(gp_fit_logs, f, protocol=-1)

    #plot(m)
    #print(m.read_trainables())
    #print(m.as_pandas_table())

    #print(m.kern.lengthscales.read_value())

    start_time = datetime.now()
    print(GREEN('Started Validation'))
    logs_for_all_tasks = validate_GP_controller(domain_name=domain_name, task_identity=task_identity, window_size=window_size, drift_per_time_step=drift_per_time_step, moving_windows_x_size=moving_windows_x_size, behavior_controller=m,
     mean_x=mean_x, deviation_x=deviation_x, mean_y=mean_y, deviation_y=deviation_y)
    print(RED('Time taken for the validation step is ' + str(datetime.now()-start_time)))

    file_to_save_logs = GP_LOGS_DIRECTORY  + str(task_identity) + '_validation.pkl'
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_tasks, f, protocol=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', type=str, help='MuJoCo domain', choices=['HalfCheetah', 'Swimmer'])
    parser.add_argument('-t', '--task', type=int, help='Task Identity', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=2)
    parser.add_argument('-nd', '--number_demonstrations', type=int, help='Number demonstrations per request', default=10)
    args = parser.parse_args()

    print(GREEN('Domain is ' + args.domain + ', task is ' + str(args.task) + ', window size is ' + str(args.window_size) + ', number of demonstrations is ' + str(args.number_demonstrations)))

    generate_GP_controller(domain_name=args.domain, task_identity=args.task, window_size=args.window_size, number_demonstrations=args.number_demonstrations)
