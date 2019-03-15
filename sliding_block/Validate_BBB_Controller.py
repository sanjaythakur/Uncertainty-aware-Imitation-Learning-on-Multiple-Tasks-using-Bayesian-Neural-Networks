import tensorflow as tf, argparse, _pickle as pickle, os, copy
from Sliding_Block import *
from LQR import *

import sys
sys.path.insert(0,'./../')
from Housekeeping import *

from Load_BBB_Controllers import Load_BBB


def validate_BBB_controller(context_code, window_size, partial_observability):
    copycat_graph = tf.Graph()
    with copycat_graph.as_default():
        copycat_controller = Load_BBB(context_code=context_code, window_size=window_size, partial_observability=partial_observability)

    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    file_to_save_logs = LOGS_DIRECTORY + str(context_code) + '_' + str(window_size) + '_' + str(partial_observability) + '_' + 'BBB' + '.pkl'

    logs_for_all_blocks = {}
    for block_mass in ALL_BLOCK_MASSES_TO_VALIDATE:
        logs_for_a_block_and_initial_state = {}
        for initial_state in INITIALIZATION_STATES_TO_VALIDATE:
            all_observations = []
            all_behavior_control_means = []
            all_behavior_control_deviations = []
            all_behavior_control_maximums = []
            all_behavior_control_minimums = []
            all_behavior_costs = []
            all_target_control_means = []
            all_target_control_deviations = []
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
            
            moving_window_x = np.zeros((1, copycat_controller.moving_windows_x_size))
            moving_window_x[0, -observation.shape[1]:] = observation[0]
            
            behavior_mean_control, behavior_var_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
            behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
            maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            behavior_var_control = behavior_var_control * copycat_controller.deviation_y
            
            step_limit = 0
            while step_limit < (window_size-1):
                step_limit += 1
                all_observations.append(observation)

                all_behavior_control_means.append(behavior_mean_control)
                all_behavior_control_deviations.append(np.sqrt(behavior_var_control))
                all_behavior_control_maximums.append(maximum_this_time_step)
                all_behavior_control_minimums.append(minimum_this_time_step)

                all_target_control_means.append(target_mean_control)
                all_target_control_deviations.append(target_var_control)

                observation, cost, finish = env.step(target_mean_control)
                all_behavior_costs.append(cost)

                if not window_size == 1:
                    moving_window_x[0, :-copycat_controller.drift_per_time_step] = moving_window_x[0, copycat_controller.drift_per_time_step:]
                    moving_window_x[0, -copycat_controller.drift_per_time_step:-(copycat_controller.drift_per_time_step-target_mean_control.shape[1])] = target_mean_control[0]
                    moving_window_x[0, -(copycat_controller.drift_per_time_step-target_mean_control.shape[0])] = -cost      
                
                target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

                if not partial_observability:
                    observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
                else:
                    observation = observation.T
                moving_window_x[0, -observation.shape[1]:] = observation[0]
                
                behavior_mean_control, behavior_var_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
                behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
                maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                behavior_var_control = behavior_var_control * copycat_controller.deviation_y


            while (step_limit < MAXIMUM_NUMBER_OF_STEPS):      
                step_limit += 1
                all_observations.append(observation)

                all_behavior_control_means.append(behavior_mean_control)
                all_behavior_control_deviations.append(np.sqrt(behavior_var_control))
                all_behavior_control_maximums.append(maximum_this_time_step)
                all_behavior_control_minimums.append(minimum_this_time_step)

                all_target_control_means.append(target_mean_control)
                all_target_control_deviations.append(target_var_control)

                observation, cost, finish = env.step(behavior_mean_control)
                all_behavior_costs.append(cost)

                target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

                if not window_size == 1:
                    moving_window_x[0, :-copycat_controller.drift_per_time_step] = moving_window_x[0, copycat_controller.drift_per_time_step:]
                    moving_window_x[0, -copycat_controller.drift_per_time_step:-(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                    moving_window_x[0, -(copycat_controller.drift_per_time_step-behavior_mean_control.shape[0])] = -cost      
                if not partial_observability:
                    observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
                else:
                    observation = observation.T
                moving_window_x[0, -observation.shape[1]:] = observation[0]
                
                behavior_mean_control, behavior_var_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
                behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
                maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                behavior_var_control = behavior_var_control * copycat_controller.deviation_y

            logs_for_a_block_and_initial_state[str(initial_state)] = {OBSERVATIONS_LOG_KEY: np.concatenate(all_observations), BEHAVIORAL_CONTROL_MEANS_LOG_KEY: np.concatenate(all_behavior_control_means),
                                                                     BEHAVIORAL_CONTROL_COSTS_LOG_KEY: np.concatenate(all_behavior_costs), BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY: np.concatenate(all_behavior_control_deviations),
                                                                     BEHAVIORAL_CONTROL_MAXIMUMS_LOG_KEY: np.concatenate(all_behavior_control_maximums), BEHAVIORAL_CONTROL_MINIMUMS_LOG_KEY: np.concatenate(all_behavior_control_minimums), 
                                                                     TARGET_CONTROL_MEANS_LOG_KEY: np.concatenate(all_target_control_means), TARGET_CONTROL_DEVIATIONS_LOG_KEY: np.concatenate(all_target_control_deviations)}
        logs_for_all_blocks[str(block_mass)] = logs_for_a_block_and_initial_state
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_blocks, f, protocol=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context_code', type=int, help='Context code to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    args = parser.parse_args()

    print(GREEN('Settings are context code ' + str(args.context_code) + ', window size is ' + str(args.window_size) + ', partial observability is ' + str(args.partial_observability)))

    validate_BBB_controller(context_code=args.context_code, window_size=args.window_size, partial_observability=str_to_bool(args.partial_observability))
