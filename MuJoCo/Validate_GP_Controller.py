import numpy as np
import _pickle as pickle
import gpflow
import os
import tensorflow as tf

from Load_Controllers import Load_Demonstrator
from multiple_tasks import get_task_on_MUJOCO_environment
import sys
sys.path.insert(0,'./../')
from Housekeeping import *


def validate_GP_controller(domain_name, task_identity, window_size, drift_per_time_step, moving_windows_x_size, behavior_controller, mean_x, deviation_x, mean_y, deviation_y):
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)

    #file_to_save_logs = LOGS_DIRECTORY + domain_name + '_' + str(task_identity) + '_' + str(window_size) + '_GP.pkl'

    logs_for_all_tasks = {}
    for task_to_validate in ALL_MUJOCO_TASK_IDENTITIES:
        logs_for_a_task = {}

        demonstrator_graph = tf.Graph()
        with demonstrator_graph.as_default():
            demonstrator_controller = Load_Demonstrator(domain_name=domain_name, task_identity=str(task_to_validate))

        for validation_trial in range(NUMBER_VALIDATION_TRIALS):
            all_observations = []
            all_behavior_control_means = []
            all_behavior_control_deviations = []
            all_behavior_rewards = []
            all_demonstrator_controls = []
            #all_target_control_means, all_target_control_deviations = [], []
            env = get_task_on_MUJOCO_environment(env_name=domain_name, task_identity=str(task_to_validate))
            total_cost = total_variance = 0.
            observation = env.reset()
            finish = False
            
            moving_window_x = np.zeros((1, moving_windows_x_size))
            moving_window_x[0, -observation.shape[0]:] = observation
            
            behavior_mean_control, behavior_var_control = behavior_controller.predict_y(NORMALIZE(moving_window_x, mean_x, deviation_x))
            behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, mean_y, deviation_y)
            behavior_var_control = behavior_var_control * deviation_y
            
            time_step = 0.0            

            while not finish:
                all_observations.append(observation)

                all_behavior_control_means.append(behavior_mean_control)
                all_behavior_control_deviations.append(np.sqrt(behavior_var_control))

                observation = np.append(observation, time_step) # add time step feature
                demonstrator_control = demonstrator_controller.sess.run(demonstrator_controller.output_action_node, feed_dict={demonstrator_controller.scaled_observation_node: (observation.reshape(1,-1) - demonstrator_controller.offset) * demonstrator_controller.scale})
                all_demonstrator_controls.append(demonstrator_control)

                time_step += 1e-3

                #all_target_control_means.append(target_mean_control)
                #all_target_control_deviations.append(target_var_control)

                observation, reward, finish, info = env.step(behavior_mean_control)
                all_behavior_rewards.append(reward)

                #target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

                if not window_size == 1:
                    moving_window_x[0, :-drift_per_time_step] = moving_window_x[0, drift_per_time_step:]
                    moving_window_x[0, -drift_per_time_step:-(drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                    moving_window_x[0, -(drift_per_time_step-behavior_mean_control.shape[1])] = reward      

                moving_window_x[0, -observation.shape[0]:] = observation
                
                behavior_mean_control, behavior_var_control = behavior_controller.predict_y(NORMALIZE(moving_window_x, mean_x, deviation_x))
                behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, mean_y, deviation_y)
                behavior_var_control = behavior_var_control * deviation_y

            all_observations = np.array(all_observations)
            all_behavior_control_means = np.concatenate(all_behavior_control_means, axis=0)
            all_behavior_rewards =  np.array(all_behavior_rewards)
            all_behavior_control_deviations = np.concatenate(all_behavior_control_deviations, axis=0)
            all_demonstrator_controls = np.array(all_demonstrator_controls)

            logs_for_a_task[str(validation_trial)] = {OBSERVATIONS_LOG_KEY: all_observations, BEHAVIORAL_CONTROL_MEANS_LOG_KEY: all_behavior_control_means,
                                                     BEHAVIORAL_CONTROL_REWARDS_LOG_KEY: all_behavior_rewards, BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY: all_behavior_control_deviations,
                                                     TARGET_CONTROL_MEANS_LOG_KEY: all_demonstrator_controls}
        logs_for_all_tasks[str(task_to_validate)] = logs_for_a_task
    #with open(file_to_save_logs, 'wb') as f:
    #    pickle.dump(logs_for_all_tasks, f, protocol=-1)
    return logs_for_all_tasks