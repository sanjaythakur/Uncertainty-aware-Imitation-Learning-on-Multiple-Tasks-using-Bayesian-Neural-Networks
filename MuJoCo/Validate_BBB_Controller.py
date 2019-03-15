import tensorflow as tf, argparse, gym, _pickle as pickle, numpy as np, sys
from Load_Controllers import Load_BBB, Load_Demonstrator
from multiple_tasks import get_task_on_MUJOCO_environment

import sys
sys.path.insert(0,'./../')
from Housekeeping import *


def Validate_BBB_Controller(domain_name, task_identity, window_size, number_demonstrations):
    configuration_identity = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/window_size_' + str(window_size) + '/' + str(task_identity) + '/'
    #tf.reset_default_graph()
    copycat_graph = tf.Graph()
    with copycat_graph.as_default():
        copycat_controller = Load_BBB(controller_identity=configuration_identity)

    file_to_save_logs = configuration_identity + 'validation_logs.pkl'

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
            time_step = 0.0
            
            observation = np.append(observation, time_step) # add time step feature
            moving_window_x = np.zeros((1, copycat_controller.moving_windows_x_size))
            moving_window_x[0, -observation.shape[0]:] = observation
            
            behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(moving_window_x, copycat_controller.mean_x, copycat_controller.deviation_x)})
            behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
            maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            behavior_dev_control = behavior_dev_control * copycat_controller.deviation_y
            
            demonstrator_control = demonstrator_controller.sess.run(demonstrator_controller.output_action_node, feed_dict={demonstrator_controller.scaled_observation_node: (observation.reshape(1,-1) - demonstrator_controller.offset) * demonstrator_controller.scale})

            while not finish:
                all_observations.append(observation)

                all_behavior_control_means.append(behavior_mean_control)
                all_behavior_control_deviations.append(behavior_dev_control)
                all_demonstrator_controls.append(demonstrator_control)

                #all_target_control_means.append(target_mean_control)
                #all_target_control_deviations.append(target_var_control)

                observation, reward, finish, info = env.step(behavior_mean_control)

                all_behavior_rewards.append(reward)

                time_step += 1e-3
                observation = np.append(observation, time_step)
                #target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

                if not window_size == 1:
                    moving_window_x[0, :-copycat_controller.drift_per_time_step] = moving_window_x[0, copycat_controller.drift_per_time_step:]
                    moving_window_x[0, -copycat_controller.drift_per_time_step:-(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                    moving_window_x[0, -(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = reward      

                moving_window_x[0, -observation.shape[0]:] = observation
                
                behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(moving_window_x, copycat_controller.mean_x, copycat_controller.deviation_x)})
                behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
                maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
                behavior_dev_control = behavior_dev_control * copycat_controller.deviation_y

                demonstrator_control = demonstrator_controller.sess.run(demonstrator_controller.output_action_node, feed_dict={demonstrator_controller.scaled_observation_node: (observation.reshape(1,-1) - demonstrator_controller.offset) * demonstrator_controller.scale})

            all_observations = np.array(all_observations)
            all_behavior_control_means = np.concatenate(all_behavior_control_means, axis=0)
            all_behavior_rewards =  np.array(all_behavior_rewards)
            all_behavior_control_deviations = np.concatenate(all_behavior_control_deviations, axis=0)
            all_demonstrator_controls = np.array(all_demonstrator_controls)

            logs_for_a_task[str(validation_trial)] = {OBSERVATIONS_LOG_KEY: all_observations, BEHAVIORAL_CONTROL_MEANS_LOG_KEY: all_behavior_control_means,
                                                     BEHAVIORAL_CONTROL_REWARDS_LOG_KEY: all_behavior_rewards, BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY: all_behavior_control_deviations,
                                                     TARGET_CONTROL_MEANS_LOG_KEY: all_demonstrator_controls}
        logs_for_all_tasks[str(task_to_validate)] = logs_for_a_task
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_tasks, f, protocol=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain_name', type=str, help='MuJoCo domain', choices=['HalfCheetah', 'Swimmer'])
    parser.add_argument('-t', '--task_identity', type=int, help='Task Identity', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=2)
    parser.add_argument('-nd', '--number_demonstrations', type=int, help='Number demonstrations per request', default=10)
    args = parser.parse_args()

    print(GREEN('Domain is ' + args.domain_name + ', task is ' + str(args.task_identity) + ', window size is ' + str(args.window_size)))
    
    Validate_BBB_Controller(domain_name=args.domain_name, task_identity=args.task_identity, window_size=args.window_size, number_demonstrations=args.number_demonstrations)