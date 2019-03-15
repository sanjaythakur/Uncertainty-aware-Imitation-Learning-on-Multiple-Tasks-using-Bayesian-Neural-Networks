import numpy as np
import random, os
import argparse
import copy
import tensorflow as tf
#from datetime import datetime

from Load_Controllers import Load_BBB, Load_Demonstrator

from tqdm import tqdm
from multiple_tasks import get_task_on_MUJOCO_environment
from Dataset import getDemonstrationsFromTask
import _pickle as pickle

import sys
sys.path.insert(0, './../')
from Housekeeping import *
from BBBNNRegression import BBBNNRegression
from Dataset import getDemonstrationsFromTask
from Detector import Detector


def train_BBB(data_x, data_y, configuration_identity, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis):
    directory_to_save_tensorboard_data = configuration_identity + TENSORBOARD_DIRECTORY
    saved_models_during_iterations_bbb = configuration_identity + SAVED_MODELS_DURING_ITERATIONS_DIRECTORY
    saved_final_model_bbb = configuration_identity + SAVED_FINAL_MODEL_DIRECTORY
  
    if not os.path.exists(directory_to_save_tensorboard_data):
        os.makedirs(directory_to_save_tensorboard_data)
    if not os.path.exists(saved_models_during_iterations_bbb):
        os.makedirs(saved_models_during_iterations_bbb)
    if not os.path.exists(saved_final_model_bbb):
        os.makedirs(saved_final_model_bbb)

    controller_graph = tf.Graph()
    with controller_graph.as_default():
        BBB_Regressor=BBBNNRegression(number_mini_batches=number_mini_batches, number_features=data_x.shape[1], number_output_units=data_y.shape[1], activation_unit=activation_unit, learning_rate=learning_rate,
                                         hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha,
                                             weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
                                                 weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(directory_to_save_tensorboard_data, sess.graph)
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)
            previous_minimum_loss = sys.float_info.max
            mini_batch_size = int(data_x.shape[0]/number_mini_batches)
            for epoch_iterator in tqdm(range(epochs)):
                data_x, data_y = randomize(data_x, data_y)
                ptr = 0
                for mini_batch_iterator in range(number_mini_batches):
                    x_batch = data_x[ptr:ptr+mini_batch_size, :]
                    y_batch = data_y[ptr:ptr+mini_batch_size, :]
                    _, loss, summary = sess.run([BBB_Regressor.train(), BBB_Regressor.getMeanSquaredError(), BBB_Regressor.summarize()], feed_dict={BBB_Regressor.X_input:x_batch, BBB_Regressor.Y_input:y_batch})
                    sess.run(BBB_Regressor.update_mini_batch_index())
                    if loss < previous_minimum_loss:
                        saver.save(sess, saved_models_during_iterations_bbb + 'iteration', global_step=epoch_iterator, write_meta_graph=False)
                        previous_minimum_loss = loss
                    ptr += mini_batch_size
                    writer.add_summary(summary, global_step=tf.train.global_step(sess, BBB_Regressor.global_step))
                #if epoch_iterator % 2 == 0:
                #    print(BLUE('Training progress: ' + str(epoch_iterator) + '/' + str(epochs)))     
            writer.close()
            saver.save(sess, saved_final_model_bbb + 'final', write_state=False)


def validate_BBB(domain_name, controller_identity, configuration_identity):
    #tf.reset_default_graph()
    copycat_graph = tf.Graph()
    with copycat_graph.as_default():
        copycat_controller = Load_BBB(controller_identity=controller_identity)

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
            
            behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
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

                if not copycat_controller.window_size == 1:
                    moving_window_x[0, :-copycat_controller.drift_per_time_step] = moving_window_x[0, copycat_controller.drift_per_time_step:]
                    moving_window_x[0, -copycat_controller.drift_per_time_step:-(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                    moving_window_x[0, -(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = reward      

                moving_window_x[0, -observation.shape[0]:] = observation
                
                behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
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


def run_on_itself(domain_name, task_identities, controller_identity, detector=None):
    #tf.reset_default_graph()
    copycat_graph = tf.Graph()
    with copycat_graph.as_default():
        copycat_controller = Load_BBB(controller_identity=controller_identity)

    stats = {}
    behavior_deviations_across_tasks = []
    for task_identity in task_identities:
        #isSafe = True
        all_behavior_deviation = []
        
        env = get_task_on_MUJOCO_environment(env_name=domain_name, task_identity=str(task_identity))
        total_cost = total_variance = 0.
        observation = env.reset()
        finish = False
        time_step = 0.0
        
        observation = np.append(observation, time_step) # add time step feature
        moving_window_x = np.zeros((1, copycat_controller.moving_windows_x_size))
        moving_window_x[0, -observation.shape[0]:] = observation
        
        behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
        behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
        maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
        minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
        behavior_dev_control = behavior_dev_control * copycat_controller.deviation_y

        while not finish:
            all_behavior_deviation.append(behavior_dev_control)
            if not detector is None:
                isSafe, stats = detector.isSafeToContinue(behavior_dev_control)
                if not str_to_bool(isSafe):
                    return 'False', np.mean(all_behavior_deviation), stats

            observation, reward, finish, info = env.step(behavior_mean_control)

            time_step += 1e-3
            observation = np.append(observation, time_step)
            #target_mean_control, target_var_control = -1. * np.dot(K, observation), np.array([[0.]])

            if not copycat_controller.window_size == 1:
                moving_window_x[0, :-copycat_controller.drift_per_time_step] = moving_window_x[0, copycat_controller.drift_per_time_step:]
                moving_window_x[0, -copycat_controller.drift_per_time_step:-(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = behavior_mean_control[0]
                moving_window_x[0, -(copycat_controller.drift_per_time_step-behavior_mean_control.shape[1])] = reward      

            moving_window_x[0, -observation.shape[0]:] = observation
            
            behavior_mean_control, behavior_dev_control, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(copy.deepcopy(moving_window_x), copycat_controller.mean_x, copycat_controller.deviation_x)})
            behavior_mean_control = REVERSE_NORMALIZE(behavior_mean_control, copycat_controller.mean_y, copycat_controller.deviation_y)
            maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
            behavior_dev_control = behavior_dev_control * copycat_controller.deviation_y

        behavior_deviations_across_tasks.append(np.mean(all_behavior_deviation))

    return 'True', np.max(behavior_deviations_across_tasks), stats


def data_efficient_imitation_across_multiple_tasks(experiment_type, controller, domain_name, window_size, number_demonstrations, adapt_detector_threshold, start_monitoring_at_time_step, detector_c, detector_m, initial_detector_threshold, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis, simulation_iteration_onset, total_simulation_runs):
    COPY_OF_ALL_MUJOCO_TASK_IDENTITIES = copy.deepcopy(ALL_MUJOCO_TASK_IDENTITIES)
    if experiment_type == 'active_learning_proof_of_concept': simulation_runs = 1
    else: simulation_runs = total_simulation_runs
    for simulation_iterator in range(simulation_iteration_onset, simulation_runs):
        if experiment_type == 'active_learning_proof_of_concept':
            if domain_name == 'HalfCheetah':
                COPY_OF_ALL_MUJOCO_TASK_IDENTITIES = np.array([8, 3, 7])
            elif domain_name == 'Swimmer':
                COPY_OF_ALL_MUJOCO_TASK_IDENTITIES = np.array([3, 4, 0])
        else:
            random.seed(simulation_iterator)
            random.shuffle(COPY_OF_ALL_MUJOCO_TASK_IDENTITIES)
        
        ###### Naive Controller ######
        if controller == 'NAIVE':
            print(RED('NAIVE controller invoked'))
            all_gathered_x, all_gathered_y = None, None
            tasks_trained_on, task_iterator_trained_on = [], []
            
            print(GREEN('Starting runs for the naive controller'))
            for task_iterator, current_task_identity in enumerate(COPY_OF_ALL_MUJOCO_TASK_IDENTITIES):
                print(RED('Simulation iteration is ' + str(simulation_iterator) + ' and task iterator is ' + str(task_iterator)))

                tasks_trained_on.append(current_task_identity)
                task_iterator_trained_on.append(task_iterator)
                moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationsFromTask(domain_name=domain_name, task_identity=current_task_identity, window_size=window_size, number_demonstrations=number_demonstrations)
           
                if all_gathered_x is None:
                    all_gathered_x, all_gathered_y = copy.deepcopy(moving_windows_x), copy.deepcopy(moving_windows_y)
                else:
                    all_gathered_x, all_gathered_y = np.append(all_gathered_x, moving_windows_x, axis=0), np.append(all_gathered_y, moving_windows_y, axis=0)

                disposible_training_x, disposible_training_y = copy.deepcopy(all_gathered_x), copy.deepcopy(all_gathered_y)
                mean_x, deviation_x = get_mean_and_deviation(data = disposible_training_x)
                disposible_training_x = NORMALIZE(disposible_training_x, mean_x, deviation_x)
                mean_y, deviation_y = get_mean_and_deviation(data = disposible_training_y)
                disposible_training_y = NORMALIZE(disposible_training_y, mean_y, deviation_y)

                configuration_identity = 'logs/' + domain_name + '/' + str(number_demonstrations) + '/naive_controller/' + experiment_type + '/' + str(simulation_iterator) + '/' + str(task_iterator) + '/'
                training_logs_configuration_identity = configuration_identity + 'training/'
                if not os.path.exists(training_logs_configuration_identity):
                    os.makedirs(training_logs_configuration_identity)

                file_name_to_save_meta_data = training_logs_configuration_identity + 'training_meta_data.pkl'
                meta_data_to_store = {MEAN_KEY_X: mean_x, DEVIATION_KEY_X: deviation_x, MEAN_KEY_Y:mean_y, DEVIATION_KEY_Y:deviation_y,
                                      DRIFT_PER_TIME_STEP_KEY: drift_per_time_step, MOVING_WINDOWS_X_SIZE_KEY: moving_windows_x_size,
                                      WINDOW_SIZE_KEY: window_size}
                with open(file_name_to_save_meta_data, 'wb') as f:
                    pickle.dump(meta_data_to_store, f)

                print(BLUE('Training phase'))
                train_BBB(data_x=disposible_training_x, data_y=disposible_training_y, configuration_identity=training_logs_configuration_identity, epochs=epochs, number_mini_batches=number_mini_batches, activation_unit=activation_unit,
                 learning_rate=learning_rate, hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha, weights_prior_mean_1=weights_prior_mean_1,
                  weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1, weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)

                meta_data_file_for_this_run = 'logs/' + domain_name + '/' + str(number_demonstrations) + '/naive_controller/' + experiment_type + '/' + str(simulation_iterator) + '/meta_data.pkl'
                meta_data_for_this_run = {TRAINING_TASK_ITERATION_KEY: task_iterator_trained_on, TASKS_TRAINED_ON_KEY: tasks_trained_on}
                with open(meta_data_file_for_this_run, 'wb') as f:
                    pickle.dump(meta_data_for_this_run, f)

                print(BLUE('Validation phase'))
                validate_BBB(domain_name=domain_name, controller_identity=configuration_identity, configuration_identity=configuration_identity)
            

        ###### BBB Controller ######
        if controller == 'BBB':
            print(RED('BBB controller invoked'))
            stats='first_run'
            did_succeed = False
            all_gathered_x, all_gathered_y = None, None
            tasks_trained_on, tasks_encountered, task_iterator_trained_on, all_thresholds, all_stats = [], [], [], [], []
            current_task_identity = COPY_OF_ALL_MUJOCO_TASK_IDENTITIES[0]
            tasks_encountered.append(current_task_identity)
            detector = Detector(domain_name=domain_name, start_monitoring_at_time_step=start_monitoring_at_time_step, initial_threshold=initial_detector_threshold, detector_m=detector_m, detector_c=detector_c)

            print(GREEN('Starting runs for the BBB controller'))
            for task_iterator in range(len(COPY_OF_ALL_MUJOCO_TASK_IDENTITIES)):
                print(RED('Simulation iteration is ' + str(simulation_iterator) + ', task iterator is ' + str(task_iterator) + ', and current task is ' + str(current_task_identity)))
                detector.reset()

                configuration_identity = 'logs/' + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/detector_c_' + str(detector_c) + '_detector_m_' + str(detector_m) + '/' + experiment_type + '/' + str(simulation_iterator) + '/' + str(task_iterator) + '/'
                if not os.path.exists(configuration_identity):
                    os.makedirs(configuration_identity)

                if not did_succeed:
                    training_logs_configuration_identity = configuration_identity + 'training/'
                    if not os.path.exists(training_logs_configuration_identity):
                        os.makedirs(training_logs_configuration_identity)

                    current_controllers_identity = configuration_identity

                    tasks_trained_on.append(current_task_identity)
                    task_iterator_trained_on.append(task_iterator)

                    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationsFromTask(domain_name=domain_name, task_identity=current_task_identity, window_size=window_size, number_demonstrations=number_demonstrations)              
                    if all_gathered_x is None:
                        all_gathered_x, all_gathered_y = copy.deepcopy(moving_windows_x), copy.deepcopy(moving_windows_y)
                    else:
                        all_gathered_x, all_gathered_y = np.append(all_gathered_x, moving_windows_x, axis=0), np.append(all_gathered_y, moving_windows_y, axis=0)
                    disposible_training_x, disposible_training_y = copy.deepcopy(all_gathered_x), copy.deepcopy(all_gathered_y)
                    mean_x, deviation_x = get_mean_and_deviation(data = disposible_training_x)
                    disposible_training_x = NORMALIZE(disposible_training_x, mean_x, deviation_x)
                    mean_y, deviation_y = get_mean_and_deviation(data = disposible_training_y)
                    disposible_training_y = NORMALIZE(disposible_training_y, mean_y, deviation_y)

                    file_name_to_save_meta_data = training_logs_configuration_identity + 'training_meta_data.pkl'
                    #meta_data_to_store = {MEAN_KEY_X: mean_x, DEVIATION_KEY_X: deviation_x, MEAN_KEY_Y:mean_y, DEVIATION_KEY_Y:deviation_y,
                    #                      DRIFT_PER_TIME_STEP_KEY: drift_per_time_step, MOVING_WINDOWS_X_SIZE_KEY: moving_windows_x_size,
                    #                      WINDOW_SIZE_KEY: window_size, TASKS_TRAINED_ON_KEY: tasks_trained_on, TASKS_ENCOUNTERED_KEY: tasks_encountered,
                    #                      STATS_KEY: stats}
                    meta_data_to_store = {MEAN_KEY_X: mean_x, DEVIATION_KEY_X: deviation_x, MEAN_KEY_Y:mean_y, DEVIATION_KEY_Y:deviation_y,
                                          DRIFT_PER_TIME_STEP_KEY: drift_per_time_step, MOVING_WINDOWS_X_SIZE_KEY: moving_windows_x_size,
                                          WINDOW_SIZE_KEY: window_size}
                    with open(file_name_to_save_meta_data, 'wb') as f:
                        pickle.dump(meta_data_to_store, f)

                    print(BLUE('Training phase'))
                    train_BBB(data_x=disposible_training_x, data_y=disposible_training_y, configuration_identity=training_logs_configuration_identity, epochs=epochs, number_mini_batches=number_mini_batches,
                     activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha,
                      weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1, weights_prior_deviation_2=weights_prior_deviation_2,
                       mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)
                    
                    #_, average_uncertainty, _ = run_on_itself(domain_name=domain_name, task_identities=current_task_identity, controller_identity= current_controllers_identity)
                    _, average_uncertainty, _ = run_on_itself(domain_name=domain_name, task_identities=tasks_trained_on, controller_identity= current_controllers_identity)
                    #### Ground the threshold according to the quantitative value of uncertainty on the current task ####
                    if adapt_detector_threshold:
                        detector.threshold = average_uncertainty
                    all_thresholds.append(detector.threshold)

                all_stats.append(stats)
                meta_data_file_for_this_run = 'logs/' + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/detector_c_' + str(detector_c) + '_detector_m_' + str(detector_m) + '/' + experiment_type + '/' + str(simulation_iterator) + '/meta_data.pkl'
                meta_data_for_this_run = {TRAINING_TASK_ITERATION_KEY: task_iterator_trained_on, DETECTOR_THRESHOLD_KEY: all_thresholds, TASKS_TRAINED_ON_KEY: tasks_trained_on, TASKS_ENCOUNTERED_KEY: tasks_encountered, STATS_KEY: all_stats}
                with open(meta_data_file_for_this_run, 'wb') as f:
                    pickle.dump(meta_data_for_this_run, f)
                #need_training = False

                print(BLUE('Validation phase'))
                validate_BBB(domain_name=domain_name, controller_identity=current_controllers_identity, configuration_identity=configuration_identity)

                if task_iterator == (len(COPY_OF_ALL_MUJOCO_TASK_IDENTITIES) - 1):
                    break

                current_task_identity = COPY_OF_ALL_MUJOCO_TASK_IDENTITIES[task_iterator + 1]
                tasks_encountered.append(current_task_identity)
                #did_succeed, average_uncertainty, stats = run_on_itself(domain_name=domain_name, task_identities=current_task_identity, controller_identity=current_controllers_identity, detector=detector)
                did_succeed, average_uncertainty, stats = run_on_itself(domain_name=domain_name, task_identities=[current_task_identity], controller_identity=current_controllers_identity, detector=detector)
                did_succeed = str_to_bool(did_succeed)
                #if not did_succeed:
                #    need_training = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-et', '--experiment_type', type=str, help='Experiment Type', choices=['active_learning_proof_of_concept', 'data_efficient_active_learning'])
    parser.add_argument('-c', '--controller_type', type=str, help='Controller Type', choices=['NAIVE', 'BBB'])
    parser.add_argument('-dn', '--domain_name', type=str, help='MuJoCo domain', choices=['HalfCheetah', 'Swimmer'])
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=2)
    parser.add_argument('-nd', '--number_demonstrations', type=int, help='Number demonstrations per request', default=10)
    parser.add_argument('-adt', '--adapt_detector_threshold', type=str, help='Adaptive Detector Threshold', choices=['True', 'False'], default='True')
    parser.add_argument('-mts', '--start_monitoring_at_time_step', type=int, help='Time step to start monitoring BBB controller uncertainty', default=200)
    parser.add_argument('-dc', '--detector_c', type=float, help='Scaling factor for the detector threshold', default=2.5)
    parser.add_argument('-dm', '--detector_m', type=int, help='Number of last few time-steps to smoothen predictive uncertainty', default=50)
    parser.add_argument('-idt', '--initial_detector_threshold', type=float, help='The detector threshold to start with or the value of non-adaptive threshold', default=0.3)
    parser.add_argument('-sio', '--simulation_iteration_onset', type=int, help='simulation_iteration_onset', default=0)
    parser.add_argument('-tsr', '--total_simulation_runs', type=int, help='Total Simulation Runs', default=5)

    args = parser.parse_args()

    data_efficient_imitation_across_multiple_tasks(experiment_type=args.experiment_type, controller=args.controller_type, domain_name=args.domain_name, window_size=args.window_size, number_demonstrations=args.number_demonstrations, adapt_detector_threshold=str_to_bool(args.adapt_detector_threshold),
     start_monitoring_at_time_step=args.start_monitoring_at_time_step, detector_c=args.detector_c, detector_m=args.detector_m, initial_detector_threshold=args.initial_detector_threshold, epochs = 10001, number_mini_batches = 20,
      activation_unit = 'RELU', learning_rate = 0.001, hidden_units= [90, 30, 10], number_samples_variance_reduction = 25, precision_alpha = 0.01,
       weights_prior_mean_1 = 0., weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4, weights_prior_deviation_2 = 0.4, mixture_pie = 0.7, rho_mean = -3.,
        extra_likelihood_emphasis = 10000000000000000., simulation_iteration_onset=args.simulation_iteration_onset, total_simulation_runs=args.total_simulation_runs)
