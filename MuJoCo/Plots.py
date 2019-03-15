import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')
from Detector import get_smoothed_transform
from Housekeeping import *

def summarize_mujoco_runs(domain_name, all_configurations, simulation_iterators=[0, 1, 2, 3, 4]):
    for configuration in all_configurations:
        if configuration['controller'] == 'BBB': 
            print(BLUE('Configuration is ' + configuration['controller'] + ', detector_c is ' + str(configuration['detector_c']) + ', detector_m is ' + str(configuration['detector_m']) + ', #demonstrations are ' + str(configuration['number_demonstrations'])))
            for simulation_iterator in simulation_iterators:
                try:    
                    log_file = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/data_efficient_active_learning/' + str(simulation_iterator) + '/meta_data.pkl'
                    with open(log_file, 'rb') as f:
                        logged_evaluation_data = pickle.load(f)
                    adaptive_thresholds = logged_evaluation_data[DETECTOR_THRESHOLD_KEY]
                    tasks_trained_on_already = logged_evaluation_data[TASKS_TRAINED_ON_KEY]
                    tasks_encountered_so_far = logged_evaluation_data[TASKS_ENCOUNTERED_KEY]
                    tasks_training_iterations_so_far = logged_evaluation_data[TRAINING_TASK_ITERATION_KEY]
                    print(RED('Simulation iteration is ' + str(simulation_iterator)))
                    print('Adaptive thresholds are ' + str(adaptive_thresholds))
                    print('Tasks trained on already are ' + str(tasks_trained_on_already))
                    print('Tasks encountered so far are ' + str(tasks_encountered_so_far))
                    print('Training tasks iterations so far are ' + str(tasks_training_iterations_so_far))
                    print()
                except:
                    print(RED('Simulation iteration ' + str(simulation_iterator) + ' is still in progress'))
        else:
            pass
        print()
        print()


def zoom_into_every_step(domain_name, number_demonstrations, detector_c, detector_m, experiment_type, demonstration_request_to_gauge, simulation_iterator, validation_task_to_plot):

    log_file = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/detector_c_' + str(detector_c) + '_detector_m_' + str(detector_m) + '/' + experiment_type + '/' + str(simulation_iterator) + '/meta_data.pkl'
    with open(log_file, 'rb') as f:
        logged_evaluation_data = pickle.load(f)

    adaptive_thresholds = logged_evaluation_data[DETECTOR_THRESHOLD_KEY]
    tasks_trained_on_already = logged_evaluation_data[TASKS_TRAINED_ON_KEY]
    tasks_encountered_so_far = logged_evaluation_data[TASKS_ENCOUNTERED_KEY]
    tasks_training_iterations_so_far = logged_evaluation_data[TRAINING_TASK_ITERATION_KEY]

    validation_logs_to_probe = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/detector_c_' + str(detector_c) + '_detector_m_' + str(detector_m) + '/' + experiment_type + '/' + str(simulation_iterator) + '/' + str(logged_evaluation_data[TRAINING_TASK_ITERATION_KEY][demonstration_request_to_gauge-1]) + '/validation_logs.pkl'
    #training_meta_data_file = LOGS_DIRECTORY + domain_name + '/bbb_controller/detector_c_' + detector_c + '_detector_m_' + detector_m + '/' + str(simulation_iterator) + '/' + str(logged_evaluation_data[TRAINING_TASK_ITERATION_KEY][demonstration_request_to_gauge-1]) + '/training/training_meta_data.pkl'

    #with open(training_meta_data_file, 'rb') as f:
    #    training_meta_data = pickle.load(f)

    with open(validation_logs_to_probe, 'rb') as f:
        logged_evaluation_data = pickle.load(f)
    
    for iterator, validation_task in enumerate(validation_task_to_plot):
        for validation_trial in range(NUMBER_VALIDATION_TRIALS-1):
            episodic_deviations = logged_evaluation_data[str(validation_task)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY]
            #episodic_deviations = np.mean(np.array(episodic_deviations), axis=1)
            #plt.plot(np.arange(0, episodic_deviations.shape[0], 1), episodic_deviations, linewidth=0.6, color=matplotlibcolors[iterator], alpha=1.0)
            plt.plot(np.arange(0, episodic_deviations.shape[0], 1), get_smoothed_transform(input_array=episodic_deviations, domain_name=domain_name, smoothening_m=int(detector_m), scaling_c=float(detector_c)), linewidth=0.6, color=matplotlibcolors[iterator], alpha=1.0)
        episodic_deviations = logged_evaluation_data[str(validation_task)][str(NUMBER_VALIDATION_TRIALS-1)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY]
        #episodic_deviations = np.mean(np.array(episodic_deviations), axis=1)
        #plt.plot(np.arange(0, episodic_deviations.shape[0], 1), episodic_deviations, linewidth=0.6, color=matplotlibcolors[iterator], alpha=1.0)
        plt.plot(np.arange(0, episodic_deviations.shape[0], 1), get_smoothed_transform(input_array=episodic_deviations, domain_name=domain_name, smoothening_m=int(detector_m), scaling_c=float(detector_c)), linewidth=0.6, color=matplotlibcolors[iterator], alpha=1.0, label='Context '+str(validation_task))
    #plt.xticks(np.arange(0, len(task_rewards_to_plot), 1))          
    plt.hlines(y=(float(detector_c)*adaptive_thresholds[demonstration_request_to_gauge-1]), xmin=0, xmax=episodic_deviations.shape[0], colors='k', linestyles='solid', linewidth=0.3, label='c*$\omega$(Adaptive Threshold)')
    #plt.ylim(bottom=0.05, top=0.6) 
    plt.xlabel('$t$', fontweight='bold')
    plt.ylabel('$\sigma_d$', fontweight='bold')
    plt.title('Relation between reward and uncertainty', fontsize=20)
    plt.legend()
    plt.show()
    plt.close()

    print('Adaptive thresholds are ' + str(adaptive_thresholds))
    print('Tasks trained on already are ' + str(tasks_trained_on_already))
    print('Tasks encountered so far are ' + str(tasks_encountered_so_far))
    print('Training tasks iterations so far are ' + str(tasks_training_iterations_so_far))


def compare_data_efficiency(domain_name, all_configurations, demonstration_request_to_gauge, simulation_iterators=[0, 1, 2, 3, 4]):
    x_axis_ticks = []
    cumulative_rewards_to_plot = []
    cumulative_minimum_rewards_to_plot = []
    cumulative_maximum_rewards_to_plot = []
    for configuration in all_configurations:
        cumulative_reward_over_all_tasks_and_simulations = []
        cumulative_minimum_reward_over_all_tasks_and_simulations = []
        cumulative_maximum_reward_over_all_tasks_and_simulations = []
        for simulation_iterator in simulation_iterators:
            if configuration['controller'] == 'BBB':    
                training_meta_data_file = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/active_learning_proof_of_concept/' + str(simulation_iterator) + '/meta_data.pkl'
                with open(training_meta_data_file, 'rb') as f:
                    training_meta_data = pickle.load(f)
                validation_logs_to_probe = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/active_learning_proof_of_concept/' + str(simulation_iterator) + '/' + str(training_meta_data[TRAINING_TASK_ITERATION_KEY][demonstration_request_to_gauge-1]) + '/validation_logs.pkl'
                label = '$c=' + str(configuration['detector_c']) + '$\n$m=' + str(configuration['detector_m']) + '$'
            elif configuration['controller'] == 'NAIVE':    
                validation_logs_to_probe = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/naive_controller/active_learning_proof_of_concept/' + str(simulation_iterator) + '/' + str(demonstration_request_to_gauge-1) + '/validation_logs.pkl'
                label = 'Naive'
            cumulative_reward_over_all_tasks = 0.
            cumulative_minimum_reward_over_all_tasks = 0.
            cumulative_maximum_reward_over_all_tasks = 0.
            with open(validation_logs_to_probe, 'rb') as f:
                logged_evaluation_data = pickle.load(f)
            for validation_task in ALL_MUJOCO_TASK_IDENTITIES:
                episodic_rewards = [logged_evaluation_data[str(validation_task)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY] for validation_trial in range(NUMBER_VALIDATION_TRIALS)]
                episodic_rewards = np.array(episodic_rewards)
                cumulative_reward_over_all_tasks += np.mean(np.sum(episodic_rewards, axis=1))
                cumulative_minimum_reward_over_all_tasks += np.amin(np.sum(episodic_rewards, axis=1))
                cumulative_maximum_reward_over_all_tasks += np.amax(np.sum(episodic_rewards, axis=1))
            cumulative_reward_over_all_tasks_and_simulations.append(cumulative_reward_over_all_tasks)
            cumulative_minimum_reward_over_all_tasks_and_simulations.append(cumulative_minimum_reward_over_all_tasks)
            cumulative_maximum_reward_over_all_tasks_and_simulations.append(cumulative_maximum_reward_over_all_tasks)
        x_axis_ticks.append(label)
        cumulative_rewards_to_plot.append(np.mean(cumulative_reward_over_all_tasks_and_simulations))
        cumulative_minimum_rewards_to_plot.append(np.amin(cumulative_minimum_reward_over_all_tasks_and_simulations))
        cumulative_maximum_rewards_to_plot.append(np.amax(cumulative_maximum_reward_over_all_tasks_and_simulations))
    plt.bar(x_axis_ticks, cumulative_rewards_to_plot, yerr=np.stack((np.abs(np.subtract(cumulative_rewards_to_plot, cumulative_minimum_rewards_to_plot)), np.abs(np.subtract(cumulative_rewards_to_plot, cumulative_maximum_rewards_to_plot)))), width=barwidth)
    #plt.xlabel('Configuration', fontweight='bold')
    plt.ylabel('Cumulative Reward', fontweight='bold')
    plt.title(domain_name, fontsize=18)
    plt.legend()
    plt.show()


def compare_conservativeness_with_scatter_plot(domain_name, all_configurations, ymax, simulation_iterators=[0, 1, 2, 3, 4]):
    simulation_iterator = 0
    x_axis_ticks = []
    cumulative_rewards_to_plot = []
    cumulative_minimum_rewards_to_plot = []
    cumulative_maximum_rewards_to_plot = []
    number_requests_made_to_demonstrator = []
    for configuration in all_configurations:
        if configuration['controller'] == 'RANDOM':
            file_to_read_logs = LOGS_DIRECTORY + str(domain_name) + '_RANDOM.pkl'
            with open(file_to_read_logs, 'rb') as f:
                reward_across_tasks = np.array(pickle.load(f))
            cumulative_rewards_across_tasks = np.sum(np.mean(reward_across_tasks, axis=1))
            plt.scatter(0, cumulative_rewards_across_tasks, label='Random', color=COLOR['RANDOM'], s=75)
        else:
            cumulative_reward_over_all_tasks_and_simulations = []
            cumulative_minimum_reward_over_all_tasks_and_simulations = []
            cumulative_maximum_reward_over_all_tasks_and_simulations = []
            number_requests_made_to_demonstrator_over_all_simulations = []
            for simulation_iterator in simulation_iterators:
                try:
                    if configuration['controller'] == 'BBB':
                        training_meta_data_file = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/data_efficient_active_learning/' + str(simulation_iterator) + '/meta_data.pkl'
                        with open(training_meta_data_file, 'rb') as f:
                            training_meta_data = pickle.load(f)
                        number_requests_made_to_demonstrator_over_all_simulations.append(len(training_meta_data[TRAINING_TASK_ITERATION_KEY]))
                        validation_logs_to_probe = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/data_efficient_active_learning/' + str(simulation_iterator) + '/' + str(len(ALL_MUJOCO_TASK_IDENTITIES)-1) + '/validation_logs.pkl'
                        label = '$c=' + str(configuration['detector_c']) + ', m=' + str(configuration['detector_m']) + '$'
                        color_key = configuration['controller'] + '_' + str(configuration['detector_c']) + '_' + str(configuration['detector_m'])
                    elif configuration['controller'] == 'NAIVE':
                        validation_logs_to_probe = LOGS_DIRECTORY + domain_name + '/' + str(configuration['number_demonstrations']) + '/naive_controller/data_efficient_active_learning/' + str(simulation_iterator) + '/' + str(len(ALL_MUJOCO_TASK_IDENTITIES)-1) + '/validation_logs.pkl'
                        number_requests_made_to_demonstrator_over_all_simulations.append(len(ALL_MUJOCO_TASK_IDENTITIES))
                        label = 'Naive'
                        color_key = configuration['controller']
                    cumulative_reward_over_all_tasks = 0.
                    cumulative_minimum_reward_over_all_tasks = 0.
                    cumulative_maximum_reward_over_all_tasks = 0.
                    with open(validation_logs_to_probe, 'rb') as f:
                        logged_evaluation_data = pickle.load(f)
                    for validation_task in ALL_MUJOCO_TASK_IDENTITIES:
                        episodic_rewards = [logged_evaluation_data[str(validation_task)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY] for validation_trial in range(NUMBER_VALIDATION_TRIALS)]
                        episodic_rewards = np.array(episodic_rewards)
                        cumulative_reward_over_all_tasks += np.mean(np.sum(episodic_rewards, axis=1))
                        cumulative_minimum_reward_over_all_tasks += np.amin(np.sum(episodic_rewards, axis=1))
                        cumulative_maximum_reward_over_all_tasks += np.amax(np.sum(episodic_rewards, axis=1))
                    cumulative_reward_over_all_tasks_and_simulations.append(cumulative_reward_over_all_tasks)
                    cumulative_minimum_reward_over_all_tasks_and_simulations.append(cumulative_minimum_reward_over_all_tasks)
                    cumulative_maximum_reward_over_all_tasks_and_simulations.append(cumulative_maximum_reward_over_all_tasks)
                except:
                    pass
            x_axis_ticks.append(label)
            #cumulative_rewards_to_plot.append(np.mean(cumulative_reward_over_all_tasks_and_simulations))
            #cumulative_minimum_rewards_to_plot.append(np.amin(cumulative_minimum_reward_over_all_tasks_and_simulations))
            #cumulative_maximum_rewards_to_plot.append(np.amax(cumulative_maximum_reward_over_all_tasks_and_simulations))
            #number_requests_made_to_demonstrator.append(np.mean(number_requests_made_to_demonstrator_over_all_simulations))
            plt.scatter(np.mean(number_requests_made_to_demonstrator_over_all_simulations), np.mean(cumulative_reward_over_all_tasks_and_simulations), label=label, color=COLOR[color_key], s=75)
    plt.ylim(top=ymax)
    plt.xticks(np.arange(0, len(ALL_MUJOCO_TASK_IDENTITIES)+2, 1))
    plt.xlabel('Number of Demonstration Requests', fontweight='bold')
    plt.ylabel('Cumulative Reward', fontweight='bold')
    plt.title(domain_name, fontsize=20)
    plt.legend(loc=4, fontsize=15)
    plt.show()


def plotDemonstrators(env_name, demonstrator_rewards_over_all_contexts, identifier):
    if env_name == 'Reacher':
        threshold = -3.75
        minimim = -50.
        maximum  = 0.
        random_reward = -44.39
    elif env_name == 'InvertedPendulum':
        threshold = 950.
        minimim = 0.
        maximum = 1100.
        random_reward = 5.2
    elif env_name == 'InvertedDoublePendulum':
        threshold = 9100.
        minimim = 0.
        maximum = 10000.
        random_reward = 53.94
    elif env_name == 'Swimmer':
        threshold = 360.
        minimim = 0.
        maximum = 400.
        random_reward = 1.83
    elif env_name == 'HalfCheetah':
        threshold = 4800.
        minimim = -1000.
        maximum = 7000.
        random_reward = -288.
    elif env_name == 'Hopper':
        threshold = 3800.
        minimim = 0.
        maximum = 4100.
        random_reward = 17.84
    elif env_name == 'Walker2d':
        threshold = 0.
        minimim = 0.
        maximum = 10000.
        random_reward = 1.282601062
    elif env_name == 'Ant':
        threshold = 6000.
        minimim = 0.
        maximum = 8000.
        random_reward = 0.
    elif env_name == 'Humanoid':
        threshold = 0.
        minimim = 0.
        maximum = 8000.
        random_reward = 116.38
    elif env_name == 'HumanoidStandup':
        threshold = 0.
        minimim = 0.
        maximum = 100000.
        random_reward = 33902.78

    demonstrator_plot_directory = './../' + DEMONSTRATOR_CONTROLLER_REWARD_LOG_DIRECTORY + env_name + '/'
    if not os.path.exists(demonstrator_plot_directory):
        os.makedirs(demonstrator_plot_directory)
    demonstrator_file_name = demonstrator_plot_directory + str(identifier) + '.png'

    plt.plot(ALL_MUJOCO_TASK_IDENTITIES, demonstrator_rewards_over_all_contexts, label='Demonstrator')
    #plt.plot([ALL_MUJOCO_TASK_IDENTITIES[demonstrator_context]], [demonstrator_rewards_over_all_contexts[demonstrator_context]], 'ko', label='demonstrator context')
    plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold for success')
    plt.axhline(y=random_reward, color='g', linestyle='-', label='Random Controller')
    plt.ylim(ymin=minimim)
    plt.ylim(ymax=maximum)
    plt.xlabel('Tasks')
    plt.ylabel('Rewards')
    plt.title('Trained context is ' + str(identifier))
    plt.legend()

    plt.savefig(demonstrator_file_name)
    plt.close('all')


def BBBvsGP_generalization(domain_name, number_demonstrations, window_size, behavioral_controller):
    BBB_log_file = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/bbb_controller/window_size_' + str(window_size) + '/' + str(behavioral_controller) + '/validation_logs.pkl'
    with open(BBB_log_file, 'rb') as f:
        BBB_logs = pickle.load(f)
    BBB_rewards = []
    for validated_task in ALL_MUJOCO_TASK_IDENTITIES:
        gathered_rewards = 0.
        gathered_predictive_squared_errors = 0.
        for validation_iterator in range(NUMBER_VALIDATION_TRIALS):
            gathered_rewards += np.sum(BBB_logs[str(validated_task)][str(validation_iterator)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY])
            gathered_predictive_squared_errors += np.sum(np.square(BBB_logs[str(validated_task)][str(validation_iterator)][BEHAVIORAL_CONTROL_MEANS_LOG_KEY] - np.squeeze(BBB_logs[str(validated_task)][str(validation_iterator)][TARGET_CONTROL_MEANS_LOG_KEY])))
        BBB_rewards.append(gathered_rewards/NUMBER_VALIDATION_TRIALS)
    
    GP_log_file = LOGS_DIRECTORY + domain_name + '/' + str(number_demonstrations) + '/GP_controller/window_size_' + str(window_size) + '/' + str(behavioral_controller) + '_validation.pkl'
    with open(GP_log_file, 'rb') as f:
        GP_logs = pickle.load(f)
    GP_rewards = []
    for validated_task in ALL_MUJOCO_TASK_IDENTITIES:
        gathered_rewards = 0.
        gathered_predictive_squared_errors = 0.
        for validation_iterator in range(NUMBER_VALIDATION_TRIALS):
            gathered_rewards += np.sum(GP_logs[str(validated_task)][str(validation_iterator)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY])
            gathered_predictive_squared_errors += np.sum(np.square(GP_logs[str(validated_task)][str(validation_iterator)][BEHAVIORAL_CONTROL_MEANS_LOG_KEY] - np.squeeze(GP_logs[str(validated_task)][str(validation_iterator)][TARGET_CONTROL_MEANS_LOG_KEY])))
        GP_rewards.append(gathered_rewards/NUMBER_VALIDATION_TRIALS)
    
    plt.bar(np.arange(0, len(ALL_MUJOCO_TASK_IDENTITIES), 1), BBB_rewards, width=barwidth/1.25, label='BBB', color=COLOR['BBBvsGP_BBB'])
    plt.bar(np.arange(0, len(ALL_MUJOCO_TASK_IDENTITIES), 1)+barwidth, GP_rewards, width=barwidth/1.25, label='GP', color=COLOR['BBBvsGP_GP'])
    plt.ylabel('Episodic Reward', fontweight='bold')
    plt.xlabel('Context Index', fontweight='bold')
    plt.title('Trained on context ' + str(behavioral_controller) + ' on ' + domain_name + '\nwith temporal window size of ' + str(window_size), fontsize=22)
    plt.xticks((np.arange(0, len(ALL_MUJOCO_TASK_IDENTITIES), 1)+barwidth/2), np.arange(0, len(ALL_MUJOCO_TASK_IDENTITIES), 1))
    plt.legend(loc='best')
    plt.show()