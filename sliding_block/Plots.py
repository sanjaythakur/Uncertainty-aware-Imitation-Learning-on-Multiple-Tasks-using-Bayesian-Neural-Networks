#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')
from Housekeeping import *


def final_flourish(all_task_configurations, uncertainty_ylim, cost_ylim, predictive_error_ylim):
    demonstrator_logs_file = LOGS_DIRECTORY + 'demonstrator_logs.pkl'
    with open(demonstrator_logs_file, 'rb') as f:
        demonstrator_data = pickle.load(f)
    iterator_GP, iterator_BBB = False, False
    for configuration in all_task_configurations:
        file_to_load_data_from = LOGS_DIRECTORY + configuration[CONTEXT_CODE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
        #configuration_label = 'Block Mass:' + str(get_sliding_block_context_from_code(int(configuration[CONTEXT_CODE_KEY]))) + ', window size:' + configuration[WINDOW_SIZE_KEY] + ', partial obs.:' + configuration[PARTIAL_OBSERVABILITY_KEY] + ', controller:' + configuration[BEHAVIORAL_CONTROLLER_KEY]
        configuration_label = configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller, window size:' + configuration[WINDOW_SIZE_KEY]
        with open(file_to_load_data_from, 'rb') as f:
            loaded_data = pickle.load(f)
        all_tasks = list(loaded_data.keys())
        all_initial_states = list(loaded_data[all_tasks[0]].keys())
        behavioral_average_task_deviations = []
        behavioral_average_task_costs = []
        average_predictive_error = []
        
        for task in all_tasks:
            behavioral_task_deviations = []
            behavioral_task_costs = []
            predictive_error = []

            for initial_state in all_initial_states:
                #observations = loaded_data[task][initial_state][OBSERVATIONS_LOG_KEY]
                behavioral_control_means = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_MEANS_LOG_KEY]
                target_control_means = loaded_data[task][initial_state][TARGET_CONTROL_MEANS_LOG_KEY]
                behavioral_control_deviations = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY]
                #behavioral_control_costs = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_COSTS_LOG_KEY]/demonstrator_data[task][initial_state][DEMONSTRATOR_COSTS_LOG_KEY]
                behavioral_control_costs = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_COSTS_LOG_KEY]
                behavioral_task_deviations.append(behavioral_control_deviations.squeeze())
                behavioral_task_costs.append(behavioral_control_costs.squeeze())
                predictive_error.append(np.mean(np.square(behavioral_control_means-target_control_means)))

            behavioral_task_deviations = np.stack(behavioral_task_deviations)
            behavioral_task_costs = np.stack(behavioral_task_costs)
            behavioral_average_task_deviations.append(np.mean(np.sum(behavioral_task_deviations, axis=1)))
            behavioral_average_task_costs.append(np.mean(np.sum(behavioral_task_costs, axis=1)))
            average_predictive_error.append(np.mean(predictive_error))
        #maximum_cost = np.amax(behavioral_average_task_costs)
        #behavioral_average_task_rewards = np.array([((-1.*each_cost)+maximum_cost) for each_cost in behavioral_average_task_costs])
        #behavioral_average_task_rewards = behavioral_average_task_costs
        if configuration[BEHAVIORAL_CONTROLLER_KEY] == 'GP':          
            if iterator_GP:
                plt.plot(all_tasks, average_predictive_error, label=configuration_label, color=COLOR['BBBvsGP_GP'], ls = '--', lw=0.8)
            else:
                iterator_GP = True
                plt.plot(all_tasks, average_predictive_error, label=configuration_label, color=COLOR['BBBvsGP_GP'], lw=0.8)
        else:
            if iterator_BBB:
                plt.plot(all_tasks, average_predictive_error, label=configuration_label, color=COLOR['BBBvsGP_BBB'], ls = '--', lw=0.8)  
            else:
                iterator_BBB = True
                plt.plot(all_tasks, average_predictive_error, label=configuration_label, color=COLOR['BBBvsGP_BBB'], lw=0.8)  
        
    plt.xlabel('Block Mass')
    plt.xticks(np.linspace(1., 100., 10))
    plt.ylabel('Predictive Mean Squared Error')
    #plt.yticks(np.arange(0., 3.0, 0.2))
    plt.yscale('log')
    plt.ylim(0., predictive_error_ylim)
    plt.title('Training is done on block mass of ' + str(get_sliding_block_masses_from_task_identifier_as_string(task_identifier=int(configuration[CONTEXT_CODE_KEY]))), fontsize=16)
    plt.legend()

    plt.show()    


def compare_generalization_in_controllers(all_task_configurations, uncertainty_ylim, cost_ylim, predictive_error_ylim):
    demonstrator_logs_file = LOGS_DIRECTORY + 'demonstrator_logs.pkl'
    with open(demonstrator_logs_file, 'rb') as f:
        demonstrator_data = pickle.load(f)
    
    fig = plt.figure()
    ax_1 = fig.add_subplot(321, frameon=False)
    ax_2 = fig.add_subplot(322, frameon=False)
    ax_3 = fig.add_subplot(323, frameon=False)
    gp_exists = False
    for iterator, configuration in enumerate(all_task_configurations):
        file_to_load_data_from = LOGS_DIRECTORY + configuration[CONTEXT_CODE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
        #configuration_label = 'Block Mass:' + str(get_sliding_block_context_from_code(int(configuration[CONTEXT_CODE_KEY]))) + ', window size:' + configuration[WINDOW_SIZE_KEY] + ', partial obs.:' + configuration[PARTIAL_OBSERVABILITY_KEY] + ', controller:' + configuration[BEHAVIORAL_CONTROLLER_KEY]
        configuration_label = configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller, trained on block mass ' + str(get_sliding_block_masses_from_task_identifier_as_string(task_identifier=int(configuration[CONTEXT_CODE_KEY]))) + configuration[PARTIAL_OBSERVABILITY_KEY]
        with open(file_to_load_data_from, 'rb') as f:
            loaded_data = pickle.load(f)
        all_tasks = list(loaded_data.keys())
        all_initial_states = list(loaded_data[all_tasks[0]].keys())
        behavioral_average_task_deviations = []
        behavioral_average_task_costs = []
        average_predictive_error = []
        
        if configuration[BEHAVIORAL_CONTROLLER_KEY] == 'GP':
            if not gp_exists:
                gp_exists = True
                ax_4 = fig.add_subplot(324, frameon=False)
                ax_5 = fig.add_subplot(325, frameon=False)
            gp_fit_logs_file = LOGS_DIRECTORY + configuration[CONTEXT_CODE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '_fit.pkl'
            with open(gp_fit_logs_file, 'rb') as f:
                gp_fit_logs_data = pickle.load(f)
            unoptimized_gp_fit_data = gp_fit_logs_data[UNOPTIMIZED_GP_FIT_KEY]
            unoptimized_gp_trainables_data  = gp_fit_logs_data[UNOPTIMIZED_GP_TRAINABLES_KEY]
            optimized_gp_fit_data = gp_fit_logs_data[OPTIMIZED_GP_FIT_KEY]
            optimized_gp_trainables_data  = gp_fit_logs_data[OPTIMIZED_GP_TRAINABLES_KEY]
            print(RED('Trainables before optimization'))
            print(unoptimized_gp_trainables_data)
            print(RED('Trainables after maximization of marginalized likelihood by marginalizing GP hyperparameters'))
            print(optimized_gp_trainables_data)
            mean_gp_fit_predictive_error = [unoptimized_gp_fit_data[MEAN_GP_FIT_PREDICTIVE_ERROR_KEY], optimized_gp_fit_data[MEAN_GP_FIT_PREDICTIVE_ERROR_KEY]]
            mean_gp_fit_predictive_variance = [unoptimized_gp_fit_data[MEAN_GP_FIT_VARIANCE_KEY], optimized_gp_fit_data[MEAN_GP_FIT_VARIANCE_KEY]]
            
            x_bar_labels = ['Unoptimized', 'Optimized']
            x_bar_ticks_1 = np.arange(len(x_bar_labels))
            x_bar_ticks_2 = [x + barwidth for x in x_bar_ticks_1]
            ax_4.bar(x_bar_labels, mean_gp_fit_predictive_error, width=barwidth/1.25, edgecolor='white', label=configuration_label)
            ax_5.bar(x_bar_labels, mean_gp_fit_predictive_variance, width=barwidth/1.25, edgecolor='white', label=configuration_label)
                        


        for task in all_tasks:
            behavioral_task_deviations = []
            behavioral_task_costs = []
            predictive_error = []

            for initial_state in all_initial_states:
                #observations = loaded_data[task][initial_state][OBSERVATIONS_LOG_KEY]
                behavioral_control_means = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_MEANS_LOG_KEY]
                target_control_means = loaded_data[task][initial_state][TARGET_CONTROL_MEANS_LOG_KEY]
                behavioral_control_deviations = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY]
                #behavioral_control_costs = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_COSTS_LOG_KEY]/demonstrator_data[task][initial_state][DEMONSTRATOR_COSTS_LOG_KEY]
                behavioral_control_costs = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_COSTS_LOG_KEY]
                behavioral_task_deviations.append(behavioral_control_deviations.squeeze())
                behavioral_task_costs.append(behavioral_control_costs.squeeze())
                predictive_error.append(np.mean(np.square(behavioral_control_means-target_control_means)))

            behavioral_task_deviations = np.stack(behavioral_task_deviations)
            behavioral_task_costs = np.stack(behavioral_task_costs)
            behavioral_average_task_deviations.append(np.mean(np.sum(behavioral_task_deviations, axis=1)))
            behavioral_average_task_costs.append(np.mean(np.sum(behavioral_task_costs, axis=1)))
            average_predictive_error.append(np.mean(predictive_error))
        #maximum_cost = np.amax(behavioral_average_task_costs)
        #behavioral_average_task_rewards = np.array([((-1.*each_cost)+maximum_cost) for each_cost in behavioral_average_task_costs])
        #behavioral_average_task_rewards = behavioral_average_task_costs
        ax_1.plot(all_tasks, behavioral_average_task_deviations, label=configuration_label, color=matplotlibcolors[iterator])
        ax_2.plot(all_tasks, behavioral_average_task_costs, label=configuration_label, color=matplotlibcolors[iterator])
        ax_3.plot(all_tasks, average_predictive_error, label=configuration_label, color=matplotlibcolors[iterator])
    ax_1.set_xlabel('Block Mass', fontweight='bold')
    ax_1.set_xticks(np.linspace(1., 100., 10))
    ax_1.set_ylabel('Standard Deviation', fontweight='bold')
    #ax_1.set_yticks(np.arange(0., 3.0, 0.2))
    #ax_1.set_yscale('log')
    ax_1.set_ylim(0., uncertainty_ylim)
    ax_1.set_title('Uncertainty on tasks')
    ax_1.legend()
    ax_2.set_xlabel('Block Mass', fontweight='bold')
    ax_2.set_xticks(np.linspace(1., 100., 10))
    ax_2.set_ylabel('Episodic Costs', fontweight='bold')
    #ax_2.set_yticks(np.arange(0., 3.0, 0.2))
    ax_2.set_yscale('log')
    ax_2.set_ylim(bottom=0., top=cost_ylim)
    ax_2.set_title('Sliding Block on ice')
    ax_2.legend()
    
    ax_3.set_xlabel('Block Mass', fontweight='bold')
    ax_3.set_xticks(np.linspace(1., 100., 10))
    ax_3.set_ylabel('Predictive Mean Squared Error', fontweight='bold')
    #ax_3.set_yticks(np.arange(0., 3.0, 0.2))
    ax_3.set_yscale('log')
    ax_3.set_ylim(0., predictive_error_ylim)
    ax_3.set_title('Predictive Errors')
    ax_3.legend()
    
    #ax_4.set_xlabel('group', fontweight='bold')
    ax_4.set_xticks(x_bar_ticks_1, x_bar_labels)
    ax_4.set_ylabel('Predictive Mean Squared Error', fontweight='bold')
    ax_4.set_yscale('log')
    ax_4.set_title('GP Optimization Effect on Predictive Error')
    ax_4.legend()

    #ax_5.set_xlabel('group', fontweight='bold')
    ax_5.set_xticks(x_bar_ticks_1, x_bar_labels)
    ax_5.set_ylabel('Predictive Mean Variance', fontweight='bold')
    ax_5.set_yscale('log')
    ax_5.set_title('GP Optimization Effect on Predictive Variance')
    ax_5.legend()
    
    plt.show()    