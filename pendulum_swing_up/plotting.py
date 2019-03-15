import numpy
import matplotlib.pyplot as plt, _pickle as pickle

from Utils import *

def visualize_transfer_between_domains(expert_domain, target_domain):
    for expert_context in ALL_CONTEXTS[expert_domain].keys():
        all_costs = []
        file_name = EXPERT_CONTROLLER_PERFORMANCE_LOG_DIRECTORY + expert_domain + '/' + expert_context + '/' + target_domain + '.pkl'
        with open(file_name, 'rb') as f:
            all_costs_for_this_contextual_expert_on_the_given_target = pickle.load(f)
        for target_context in ALL_CONTEXTS[target_domain].keys(): 
            costs_for_variance_reduction = 0.0
            for iterator in range(NUMBER_SAMPLES_FOR_VARIANCE_REDUCTION):
                costs_for_variance_reduction += all_costs_for_this_contextual_expert_on_the_given_target[target_context][iterator][0]
            all_costs.append(costs_for_variance_reduction/NUMBER_SAMPLES_FOR_VARIANCE_REDUCTION)
        plt.plot(np.arange(0, len(ALL_CONTEXTS[target_domain].keys()), 1), np.array(all_costs), 'ko')
        #expert_index = list(ALL_CONTEXTS[target_domain].keys()).index(expert_context)
        #plt.plot(np.arange(0, len(ALL_CONTEXTS[target_domain].keys()), 1)[expert_index], all_costs[expert_index], 'ro', label='Trained contexts')
        plt.ylim(ymin=0.)
        #plt.ylim(ymin=0.)
        plt.xticks(np.arange(0, len(ALL_CONTEXTS[target_domain].keys()), 1))
        #plt.xticks(np.arange(0, len(ALL_CONTEXTS[target_domain].keys()), 1), list(ALL_CONTEXTS[target_domain].keys()))
        plt.xlabel('Contexts')
        plt.ylabel('Costs')
        plt.title('Trained context is ' + expert_context)
        plt.legend()
        plt.show()


def episodic_cost_vs_uncertainty(source_domain, source_task, target_domain):
    all_target_tasks = ALL_CONTEXTS[target_domain].keys()
    all_task_names = ['Context 0', 'Context 1', 'Context 2', 'Context 3', 'Context 4', 'Context 5', 'Context 6']
    #all_costs = []
    #all_deviations = []
    for target_task, task_name in zip(all_target_tasks, all_task_names):
        file_name = COPYCAT_CONTROLLER_PERFORMANCE_LOG_DIRECTORY + source_domain + '_' + source_task + '_' + target_domain + '_' + target_task + '.pkl'
        try:
            with open(file_name, 'rb') as f:
                learner_induced_data = pickle.load(f)
            all_episodic_rewards = []
            all_episodic_uncertainties = []
            for iterator in range(NUMBER_SAMPLES_FOR_VALIDATION):
                episodic_cost = np.sum(learner_induced_data[LEARNER_INDUCED_COST_Ts_KEY][str(iterator)])
                episodic_reward = -1.* episodic_cost
                all_episodic_rewards.append(episodic_reward)
                all_episodic_uncertainties.append(np.sum(learner_induced_data[LEARNER_INDUCED_SIGMA_Ts_KEY][str(iterator)]))

            #all_costs.append(np.mean(all_episodic_costs))
            #all_deviations.append(np.mean(all_episodic_uncertainties))
            #all_costs = all_costs + all_episodic_costs
            #all_deviations = all_deviations + all_episodic_uncertainties
            plt.scatter(all_episodic_rewards, all_episodic_uncertainties, color=ALL_CONTEXTS[target_domain][target_task][COLOR], label=task_name)
            plt.ylim(ymin=0.)
        except:
            pass

    #plt.scatter(all_costs, all_deviations)
    plt.xlabel('Episodic Reward', fontweight='bold')
    plt.ylabel('Standard Deviation', fontweight='bold')
    plt.title('Trained on context 6', fontsize=18)
    plt.legend()
    plt.show()