# Uncertainty Aware Imitation Learning in Multiple Contexts using Bayesian Neural Networks
[Link to paper](https://arxiv.org/pdf/1903.05697.pdf)

## MuJoCo Experiments

##### Set-up
1. The experiments have been performed on the following packages:
	- openai gym, version = '0.9.3'
	- MuJoCo mjpro131

2. Some experiments need changes to be made to the openai gym source code. Perform one of the following operations to have things set up for yourself.
	- Copy, paste (, and replace if asked) each and every file inside `MuJoCo/gym_files_to_be_merged` to their equivalent locations of your gym installation.
	- Use the OpenAI gym source code from [here](https://github.com/sanjaythakur/Multiple_Task_MuJoCo_Domains).
	Note that the new files needed for the experiment does not break any other part of the original code.

##### Setting up the demonstrator controllers
1. There are 20 tasks defined both for Swimmer and HalfCheetah domains. In order to generate demonstrators on these tasks use the script `train.py` under `MuJoCo/PPO` or just run `./overnight_1.sh` for generating HalfCheetah demonstrators and `./overnight_2.sh` for generating Swimmer demonstrators. The generated demonstrator controllers are stored under `MuJoCo/saved_demonstrator_models` directory. Note that at the end of training a new demonstrator controller, a certain number of demonstration episodes are run and stored for later use under the directory `MuJoCo/demonstrator_trajectories`. The number of demonstration episodes to record can be changed by changing the value of variable `DEMONSTRATOR_EPISODES_TO_LOG` in the file `Housekeeping.py`.
Note that the code for PPO has been taken from [here](https://github.com/sanjaythakur/trpo).

2. The quality of demonstrators can be checked by running the script `test_demonstrator_models.py` under `MuJoCo/PPO`. This script will generate plots showing performance of controllers under the directory `MuJoCo/demonstrator_controller_reward_log`. An example usage is ```python test_demonstrator_models.py HalfCheetah```.

##### Reproducing the results from the paper
1. To reproduce the top subfigure in figure 2, run the following command from the directory `sliding_block/`.
	- ```python Generate_BBB_Controllers.py -c 3 -ws 2 -po True```
	- ```python Validate_GP_Controller.py -c 3 -ws 2 -po True```
	- ```python Validate_BBB_Controller.py -c 3 -ws 2 -po True```
	After these commands finish execution, open `sliding_block/plots.ipynb`
	- ```all_task_configurations = [
		    {BEHAVIORAL_CONTROLLER_KEY: 'BBB', CONTEXT_CODE_KEY: '3', WINDOW_SIZE_KEY: '2', PARTIAL_OBSERVABILITY_KEY: 'True'},
		    {BEHAVIORAL_CONTROLLER_KEY: 'GP', CONTEXT_CODE_KEY: '3', WINDOW_SIZE_KEY: '2', PARTIAL_OBSERVABILITY_KEY: 'True'},
		    {BEHAVIORAL_CONTROLLER_KEY: 'BBB', CONTEXT_CODE_KEY: '3', WINDOW_SIZE_KEY: '5', PARTIAL_OBSERVABILITY_KEY: 'True'},
		    {BEHAVIORAL_CONTROLLER_KEY: 'GP', CONTEXT_CODE_KEY: '3', WINDOW_SIZE_KEY: '5', PARTIAL_OBSERVABILITY_KEY: 'True'},
		]

		plt.rcParams["grid.alpha"] = 1
		plt.rcParams["grid.color"] = "#cccccc"

		final_flourish(all_task_configurations, uncertainty_ylim=250., cost_ylim=1e6, predictive_error_ylim=1e6)```
2. To reproduce the bottom subfigure in figure 2, run the following command from the directory `MuJoCo/`.
	- ```python Generate_BBB_Controller.py -d Swimmer -t 4 -ws 1 -nd 1```
	- ```python Generate_GP_Controller.py -d Swimmer -t 4 -ws 1 -nd 1```
	- ```python Validate_BBB_Controller.py -d Swimmer -t 4 -ws 1 -nd 1```
	After these commands finish execution, open `MuJoCo/Plots.ipynb`
	- ```BBBvsGP_generalization(domain_name='Swimmer', number_demonstrations='1', window_size='1', behavioral_controller='4')```

3. To reproduce the results shown in figure 4, run the following commands from directory `MuJoCo/`.
	-  ```python proposed_mechanism.py -et active_learning_proof_of_concept -c BBB -dn HalfCheetah -dc 7.0 -dm 200```
	-  ```python proposed_mechanism.py -et active_learning_proof_of_concept -c BBB -dn Swimmer -dc 20.0 -dm 200```
	-  ```python proposed_mechanism.py -et active_learning_proof_of_concept -c NAIVE -dn HalfCheetah```
	-  ```python proposed_mechanism.py -et active_learning_proof_of_concept -c NAIVE -dn Swimmer```
	After these commands finish execution, open `MuJoCo/Plots.ipynb`
	- ```all_configurations = [
		    {'controller': 'BBB' , 'detector_c': 7.0, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'NAIVE', 'number_demonstrations': 10},
		]

		compare_data_efficiency(domain_name='HalfCheetah',
		                        all_configurations=all_configurations,
		                        simulation_iterators=[0],
		                        demonstration_request_to_gauge=2)```
    - ```all_configurations = [
		    {'controller': 'BBB' , 'detector_c': 20.0, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'NAIVE', 'number_demonstrations': 10},
		]

		compare_data_efficiency(domain_name='Swimmer',
		                        all_configurations=all_configurations,
		                        simulation_iterators=[0],
		                        demonstration_request_to_gauge=2)```

4. To reproduce the results shown in figure 4, run the following commands from directory `MuJoCo/`.
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn HalfCheetah -dc 1.7 -dm 200```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn HalfCheetah -dc 1.7 -dm 50```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn HalfCheetah -dc 1.5 -dm 100```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn HalfCheetah -dc 1.2 -dm 200```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn HalfCheetah -dc 1.2 -dm 50```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn Swimmer -dc 1.7 -dm 200```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn Swimmer -dc 1.7 -dm 50```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn Swimmer -dc 1.5 -dm 100```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn Swimmer -dc 1.2 -dm 200```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c BBB -dn Swimmer -dc 1.2 -dm 50```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c NAIVE -dn HalfCheetah```
	-  ```python proposed_mechanism.py -et data_efficient_active_learning -c NAIVE -dn Swimmer```
	-  ```python random_controller.py HalfCheetah```
	-  ```python random_controller.py Swimmer```
	After these commands finish execution, open `MuJoCo/Plots.ipynb`
	- ```all_configurations = [
		    {'controller': 'BBB' , 'detector_c': 1.7, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.7, 'detector_m': 50, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.5, 'detector_m': 100, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.2, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.2, 'detector_m': 50, 'number_demonstrations': 10},
		    {'controller': 'NAIVE', 'number_demonstrations': 10},
		    {'controller': 'RANDOM'},
		]

		compare_conservativeness_with_scatter_plot(domain_name='HalfCheetah',
		                         all_configurations=all_configurations, ymax=49000.,
		                         simulation_iterators=[0,1,2,3,4])```
	- ```all_configurations = [
		    {'controller': 'BBB' , 'detector_c': 1.7, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.7, 'detector_m': 50, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.5, 'detector_m': 100, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.2, 'detector_m': 200, 'number_demonstrations': 10},
		    {'controller': 'BBB' , 'detector_c': 1.2, 'detector_m': 50, 'number_demonstrations': 10},
		    {'controller': 'NAIVE', 'number_demonstrations': 10},
		    {'controller': 'RANDOM'},
		]

		compare_conservativeness_with_scatter_plot(domain_name='Swimmer',
		                         all_configurations=all_configurations, ymax=3500.,
		                         simulation_iterators=[0,1,2,3,4])```                  
