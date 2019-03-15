import tensorflow as tf
import argparse, gym, sys, os
import _pickle as pickle
import numpy as np

sys.path.insert(0, './../../')
from Housekeeping import *

sys.path.insert(0, './../')
from multiple_tasks import get_task_on_MUJOCO_environment

from Plots import plotDemonstrators

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)


def getScaleAndOffset(env_name, demonstrator_task_identity):
	FILE_NAME = './../' + DEMONSTRATOR_TRAJECTORIES_DIRECTORY + env_name + '_' + demonstrator_task_identity + '.pkl'
	with open(FILE_NAME, 'rb') as f:
		all_stored_data = pickle.load(f)
	scale = all_stored_data[SCALE_KEY]
	offset = all_stored_data[OFFSET_KEY]
	return scale, offset


def load_trained_model(env_name): 

	reward_on_itself = []

	for demonstrator_task_identity in ALL_MUJOCO_TASK_IDENTITIES:
		demonstrator_task_identity = str(demonstrator_task_identity)
		scale, offset = getScaleAndOffset(env_name, demonstrator_task_identity)

		saved_final_model = './../' + SAVED_DEMONSTRATOR_MODELS_DIRECTORY + env_name + '/' + demonstrator_task_identity + '/'
		tf.reset_default_graph()
		imported_meta = tf.train.import_meta_graph(saved_final_model + 'final.meta')

		with tf.Session() as sess:  
			imported_meta.restore(sess, tf.train.latest_checkpoint(saved_final_model))
			graph = tf.get_default_graph()

			scaled_observation_node = graph.get_tensor_by_name('obs:0')
			output_action_node = graph.get_tensor_by_name('output_action:0')

			demonstrator_rewards_over_all_contexts = []
			for validation_task_identity in ALL_MUJOCO_TASK_IDENTITIES:
				validation_task_identity = str(validation_task_identity)
				total_reward_over_all_episodes = 0.
				env = get_task_on_MUJOCO_environment(env_name=env_name, task_identity=validation_task_identity)
				#env = wrappers.Monitor(env, "./recordings/" + env_name, force=True)
				observation = env.reset()
				for episode_iterator in range(NUMBER_VALIDATION_TRIALS):
					#print(GREEN('Episode number to validate ' + str(episode_iterator)))
					total_reward = 0.
					finish = False
					time_step = 0.

					while not finish:
						#env.render()
						observation = observation.astype(np.float32).reshape((1, -1))
						observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
						output_action = sess.run(output_action_node, feed_dict={scaled_observation_node:(observation - offset) * scale})
						observation, reward, finish, info = env.step(output_action)
						total_reward += reward
						time_step += 1e-3
						if finish:
							observation = env.reset()
					total_reward_over_all_episodes += total_reward
					#print(RED('Reward obtained during this episode is ' + str(total_reward)))

				#print(RED('Average demonstrator reward is ' + str(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)))
				if demonstrator_task_identity == validation_task_identity:
					reward_on_itself.append(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)
				demonstrator_rewards_over_all_contexts.append(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)
				env.close()

			plotDemonstrators(env_name=env_name, demonstrator_rewards_over_all_contexts=demonstrator_rewards_over_all_contexts, identifier=str(demonstrator_task_identity))

	file_name_to_log_data = './../' + DEMONSTRATOR_CONTROLLER_REWARD_LOG_DIRECTORY + env_name + '.pkl'
	with open(file_name_to_log_data, "wb") as f:
		pickle.dump(reward_on_itself, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=('Reload and reuse policy trained using PPO '
												  ' on OpenAI Gym environment'))
	parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')

	args = parser.parse_args()
	load_trained_model(**vars(args))
