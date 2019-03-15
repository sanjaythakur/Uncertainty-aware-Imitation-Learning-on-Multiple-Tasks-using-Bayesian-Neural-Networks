import gym, _pickle as pickle, os, argparse

from multiple_tasks import get_task_on_MUJOCO_environment
import sys
sys.path.insert(0,'./../')
from Housekeeping import *


def log_random_controller_reward(env_name):

    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    file_to_save_logs = LOGS_DIRECTORY + str(env_name) + '_RANDOM.pkl'

    all_contextual_rewards = []
    for context_to_validate in ALL_MUJOCO_TASK_IDENTITIES: 
        env = get_task_on_MUJOCO_environment(env_name=env_name, task_identity=str(context_to_validate))
        cumulative_reward = []
        for episode_iterator in range(NUMBER_VALIDATION_TRIALS):
            observation = env.reset()
            done = False
            reward_at_this_episode = 0.
            while not done:
                observation, reward, done, info = env.step(env.action_space.sample())
                reward_at_this_episode += reward
            cumulative_reward.append(reward_at_this_episode)
        all_contextual_rewards.append(cumulative_reward)

    with open(file_to_save_logs, "wb") as f:
        pickle.dump(all_contextual_rewards, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('A random controller on contextually modified MUJOCO environment'))
    parser.add_argument('env_name', type=str, help='MUJOCO Environment')

    args = parser.parse_args()
    log_random_controller_reward(**vars(args))