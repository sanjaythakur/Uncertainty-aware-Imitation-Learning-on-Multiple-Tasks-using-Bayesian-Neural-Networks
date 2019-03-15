import gym, numpy as np, argparse


def get_HalfCheetah_task_by_mass(task_identity):
    env = gym.make('HalfCheetah-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[1,0] = 10.
        body_mass[2,0] = 5.
    elif task_identity == '2':
        body_mass[1,0] = 0.5
    elif task_identity == '3':
        body_mass[2,0] = 4.5
    elif task_identity == '4':
        body_mass[2,0] = 3.
    elif task_identity == '5':
        body_mass[1,0] = 0.8
    elif task_identity == '6':
        body_mass[1,0] = 0.8
        body_mass[2,0] = 0.5
    elif task_identity == '7':
        body_mass[1,0] = 0.5
        body_mass[2,0] = 0.5
    elif task_identity == '8':
        body_mass[1,0] = 10.
        body_mass[2,0] = 4.5
    elif task_identity == '9':
        body_mass[1,0] = 10.
        body_mass[2,0] = 6.

    elif task_identity == '10':
        body_mass[1,0] = 7.
        body_mass[5,0] = 0.4

    elif task_identity == '11':
        body_mass[1,0] = 0.7
        body_mass[5,0] = 0.7

    elif task_identity == '12':
        body_mass[2,0] = 1.0
        body_mass[5,0] = 0.2

    elif task_identity == '13':
        body_mass[2,0] = 4.5
        body_mass[5,0] = 0.7

    elif task_identity == '14':
        body_mass[5,0] = 0.3

    elif task_identity == '15':
        body_mass[1,0] = 7.
        body_mass[3,0] = 0.4

    elif task_identity == '16':
        body_mass[1,0] = 0.7
        body_mass[3,0] = 0.7

    elif task_identity == '17':
        body_mass[2,0] = 0.7
        body_mass[3,0] = 0.4

    elif task_identity == '18':
        body_mass[3,0] = 0.4

    elif task_identity == '19':
        body_mass[2,0] = 4.5
        body_mass[3,0] = 0.7

    else:
        print('HalfCheetah-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env



def get_Swimmer_task_by_mass(task_identity):
    env = gym.make('Swimmer-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[0,0] = 10.
    elif task_identity == '2':
        body_mass[1,0] = 10.
    elif task_identity == '3':
        body_mass[2,0] = 10.
    elif task_identity == '4':
        body_mass[3,0] = 10.
    elif task_identity == '5':
        body_mass[2,0] = 28.
    elif task_identity == '6':
        body_mass[2,0] = 32.
    elif task_identity == '7':
        body_mass[3,0] = 32.
    elif task_identity == '8':
        body_mass[2,0] = 25.
    elif task_identity == '9':
        body_mass[1,0] = 30.
    else:
        print('Swimmer-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_Swimmer_task_by_length(task_identity):
    if task_identity == '0':
        # Keeping the default settings
        env = gym.make('Swimmer-v1')
    elif task_identity == '1':
        env = gym.make('Swimmer_1-v1')
    elif task_identity == '2':
        env = gym.make('Swimmer_2-v1')
    elif task_identity == '3':
        env = gym.make('Swimmer_3-v1')
    elif task_identity == '4':
        env = gym.make('Swimmer_4-v1')
    elif task_identity == '5':
        env = gym.make('Swimmer_5-v1')
    elif task_identity == '6':
        env = gym.make('Swimmer_6-v1')
    elif task_identity == '7':
        env = gym.make('Swimmer_7-v1')
    elif task_identity == '8':
        env = gym.make('Swimmer_8-v1')
    elif task_identity == '9':
        env = gym.make('Swimmer_9-v1')
    else:
        print('Swimmer-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    return env



def get_Swimmer_task_by_mass_and_length(task_identity):
    if task_identity == '9':
        env = gym.make('Swimmer_4-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[3,0] = 32.

    elif task_identity == '10':
        env = gym.make('Swimmer_1-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 28.

    elif task_identity == '11':
        env = gym.make('Swimmer_1-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 32.

    elif task_identity == '12':
        env = gym.make('Swimmer_1-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[3,0] = 32.

    elif task_identity == '13':
        env = gym.make('Swimmer_1-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 25.

    elif task_identity == '14':
        env = gym.make('Swimmer_3-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 28.

    elif task_identity == '15':
        env = gym.make('Swimmer_3-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 32.

    elif task_identity == '16':
        env = gym.make('Swimmer_3-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[3,0] = 32.

    elif task_identity == '17':
        env = gym.make('Swimmer_3-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 25.

    elif task_identity == '18':
        env = gym.make('Swimmer_4-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 28.

    elif task_identity == '19':
        env = gym.make('Swimmer_4-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 25.      

    else:
        print('Swimmer-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    return env



def get_Swimmer_task(task_identity):
    if task_identity == '0' or task_identity == '1' or task_identity == '2' or task_identity == '3' or task_identity == '4':
        env = get_Swimmer_task_by_length(task_identity)
    elif task_identity == '5' or task_identity == '6' or task_identity == '7' or task_identity == '8':
        env = get_Swimmer_task_by_mass(task_identity)
    else:
        env = get_Swimmer_task_by_mass_and_length(task_identity)
    return env


def get_task_identityual_Walker2d_by_mass(task_identity):
    env = gym.make('Walker2d-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[1,0] = 35. 
    elif task_identity == '2':
        body_mass[3,0] = 27.
    elif task_identity == '3':
        body_mass[5,0] = 40.
    elif task_identity == '4':
        body_mass[7,0] = 30.
    elif task_identity == '5':
        body_mass[1,0] = 0.35
    elif task_identity == '6':
        body_mass[3,0] = 0.27
    elif task_identity == '7':
        body_mass[5,0] = 0.4
    elif task_identity == '8':
        body_mass[7,0] = 0.3
    elif task_identity == '9':
        body_mass[1,0] = 35.
        body_mass[5,0] = .4
    else:
        print('Walker2d-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_HumanoidStandup_by_mass(task_identity):
    env = gym.make('HumanoidStandup-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[1,0] = 83.
    elif task_identity == '2':
        body_mass[4,0] = 45.
    elif task_identity == '3':
        body_mass[8,0] = 26.
    elif task_identity == '4':
        body_mass[11,0] = 11.
    elif task_identity == '5':
        body_mass[12,0] = 15.
    elif task_identity == '6':
        body_mass[1,0] = 0.83
    elif task_identity == '7':
        body_mass[4,0] = 0.45
    elif task_identity == '8':
        body_mass[8,0] = 0.26
    elif task_identity == '9':
        body_mass[12,0] = 0.15
    else:
        print('HumanoidStandup-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_Humanoid_by_mass(task_identity):
    env = gym.make('Humanoid-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[1,0] = 83.
    elif task_identity == '2':
        body_mass[4,0] = 45.
    elif task_identity == '3':
        body_mass[8,0] = 26.
    elif task_identity == '4':
        body_mass[11,0] = 11.
    elif task_identity == '5':
        body_mass[12,0] = 15.
    elif task_identity == '6':
        body_mass[1,0] = 0.83
    elif task_identity == '7':
        body_mass[4,0] = 0.45
    elif task_identity == '8':
        body_mass[8,0] = 0.26
    elif task_identity == '9':
        body_mass[12,0] = 0.15
    else:
        print('Humanoid-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_Ant_by_mass(task_identity):
    env = gym.make('Ant-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[1,0] = 3.2
    elif task_identity == '2':
        body_mass[4,0] = 0.6
    elif task_identity == '3':
        body_mass[8,0] = 0.3
    elif task_identity == '4':
        body_mass[11,0] = 0.3
    elif task_identity == '5':
        body_mass[12,0] = 0.3
    elif task_identity == '6':
        body_mass[1,0] = 0.03
    elif task_identity == '7':
        body_mass[4,0] = 0.006
    elif task_identity == '8':
        body_mass[8,0] = 0.003
    elif task_identity == '9':
        body_mass[12,0] = 0.003
    else:
        print('Ant-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_Hopper_by_mass(task_identity):
    env = gym.make('Hopper-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[0,0] = 10.
    elif task_identity == '2':
        body_mass[1,0] = 35.
    elif task_identity == '3':
        body_mass[2,0] = 40.
    elif task_identity == '4':
        body_mass[3,0] = 30.
    elif task_identity == '5':
        body_mass[4,0] = 50.
    elif task_identity == '6':
        body_mass[1,0] = 0.35
    elif task_identity == '7':
        body_mass[2,0] = 0.4
    elif task_identity == '8':
        body_mass[3,0] = 0.3
    elif task_identity == '9':
        body_mass[4,0] = 0.5
    else:
        print('Hopper-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_InvertedDoublePendulum_by_mass(task_identity):
    env = gym.make('InvertedDoublePendulum-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[0,0] = 10.
    elif task_identity == '2':
        body_mass[1,0] = 100.
    elif task_identity == '3':
        body_mass[2,0] = 40.
    elif task_identity == '4':
        body_mass[3,0] = 40.
    elif task_identity == '5':
        body_mass[1,0] = 1.
    elif task_identity == '6':
        body_mass[2,0] = .6
    elif task_identity == '7':
        body_mass[3,0] = .6
    elif task_identity == '8':
        body_mass[2,0] = 40.
        body_mass[3,0] = 40.
    elif task_identity == '9':
        body_mass[2,0] = 0.6
        body_mass[3,0] = 40.
    else:
        print('InvertedDoublePendulum-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_InvertedPendulum_by_mass(task_identity):
    env = gym.make('InvertedPendulum-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[0,0] = 10.
    elif task_identity == '2':
        body_mass[1,0] = 100.5
    elif task_identity == '3':
        body_mass[2,0] = 60.
    elif task_identity == '4':
        body_mass[1,0] = .8
    elif task_identity == '5':
        body_mass[2,0] = .3
    elif task_identity == '6':
        body_mass[1,0] = 100.5
        body_mass[2,0] = 60.
    elif task_identity == '7':
        body_mass[1,0] = 100.5
        body_mass[2,0] = .8
    elif task_identity == '8':
        body_mass[1,0] = .8
        body_mass[2,0] = 60.
    elif task_identity == '9':
        body_mass[1,0] = .8
        body_mass[2,0] = .3
    else:
        print('InvertedPendulum-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_identityual_Reacher_by_mass(task_identity):
    env = gym.make('Reacher-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[0,0] = 10.
    elif task_identity == '2':
        body_mass[1,0] = 0.3
    elif task_identity == '3':
        body_mass[2,0] = 0.3
    elif task_identity == '4':
        body_mass[3,0] = 0.04
    elif task_identity == '5':
        body_mass[4,0] = 0.04
    elif task_identity == '6':
        body_mass[1,0] = 0.003
    elif task_identity == '7':
        body_mass[2,0] = 0.003
    elif task_identity == '8':
        body_mass[3,0] = 0.0004
    elif task_identity == '9':
        body_mass[4,0] = 0.0004
    else:
        print('Reacher-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_task_on_MUJOCO_environment(env_name, task_identity):
    if env_name == 'Reacher':
        env = get_task_identityual_Reacher_by_mass(task_identity)
    elif env_name == 'InvertedPendulum':
        env = get_task_identityual_InvertedPendulum_by_mass(task_identity)
    elif env_name == 'InvertedDoublePendulum':
        env = get_task_identityual_InvertedDoublePendulum_by_mass(task_identity)
    elif env_name == 'Swimmer':
        env = get_Swimmer_task(task_identity=task_identity)
    elif env_name == 'HalfCheetah':
        env = get_HalfCheetah_task_by_mass(task_identity=task_identity)
    elif env_name == 'Hopper':
        env = get_task_identityual_Hopper_by_mass(task_identity)
    elif env_name == 'Walker2d':
        env = get_task_identityual_Walker2d_by_mass(task_identity)
    elif env_name == 'Ant':
        env = get_task_identityual_Ant_by_mass(task_identity)
    elif env_name == 'Humanoid':
        env = get_task_identityual_Humanoid_by_mass(task_identity)
    elif env_name == 'HumanoidStandup':
        env = get_task_identityual_HumanoidStandup_by_mass(task_identity)

    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Get modified MUJOCO environment based on your task_identity specified'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-ti', '--task_identity', type=str,
                        help='The underlying and typically unobservable task_identity during operation of a controller',
                        default='0')
    args = parser.parse_args()
    
    env = get_task_identityual_MUJOCO_environment(**vars(args))
    print(env.env.model.body_mass)


#env = gym.make('InvertedPendulum-v1')
#env = gym.make('InvertedDoublePendulum-v1')
#env = gym.make('Reacher-v1')
#env = gym.make('HalfCheetah-v1')
#env = gym.make('Swimmer-v1')
#env = gym.make('Hopper-v1')
#env = gym.make('Walker2d-v1')
#env = gym.make('Ant-v1')
#env = gym.make('Humanoid-v1')
#env = gym.make('HumanoidStandup-v1')

#observation = env.reset()
#print(observation)

#print(env.observation_space)
#print(env.action_space)

#mb = env.env.model.body_mass
#print(mb)


'''

mb = np.array(mb)
mb[3,0] = 70000000.7777
env.model.body_mass = mb
print(env.model.body_mass)

for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            break


mb = env.model.body_mass
print(mb)


mb = np.array(mb)
mb[1,0] = 7.7777
env.model.body_mass = mb
print(env.model.body_mass)
'''
