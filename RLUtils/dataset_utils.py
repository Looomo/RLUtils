import gym
import d4rl
import numpy as np
import collections


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    # with suppress_output():
    #     wrapped_env = gym.make(name)
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env
def get_dataset(env):
    dataset = env.get_dataset()
    return dataset


# 返回值： 1. 所有的切分的trajectory  2. 原始的dataset
# 注意： 没有做padding
# 需要检查maze2d
def sequence_dataset(env):
    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    all_data = []
    episode_step = 0
    start = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            # if done_bool:
            #     print("what the hell")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset



def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode


meta_infos = {
    "maze2d": {
        'tasks': ['maze2d'],
        'datasets':["umaze", "medium", "large"], # "umaze-dense", "medium-dense", "large-dense"],
        "versions":["v1"],
        "sparse": "Sparse except dense",
        "observshape": ""
    },
    "antmaze": {
        'tasks':['antmaze'],
        'datasets':["umaze", "umaze-diverse", "medium-diverse","medium-play", "large-diverse", "large-play"],
        "versions":["v0"],
        "sparse": "Sparse"
    },
    "adroit": {
        'tasks': ['pen', 'hammer', 'door', 'relocate'],
        'datasets':['human', 'cloned', 'expert'],
        "versions":["v1"],
        "sparse": "Dense"
    },
    "gym": {
        'tasks': ['halfcheetah', 'walker2d', 'hopper', 'ant'],
        'datasets':['random', 'medium', 'expert', 'medium-expert', 'medium-replay'],
        "versions":["v2"],
        "sparse": "Dense"
    },
    "frankakitchen": {
        'tasks': ['kitchen'],
        'datasets':['complete', 'partial', 'mixed'],
        "versions":["v0"],
        "sparse": "Dense"
    },
}


def determain_env(env_full_name):

    what_env_is_it = {
        'maze2d': 'maze2d' in env_full_name,
        'antmaze': 'antmaze-' in env_full_name,
        'adroit': 'pen-' in env_full_name or 'hammer-' in env_full_name or 'door-' in env_full_name or 'relocate-' in env_full_name,
        'gym': '-v2' in env_full_name,
        'kitchen': 'kitchen' in env_full_name
    }

    return what_env_is_it

def predownload(env_name):
    env = load_environment(env_name)
    trajs, _ = sequence_dataset_mix(env)

    return