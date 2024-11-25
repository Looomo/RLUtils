import gym
import d4rl
import numpy as np
import collections
import copy
import d4rl_ext
from tqdm import tqdm
from collections import defaultdict
from .common import rebuild_terminals_by_thresh, check_undetected_terminal, check_invalid_terminal
import GymCalvin
def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    env_types = determain_env(name)
    if env_types['calvin']:
        return GymCalvin.make(name)
    # with suppress_output():
    #     wrapped_env = gym.make(name)
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    # if "antmaze" in name: 
    #     env.max_episode_steps = env._max_episode_steps
    # else:
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env
def get_dataset(env):
    dataset = env.get_dataset()
    return dataset


# 返回值： 1. 所有的切分的trajectory  2. 原始的dataset
# 注意： 没有做padding
# 需要检查maze2d

def get_antmaze_dataset(env):

    # dones_float = np.zeros_like(dataset['rewards'])
    dataset = d4rl.qlearning_dataset(env)
    dataset_ = copy.deepcopy(dataset)
    dataset_['terminals'][:] = 0.
    dataset_['timeouts'] = np.zeros_like(  dataset_['terminals']  )
    for i in tqdm(range(len(dataset_['terminals']) - 1), desc = "Loading Antmaze. Rebuilding terminal signal"):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
            dataset_['terminals'][i] = 1
        else:
            dataset_['terminals'][i] = 0
    dataset_['terminals'][-1] = 1
    return dataset_

def terminal_on_timeout(env_name):

    datasets_terminal_on_timeout = [

    ]

    # except kitchen
    # except antmaze

    for dataset in datasets_terminal_on_timeout:
        if dataset in env_name: return True
    return False


# note: i do not build next observ
def sequence_to_plain(all_data: list):
    dataset_ = defaultdict(list)


    for traj in all_data:
        for k,v in traj.items():
            if k in ['start', 'end', 'accumulated_reward']: continue
            dataset_[k].append(v)
    
    for k,v in dataset_.items(): 
        
        try:
            dataset_[k] = np.concatenate(v)
        except Exception as e:
            dataset_[k] = np.array(v)
    dataset = dataset_

    return dataset



def sequence_dataset(env,  rewrite_last_terminal = True, build_next_obs = True, check_terminals = True, dataset = None,  *args, **kwargs):
    
    # dataset = preprocess_fn(dataset)
    env_type = determain_env(env.spec.id)
    if dataset is None:
        dataset = get_dataset(env)

    
    
    if env_type['antmaze']: dataset = get_antmaze_dataset(env)
    if env_type['frankakitchen']: dataset = rebuild_terminals_by_thresh(dataset)



    if rewrite_last_terminal: dataset['terminals'][-1] = 1.

    # if check_terminals:
    #     check_undetected_terminal(dataset)
    #     check_invalid_terminal(dataset)

    


    rebuild_next_observ_required = env_type['maze2d'] or env_type['frankakitchen'] # 'kitchen' in env.spec.id


    # if "kitchen" in env.spec.id: dataset = build_next_observ(dataset)

    # if "kitchen" in env.spec.id:
    #     dataset['observations'] += (np.random.rand(dataset['observations'].shape[0]*dataset['observations'].shape[1]).reshape(dataset['observations'].shape)-0.5)/1e14
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
            final_timestep = (episode_step == env.max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or (final_timestep and terminal_on_timeout(env)):
            # if done_bool and final_timestep: print("whatthehell")
            # if done_bool:
            #     print("what the hell")
            # if final_timestep:
            #     print("hello")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if rebuild_next_observ_required and build_next_obs:
                episode_data = rebuild_next_observ(episode_data)
                if "kitchen" in env.name and done_bool: episode_data['terminals'][-1] = True
                # episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end + 1
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1
    
    if rebuild_next_observ_required and build_next_obs:
        dataset_ = defaultdict(list)


        for traj in all_data:
            for k,v in traj.items():
                if k in ['start', 'end', 'accumulated_reward']: continue
                dataset_[k].append(v)
        
        for k,v in dataset_.items(): 
            
            try:
                dataset_[k] = np.concatenate(v)
            except Exception as e:
                dataset_[k] = np.array(v)
        dataset = dataset_
    return all_data, dataset



def rebuild_next_observ(episode):
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



# def process_maze2d_episode(episode):
#     '''
#         adds in `next_observations` field to episode
#     '''
#     assert 'next_observations' not in episode
#     length = len(episode['observations'])
#     next_observations = episode['observations'][1:].copy()
#     for key, val in episode.items():
#         episode[key] = val[:-1]
#     episode['next_observations'] = next_observations
#     return episode


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
    "calvin": {
        'tasks': ['calvin'],
        'datasets':[''],
        "versions":[''],
        "sparse": "All zero. No reward at all."
    },
}


def determain_env(env_full_name):
    # base_task = env_full_name.split("-")[0]
    # what_env_is_it = {
    #     'maze2d': 'maze2d' in env_full_name,
    #     'antmaze': 'antmaze-' in env_full_name,
    #     'adroit': 'pen-' in env_full_name or 'hammer-' in env_full_name or 'door-' in env_full_name or 'relocate-' in env_full_name,
    #     'gym': base_task in meta_infos['gym'],
    #     'kitchen': 'kitchen' in env_full_name
    # }
    what_env_is_it = {
        "__UNKNOWN_ENV__": True
    }
    for env_type in meta_infos.keys(): 
        is_env_type = determain_env_type_match( env_full_name,env_type  )
        what_env_is_it[env_type] = is_env_type
        if is_env_type: what_env_is_it["__UNKNOWN_ENV__"] = False
    return what_env_is_it


def determain_env_type_match(env_full_name, env_type: str):
    # if env_type not in meta_infos.keys():
    #     print
    assert env_type in meta_infos.keys(), f"env_type must be selected from: { meta_infos.keys()}"
    base_task = env_full_name.split("-")[0]
    return base_task in meta_infos[env_type]['tasks']


def predownload(env_name):
    env = load_environment(env_name)
    trajs, _ = sequence_dataset(env)
    return