import numpy as np
from procgen import ProcgenEnv
import copy
from tqdm import tqdm
class SpecID:
    def __init__(self, id = None) -> None:
        self.id = id
class ProcgenWrappedEnv:
    def __init__(self, name, num_envs, env_name, start_level, num_levels, distribution_mode='easy'):
        self.envs = ProcgenEnv(num_envs=num_envs,
                               env_name=env_name,
                               distribution_mode=distribution_mode,
                               start_level=start_level,
                               num_levels=num_levels)
        self.r = np.zeros(num_envs)
        self.t = np.zeros(num_envs)

        self.spec = SpecID(id = name)
        self.load_procgen_dataset()
        self.load_sequence_dataset()

    def load_sequence_dataset(self, ):
        buffer = self.buffer
        nums_ = len(buffer['action'])

        idx_traj_ended_list_ = np.where(self.buffer['terminals'])[0].tolist()
        idx_traj_ended = np.unique(sorted(idx_traj_ended_list_)) + 1
        idx_traj_ended_list = idx_traj_ended.tolist()

        # idx_traj_started_ = idx_traj_ended - 1
        idx_traj_started_list = [0] + idx_traj_ended_list[:-1]
        idx_traj_started = np.array(idx_traj_started_list)
        # indexs = np.arange(nums_)

        trajs_of_idxs = [
            np.arange( idx_traj_started[i],  idx_traj_ended[i] ) for i in range(len(idx_traj_started))
        ]

        # buffer['slices']

        self.trajs_of_idxs = trajs_of_idxs

        # self.sequence_dataset = copy.deepcopy(self.buffer)
        # self.sequence_dataset['trajs_of_idxs'] = trajs_of_idxs
        trajs = []
        for idx_sequence in tqdm(trajs_of_idxs, desc=f"Building sequence of {self.spec.id}"):
            traj_info = {}
            for k in self.buffer.keys():
                traj_info[k] = copy.deepcopy( self.buffer[k][idx_sequence] )
            traj_info['start'] = idx_sequence[0]
            traj_info['end'] = idx_sequence[-1] + 1
            traj_info['idxs'] = idx_sequence
            trajs.append( traj_info )
        self.trajs = trajs
        return

    def load_procgen_dataset(self,):
        level = self.spec.id.split("-")[-1].split(".")[0]
        buffer_fname = f"data/procgen/level{level}.npz"
        print(f'get_procgen_dataset Attempting to load buffer: {buffer_fname}')
        buffer = np.load(buffer_fname)
        buffer = {k: buffer[k] for k in buffer}

        nums_ = len(buffer['action'])
        idx_level_reset = np.where( buffer['level_ended'] )[0].tolist()
        idx_goal_reached = np.where( buffer['goal_ended'] )[0].tolist()
        idx_traj_ended_list_ = idx_level_reset + idx_goal_reached
        buffer['terminals'] = np.zeros(nums_)
        buffer['terminals'][idx_traj_ended_list_] = 1.0
        buffer['terminals']  = buffer['terminals'].astype(bool)

        buffer['observations'] = buffer['observation']
        buffer['next_observations'] = buffer['next_observation']
        buffer['actions'] = buffer['action']

        self.buffer = buffer

        return
    def get_procgen_dataset(self,):
        
        return self.buffer
    def get_sequence_dataset(self,):

        # raw = self.get_procgen_dataset()
        # trajs = self.sequence_dataset_procgen_maze()
        
        # raw['trajs'] = trajs
        
        return self.trajs, self.buffer
    
    def get_dataset(self,):

        return self.buffer
    def obs(self, x):
        return x['rgb']

    def reset(self):
        self.r = np.zeros_like(self.r)
        self.t = np.zeros_like(self.r)
        return self.obs(self.envs.reset())

    def step(self, a):
        obs, r, done, info = self.envs.step(a)
        self.r += r
        self.t += 1
        for n in range(len(done)):
            if done[n]:
                info[n]['episode'] = dict(r=self.r[n], t=self.t[n], time_r=max(-500, -1 * self.t[n]))
                self.r[n] = 0
                self.t[n] = 0
        return self.obs(obs), r, done, info