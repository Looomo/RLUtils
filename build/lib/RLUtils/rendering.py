import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb
from dm_control.mujoco import engine
from .dataset_utils import determain_env
import warnings
#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#


                
# renderer = MuJoCoRenderer(env)
# if "kitchen" in name:
#     renderer = engine.MovableCamera(env.sim, 1920, 2560)

# render_state_idx(   env_name, dataset, state_id, renderer,  savepath = target_folder, ismaze = bool("maze2d" in name), ismujoco = bool("ant-" in name) , isantmaze = bool("antmaze" in name), env=env   )

                

def render_image(env_name, dataset, idx, env_type, env, renderer = None):

    render_kwargs = {
            'trackbodyid': 2,
            'distance': 1,
            'lookat': [0, -5, 2],
            'elevation': 0
        }
    
    if env_type['frankakitchen']:
        assert 'MUJOCO_GL' in os.environ and os.environ['MUJOCO_GL'] == "egl", "Please set environment variable MUJOCO_GL=egl to render Kitchen."
        renderer = engine.MovableCamera(env.sim, 1920, 2560)
        return render_image_kitchen( dataset, idx, env )
    
    elif env_type['maze2d']:
        renderer = MuJoCoRenderer(env)
        renderer.env.set_target(dataset['infos/goal'][idx])
        renderer.env.set_marker()
        return render_image_default(dataset, idx, renderer)
    elif env_type['gym']:
        renderer = MuJoCoRenderer(env)
        for key, val in render_kwargs.items():
            if key == 'lookat':
                renderer.viewer.cam.lookat[:] = val[:]
            else:
                setattr(renderer.viewer.cam, key, val)
        return render_image_default(dataset, idx, renderer)
    elif env_type['antmaze']:
        renderer = MuJoCoRenderer(env)
        renderer.env.set_target(dataset['infos/goal'][idx])
        return render_image_default(dataset, idx, renderer)
    elif env_type['calvin']:
        # renderer = MuJoCoRenderer(env)
        return render_image_calvin(dataset, idx, renderer = env)
    else:
        assert env_type["__UNKNOWN_ENV__"]
        # assert renderer is not None, f"Env {env_name} is not supported yet, you must porovide a renderer."
        raise NotImplementedError(f"Env {env_name} is not supported yet.")

    

    return False
    # return render_image(dataset, idx, renderer)


def render_image_default( dataset, idx, renderer):
    
    obs = dataset['observations'][idx]
    renderer.env.set_state(dataset['infos/qpos'][idx], dataset['infos/qvel'][idx])
    renderer.viewer.render(512,512)
    data = renderer.viewer.read_pixels(512,512, depth=False)
    image = data[::-1, :, :]
    return image


def render_image_calvin( dataset, idx, renderer):
    
    obs = dataset['observations'][idx]
    renderer.reset_to_state(obs)
    rgb_obs, depth_obs = renderer.get_camera_obs(wh=512)
    # renderer.viewer.render(512,512)
    # data = renderer.viewer.read_pixels(512,512, depth=False)
    # image = rgb_obs[::-1, :, :]
    for k, v in rgb_obs.items(): image = v
    return image


def render_image_kitchen( dataset, idx, env):
    
    obs = dataset['observations'][idx]
    qpos = obs[:30]
    env.robot.reset(env, qpos, np.zeros_like(qpos))
    camera = engine.MovableCamera(env.sim, 1920, 2560)
    camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
    image = camera.render()
    return image


def render_state_idx(env_name, dataset, idx, savepath = "./", env = None, dpi = 200, renderer = None):

    env_type = determain_env(env_name)

    assert env is not None and not env_type["__UNKNOWN_ENV__"], "env must be provided for rendering."
    if not os.path.exists(savepath): os.makedirs(savepath)
    
    image = render_image(env_name, dataset, idx, env_type, env, renderer=renderer)
    plt.figure(); plt.clf(); plt.cla(); plt.imshow(image)
    plt.savefig(os.path.join(savepath,  f"{env_name}-{idx}.png"), dpi = dpi  )



class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env, deviceid = 0):
        self.deviceid = deviceid
        if type(env) is str:
            env = env_map(env)
            env_types = determain_env(env)
            if env_types['calvin']:
                import GymCalvin
                self.env = GymCalvin.make(env).unwrapped
            else:
                self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim, device_id = self.deviceid)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(np.array(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        # save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30, jump=1):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions, jump)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1][:, ::jump]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        # save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = np.array(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        # save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l, jump):
    rollouts = np.stack([
        rollout_from_state(env, state, actions, jump)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions, jump):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        act = act.reshape(jump, -1)
        for j in range(jump):
          obs, rew, term, _ = env.step(act[j])
          observations.append(obs)
          if term:
              break
    for i in range(len(observations), len(actions)*jump+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)