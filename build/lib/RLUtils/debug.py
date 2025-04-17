

if __name__ == "__main__":
    from dataset_utils import *
    import d4rl_ext
    import torch
    import numpy as np
    import gym
    import d4rl
    # from branch_control import FallbackController
    env_name = "walker2d-medium-replay-v2"
    env = load_environment(env_name)
    itr, dataset = sequence_dataset(env)
    # fallback_controller = FallbackController()

    print("Done")