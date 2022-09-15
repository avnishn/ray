from ray.rllib.algorithms.crr import CRRConfig
from ray.rllib.algorithms.crr.torch import CRRTorchPolicy

import gym
import numpy as np
import ray


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ray import train

class WorkerCls:

    def setup(self, rank, world_size):
        obs_space = gym.spaces.Box(np.array((-1, -1)), np.array((1, 1)))
        act_space = gym.spaces.Box(np.array((-1, -1)), np.array((1, 1)))
        config = CRRConfig().to_dict()
        policy = CRRTorchPolicy(obs_space, act_space, config=config)

train_ctx = train.Trainer(backend="torch", num_workers=2, use_gpu=True)
workers = train_ctx.to_worker_group(train_cls=WorkerCls)
# for worker in workers:
workers[0].setup.remote()



