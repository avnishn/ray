import gym
from ray.rllib.examples.env import random_env
from ray.rllib.examples.env.multi_agent import make_multi_agent
import time
import ray
from ray import tune
from ray.rllib.policy.policy import PolicySpec


class LargeRandomEnv(random_env.RandomEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1000,))
        self.action_space = gym.spaces.Discrete(10)
        self.p_done = 0.0
        self.max_episode_len = 10000

    def step(self, action):
        ret = super().step(action)
        time.sleep(0.1)
        return ret


LargeMAEnv = make_multi_agent(LargeRandomEnv)

if __name__ == "__main__":
    ray.init(address="auto")
    def gen_policy(i):
        config = {
            "model": {
                "fcnet_hiddens": [128, 128],
            },
        }
        return PolicySpec(config=config)
    policies = {0: gen_policy(0)}

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return 0

    config = {
        "env": LargeMAEnv,
        "env_config": {
            "num_agents": 2,
        },
        "multiagent": {"policies": policies, "policy_mapping_fn": policy_mapping_fn, },
        "framework": "tf2",
        "eager_tracing": True,
        "num_workers": 42,
        "num_envs_per_worker": 4,
        "disable_env_checking": True,
        "num_gpus": 1,
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 16,
        "train_batch_size": 2048,
        "gamma": 1.0,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 3e-3,
        "grad_clip": 40.0,
        "lr": 4e-4,
        "_tf_policy_handles_more_than_one_loss": True,
        "_separate_vf_optimizer": True,
        "_lr_vf": 2e-4,
    }


    stop = {"timesteps_total": 50000000}
    results = tune.run("APPO", stop=stop, config=config, verbose=3)
    ray.shutdown()
