import gym
import time
import ray

class MultiEnvActorSerial:
    def __init__(self, config):
        if "env_cls" not in config:
            raise ValueError("env_cls is required")
        env_cfg = config.get("env_cfg", {})
        num_envs = config.get("num_envs", 1)
        env = config["env_cls"](env_cfg)
        self.actions = [env.action_space.sample() for _ in range(num_envs)]
        self.envs = [config["env_cls"](env_cfg) for _ in range(num_envs)]
        [env.reset() for env in self.envs]

    def step(self):
        start = time.time()
        steps = [env.step(action) for env, action in zip(self.envs, self.actions)]
        # print("avg time per env serial", (time.time() - start) / len(self.envs))
        return steps

@ray.remote
class EnvActor:
    def __init__(self, config):
        if "env_cls" not in config:
            raise ValueError("env_cls is required")
        env_cfg = config.get("env_cfg", {})
        self.env = config["env_cls"](env_cfg)
        self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if done:
            self.env.reset()
        return next_obs, reward, done, info

    def step_multiple_times(self, action, num_times):
        results = []
        for _ in range(num_times):
            results.append(self.step(action))
        return results

    def wait(self):
        return 0


class MultiEnvActorWithActors:
    def __init__(self, config):
        if "env_cls" not in config:
            raise ValueError("env_cls is required")
        env_cfg = config.get("env_cfg", {})
        num_envs = config.get("num_envs", 1)
        env = config["env_cls"](env_cfg)
        self.actions = [env.action_space.sample() for _ in range(num_envs)]
        self.envs = [EnvActor.remote(config) for _ in range(num_envs)]
        ray.get([env.wait.remote() for env in self.envs])

    def step(self):
        start = time.time()
        steps = ray.get([env.step.remote(action) for env, action in zip(self.envs, self.actions)])
        # print("avg time per env ray", (time.time() - start) / len(self.envs))
        return steps


class MultiEnvActorWithMultiProcessing:
    def __init__(self, config):
        if "env_cls" not in config:
            raise ValueError("env_cls is required")
        self.num_envs = config.get("num_envs", 1)
        env_cfg = config.get("env_cfg", {})

        self.envs = (
            gym.vector.AsyncVectorEnv([(lambda: config["env_cls"](env_cfg)) for _ in range(self.num_envs)], shared_memory=True)
        )
        self.envs.reset()
        self.actions = self.envs.action_space.sample()

    def step(self):
        start = time.time()
        steps = self.envs.step(self.actions)
        # print("avg time per env mp", (time.time() - start) / self.num_envs)
        return steps

    def close(self):
        self.envs.close()


ray.init()
num_envs = 10
num_steps = 100
env_name = "BreakoutNoFrameskip-v4"

env_mp = MultiEnvActorWithMultiProcessing({"env_cls": lambda x: gym.make(env_name), "num_envs": num_envs})
env_actor = MultiEnvActorWithActors({"env_cls": lambda x: gym.make(env_name), "num_envs": num_envs})
env_serial = MultiEnvActorSerial({"env_cls": lambda x: gym.make(env_name), "num_envs": num_envs})

start = time.time()
for i in range(num_steps):
    env_mp.step()
diff = time.time() - start
print(f"Env with multiprocessing took {diff} seconds")
env_mp.close()

start = time.time()
for i in range(num_steps):
    env_actor.step()
diff = time.time() - start
print(f"Env with actors took {diff} seconds")
ray.shutdown()

start = time.time()
for i in range(num_steps):
    env_serial.step()
diff = time.time() - start
print(f"Env with serial took {diff} seconds")



