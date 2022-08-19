import ray
from ray.rllib.offline.dataset_reader import _unzip_if_needed

runtime_env = {"working_dir": "/home/ray/workspace-project-avnish-dev/scratch"}
ray.init(runtime_env=runtime_env)

_unzip_if_needed(["half_cheetah_data"], format="json")


