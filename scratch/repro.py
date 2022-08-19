import ray
import ray.data
from ray.rllib.offline.dataset_reader import _unzip_if_needed


runtime_env={"working_dir": "/home/ray/workspace-project-avnish-dev/scratch"}

ray.init(runtime_env=runtime_env)

paths=["half_cheetah_data"]
dataset = ray.data.read_json(
        paths, parallelism=1, ray_remote_args={"num_cpus": 0.5}
    )

_unzip_if_needed(["half_cheetah_data"], format="json")

