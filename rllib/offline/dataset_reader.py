import logging
import math
import os
import re
import zipfile
from pathlib import Path

import ray.data
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import from_json_data
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType, AlgorithmConfigDict
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_NUM_CPUS_PER_TASK = 0.5


def _get_resource_bundles(config: AlgorithmConfigDict):
    input_config = config.get("input_config", {})
    parallelism = input_config.get("parallelism", config.get("num_workers", 1))
    cpus_per_task = input_config.get(
        "num_cpus_per_read_task", DEFAULT_NUM_CPUS_PER_TASK
    )
    return [{"CPU": math.ceil(parallelism * cpus_per_task)}]


def unzip_if_needed(paths: List[str], format: str):
    """If a path in paths is a zip file, unzip it and use path of the unzipped file"""
    ret = []
    for path in paths:
        if path.startswith("~/"):
            path = os.path.join(os.environ.get("HOME", ""), path[2:])

        # If path doesn't exist, try to interpret is as relative to the
        # rllib directory (located ../../ from this very module).
        path_orig = path
        if not os.path.exists(path):
            path = os.path.join(Path(__file__).parent.parent, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline file {path_orig} not found!")
        if re.search("\\.zip$", path):
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(Path(path).parent)
            path = re.sub("\\.zip$", f".{format}", path)
        ret.append(path)
    return ret


@PublicAPI
def get_dataset_and_shards(
    config: AlgorithmConfigDict, num_workers: int, local_worker: bool
) -> (ray.data.dataset.Dataset, List[ray.data.dataset.Dataset]):
    assert config["input"] == "dataset"
    assert (
        "input_config" in config
    ), "Must specify input_config dict if using Dataset input."

    input_config = config["input_config"]

    format = input_config.get("format")
    assert format in ("json", "parquet"), (
        "Offline input data format must be " "parquet " "or json"
    )
    paths = input_config.get("paths")
    loader_fn = input_config.get("loader_fn")
    if loader_fn and (format or paths):
        raise ValueError(
            "When using a `loader_fn`, you cannot specify a `format` or `path`."
        )

    if not (format and paths) and not loader_fn:
        raise ValueError(
            "Must specify format and path, or a loader_fn via input_config key"
            " when using Ray dataset input."
        )

    if not isinstance(paths, (list, str)):
        raise ValueError("Paths must be a list of path strings or a path string")
    if isinstance(paths, str):
        paths = [paths]
    paths = unzip_if_needed(paths, format)

    parallelism = input_config.get("parallelism", num_workers or 1)
    cpus_per_task = input_config.get(
        "num_cpus_per_read_task", DEFAULT_NUM_CPUS_PER_TASK
    )

    assert loader_fn or (format and paths)

    if loader_fn:
        dataset = loader_fn()
    elif format == "json":
        dataset = ray.data.read_json(
            paths, parallelism=parallelism, ray_remote_args={"num_cpus": cpus_per_task}
        )
    elif format == "parquet":
        dataset = ray.data.read_parquet(
            paths, parallelism=parallelism, ray_remote_args={"num_cpus": cpus_per_task}
        )
    else:
        raise ValueError("Un-supported Ray dataset format: ", format)

    # Local worker will be responsible for sampling.
    if local_worker and num_workers == 0:
        # Dataset is the only shard we need.
        return dataset, [dataset]
    # Remote workers are responsible for sampling:
    else:
        # Each remote worker gets 1 shard.
        # The first None shard is for the local worker, which
        # shouldn't be doing rollout work anyways.
        return dataset, [None] + dataset.repartition(
            num_blocks=num_workers, shuffle=False
        ).split(num_workers)


@PublicAPI
class DatasetReader(InputReader):
    """Reader object that loads data from Ray Dataset.

    Examples:
        config = {
            "input": "dataset",
            "input_config": {
                "format": "json",
                # A single data file, a directory, or anything
                # that ray.data.dataset recognizes.
                "paths": "/tmp/sample_batches/",
                # By default, parallelism=num_workers.
                "parallelism": 3,
                # Dataset allocates 0.5 CPU for each reader by default.
                # Adjust this value based on the size of your offline dataset.
                "num_cpus_per_read_task": 0.5,
            }
        }
    """

    @PublicAPI
    def __init__(self, ioctx: IOContext, ds: ray.data.Dataset):
        """Initializes a DatasetReader instance.

        Args:
            ds: Ray dataset to sample from.
        """
        self._ioctx = ioctx
        self._dataset = ds
        self.count = self._dataset.count()
        self.seed = self._ioctx.worker.policy_config.get("seed")
        if self.seed and not isinstance(self.seed, int):
            raise ValueError(
                "If a random seed is specified, seed can only be an " "integer type."
            )
        self._dataset.random_shuffle(seed=self.seed)
        # We allow the creation of a non-functioning None DatasetReader.
        # It's useful for example for a non-rollout local worker.
        if ds:
            print(
                "DatasetReader ", ioctx.worker_index, " has ", ds.count(), " samples."
            )
            self._iter = self._dataset.repeat().iter_rows()
        else:
            self._iter = None

    @override(InputReader)
    def next(self) -> SampleBatchType:
        # next() should not get called on None DatasetReader.
        assert self._iter is not None

        d = next(self._iter).as_pydict()
        # Columns like obs are compressed when written by DatasetWriter.
        d = from_json_data(d, self._ioctx.worker)

        return d
