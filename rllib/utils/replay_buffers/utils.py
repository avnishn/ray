import logging
import psutil
from typing import Optional, Any

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.replay_buffers import (
    MultiAgentPrioritizedReplayBuffer,
    ReplayBuffer,
    MultiAgentReplayBuffer,
)
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import ResultDict, SampleBatchType, AlgorithmConfigDict
from ray.util import log_once

logger = logging.getLogger(__name__)


@DeveloperAPI
def update_priorities_in_replay_buffer(
    replay_buffer: ReplayBuffer,
    config: AlgorithmConfigDict,
    train_batch: SampleBatchType,
    train_results: ResultDict,
) -> None:
    """Updates the priorities in a prioritized replay buffer, given training results.

    The `abs(TD-error)` from the loss (inside `train_results`) is used as new
    priorities for the row-indices that were sampled for the train batch.

    Don't do anything if the given buffer does not support prioritized replay.

    Args:
        replay_buffer: The replay buffer, whose priority values to update. This may also
            be a buffer that does not support priorities.
        config: The Algorithm's config dict.
        train_batch: The batch used for the training update.
        train_results: A train results dict, generated by e.g. the `train_one_step()`
            utility.
    """
    # Only update priorities if buffer supports them.
    if isinstance(replay_buffer, MultiAgentPrioritizedReplayBuffer):
        # Go through training results for the different policies (maybe multi-agent).
        prio_dict = {}
        for policy_id, info in train_results.items():
            # TODO(sven): This is currently structured differently for
            #  torch/tf. Clean up these results/info dicts across
            #  policies (note: fixing this in torch_policy.py will
            #  break e.g. DDPPO!).
            td_error = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
            # Set the get_interceptor to None in order to be able to access the numpy
            # arrays directly (instead of e.g. a torch array).
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            # Get the replay buffer row indices that make up the `train_batch`.
            batch_indices = train_batch.policy_batches[policy_id].get("batch_indexes")

            if td_error is None:
                if log_once(
                    "no_td_error_in_train_results_from_policy_{}".format(policy_id)
                ):
                    logger.warning(
                        "Trying to update priorities for policy with id `{}` in "
                        "prioritized replay buffer without providing td_errors in "
                        "train_results. Priority update for this policy is being "
                        "skipped.".format(policy_id)
                    )
                continue

            if batch_indices is None:
                if log_once(
                    "no_batch_indices_in_train_result_for_policy_{}".format(policy_id)
                ):
                    logger.warning(
                        "Trying to update priorities for policy with id `{}` in "
                        "prioritized replay buffer without providing batch_indices in "
                        "train_batch. Priority update for this policy is being "
                        "skipped.".format(policy_id)
                    )
                continue

            #  Try to transform batch_indices to td_error dimensions
            if len(batch_indices) != len(td_error):
                T = replay_buffer.replay_sequence_length
                assert (
                    len(batch_indices) > len(td_error) and len(batch_indices) % T == 0
                )
                batch_indices = batch_indices.reshape([-1, T])[:, 0]
                assert len(batch_indices) == len(td_error)
            prio_dict[policy_id] = (batch_indices, td_error)

        # Make the actual buffer API call to update the priority weights on all
        # policies.
        replay_buffer.update_priorities(prio_dict)


@DeveloperAPI
def sample_min_n_steps_from_buffer(
    replay_buffer: ReplayBuffer, min_steps: int, count_by_agent_steps: bool
) -> Optional[SampleBatchType]:
    """Samples a minimum of n timesteps from a given replay buffer.

    This utility method is primarily used by the QMIX algorithm and helps with
    sampling a given number of time steps which has stored samples in units
    of sequences or complete episodes. Samples n batches from replay buffer
    until the total number of timesteps reaches `train_batch_size`.

    Args:
        replay_buffer: The replay buffer to sample from
        num_timesteps: The number of timesteps to sample
        count_by_agent_steps: Whether to count agent steps or env steps

    Returns:
        A concatenated SampleBatch or MultiAgentBatch with samples from the
        buffer.
    """
    train_batch_size = 0
    train_batches = []
    while train_batch_size < min_steps:
        batch = replay_buffer.sample(num_items=1)
        batch_len = batch.agent_steps() if count_by_agent_steps else batch.env_steps()
        if batch_len == 0:
            # Replay has not started, so we can't accumulate timesteps here
            return batch
        train_batches.append(batch)
        train_batch_size += batch_len
    # All batch types are the same type, hence we can use any concat_samples()
    train_batch = SampleBatch.concat_samples(train_batches)
    return train_batch


@DeveloperAPI
def validate_buffer_config(config: dict) -> None:
    """Checks and fixes values in the replay buffer config.

    Checks the replay buffer config for common misconfigurations, warns or raises
    error in case validation fails. The type "key" is changed into the inferred
    replay buffer class.

    Args:
        config: The replay buffer config to be validated.

    Raises:
        ValueError: When detecting severe misconfiguration.
    """
    if config.get("replay_buffer_config", None) is None:
        config["replay_buffer_config"] = {}

    if config.get("worker_side_prioritization", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning(
            old="config['worker_side_prioritization']",
            new="config['replay_buffer_config']['worker_side_prioritization']",
            error=True,
        )

    prioritized_replay = config.get("prioritized_replay", DEPRECATED_VALUE)
    if prioritized_replay != DEPRECATED_VALUE:
        deprecation_warning(
            old="config['prioritized_replay'] or config['replay_buffer_config']["
            "'prioritized_replay']",
            help="Replay prioritization specified by config key. RLlib's new replay "
            "buffer API requires setting `config["
            "'replay_buffer_config']['type']`, e.g. `config["
            "'replay_buffer_config']['type'] = "
            "'MultiAgentPrioritizedReplayBuffer'` to change the default "
            "behaviour.",
            error=True,
        )

    capacity = config.get("buffer_size", DEPRECATED_VALUE)
    if capacity == DEPRECATED_VALUE:
        capacity = config["replay_buffer_config"].get("buffer_size", DEPRECATED_VALUE)
    if capacity != DEPRECATED_VALUE:
        deprecation_warning(
            old="config['buffer_size'] or config['replay_buffer_config']["
            "'buffer_size']",
            new="config['replay_buffer_config']['capacity']",
            error=True,
        )

    replay_burn_in = config.get("burn_in", DEPRECATED_VALUE)
    if replay_burn_in != DEPRECATED_VALUE:
        config["replay_buffer_config"]["replay_burn_in"] = replay_burn_in
        deprecation_warning(
            old="config['burn_in']",
            help="config['replay_buffer_config']['replay_burn_in']",
        )

    replay_batch_size = config.get("replay_batch_size", DEPRECATED_VALUE)
    if replay_batch_size == DEPRECATED_VALUE:
        replay_batch_size = config["replay_buffer_config"].get(
            "replay_batch_size", DEPRECATED_VALUE
        )
    if replay_batch_size != DEPRECATED_VALUE:
        deprecation_warning(
            old="config['replay_batch_size'] or config['replay_buffer_config']["
            "'replay_batch_size']",
            help="Specification of replay_batch_size is not supported anymore but is "
            "derived from `train_batch_size`. Specify the number of "
            "items you want to replay upon calling the sample() method of replay "
            "buffers if this does not work for you.",
            error=True,
        )

    # Deprecation of old-style replay buffer args
    # Warnings before checking of we need local buffer so that algorithms
    # Without local buffer also get warned
    keys_with_deprecated_positions = [
        "prioritized_replay_alpha",
        "prioritized_replay_beta",
        "prioritized_replay_eps",
        "no_local_replay_buffer",
        "replay_zero_init_states",
        "learning_starts",
        "replay_buffer_shards_colocated_with_driver",
    ]
    for k in keys_with_deprecated_positions:
        if config.get(k, DEPRECATED_VALUE) != DEPRECATED_VALUE:
            deprecation_warning(
                old="config['{}']".format(k),
                help="config['replay_buffer_config']['{}']" "".format(k),
                error=False,
            )
            # Copy values over to new location in config to support new
            # and old configuration style.
            if config.get("replay_buffer_config") is not None:
                config["replay_buffer_config"][k] = config[k]

    replay_mode = config.get("multiagent", {}).get("replay_mode", DEPRECATED_VALUE)
    if replay_mode != DEPRECATED_VALUE:
        deprecation_warning(
            old="config['multiagent']['replay_mode']",
            help="config['replay_buffer_config']['replay_mode']",
            error=False,
        )
        config["replay_buffer_config"]["replay_mode"] = replay_mode

    # Can't use DEPRECATED_VALUE here because this is also a deliberate
    # value set for some algorithms
    # TODO: (Artur): Compare to DEPRECATED_VALUE on deprecation
    replay_sequence_length = config.get("replay_sequence_length", None)
    if replay_sequence_length is not None:
        config["replay_buffer_config"][
            "replay_sequence_length"
        ] = replay_sequence_length
        deprecation_warning(
            old="config['replay_sequence_length']",
            help="Replay sequence length specified at new "
            "location config['replay_buffer_config']["
            "'replay_sequence_length'] will be overwritten.",
            error=False,
        )

    replay_buffer_config = config["replay_buffer_config"]
    assert (
        "type" in replay_buffer_config
    ), "Can not instantiate ReplayBuffer from config without 'type' key."

    # Check if old replay buffer should be instantiated
    buffer_type = config["replay_buffer_config"]["type"]

    if isinstance(buffer_type, str) and buffer_type.find(".") == -1:
        # Create valid full [module].[class] string for from_config
        config["replay_buffer_config"]["type"] = (
            "ray.rllib.utils.replay_buffers." + buffer_type
        )

    # Instantiate a dummy buffer to fail early on misconfiguration and find out about
    # inferred buffer class
    dummy_buffer = from_config(buffer_type, config["replay_buffer_config"])

    config["replay_buffer_config"]["type"] = type(dummy_buffer)

    if hasattr(dummy_buffer, "update_priorities"):
        if config["multiagent"]["replay_mode"] == "lockstep":
            raise ValueError(
                "Prioritized replay is not supported when replay_mode=lockstep."
            )
        elif config["replay_buffer_config"].get("replay_sequence_length", 0) > 1:
            raise ValueError(
                "Prioritized replay is not supported when "
                "replay_sequence_length > 1."
            )
    else:
        if config["replay_buffer_config"].get("worker_side_prioritization"):
            raise ValueError(
                "Worker side prioritization is not supported when "
                "prioritized_replay=False."
            )


@DeveloperAPI
def warn_replay_buffer_capacity(*, item: SampleBatchType, capacity: int) -> None:
    """Warn if the configured replay buffer capacity is too large for machine's memory.

    Args:
        item: A (example) item that's supposed to be added to the buffer.
            This is used to compute the overall memory footprint estimate for the
            buffer.
        capacity: The capacity value of the buffer. This is interpreted as the
            number of items (such as given `item`) that will eventually be stored in
            the buffer.

    Raises:
        ValueError: If computed memory footprint for the buffer exceeds the machine's
            RAM.
    """
    if log_once("warn_replay_buffer_capacity"):
        item_size = item.size_bytes()
        psutil_mem = psutil.virtual_memory()
        total_gb = psutil_mem.total / 1e9
        mem_size = capacity * item_size / 1e9
        msg = (
            "Estimated max memory usage for replay buffer is {} GB "
            "({} batches of size {}, {} bytes each), "
            "available system memory is {} GB".format(
                mem_size, capacity, item.count, item_size, total_gb
            )
        )
        if mem_size > total_gb:
            raise ValueError(msg)
        elif mem_size > 0.2 * total_gb:
            logger.warning(msg)
        else:
            logger.info(msg)


def patch_buffer_with_fake_sampling_method(
    buffer: ReplayBuffer, fake_sample_output: SampleBatchType
) -> None:
    """Patch a ReplayBuffer such that we always sample fake_sample_output.

    Transforms fake_sample_output into a MultiAgentBatch if it is not a
    MultiAgentBatch and the buffer is a MultiAgentBuffer. This is useful for testing
    purposes if we need deterministic sampling.

    Args:
        buffer: The buffer to be patched
        fake_sample_output: The output to be sampled

    """
    if isinstance(buffer, MultiAgentReplayBuffer) and not isinstance(
        fake_sample_output, MultiAgentBatch
    ):
        fake_sample_output = SampleBatch(fake_sample_output).as_multi_agent()

    def fake_sample(_: Any = None, **kwargs) -> Optional[SampleBatchType]:
        """Always returns a predefined batch.

        Args:
            _: dummy arg to match signature of sample() method
            __: dummy arg to match signature of sample() method
            ``**kwargs``: dummy args to match signature of sample() method

        Returns:
            Predefined MultiAgentBatch fake_sample_output
        """

        return fake_sample_output

    buffer.sample = fake_sample
