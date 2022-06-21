import logging
from typing import List, Optional, Type

from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_TARGET_UPDATES,
    TARGET_NET_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
)
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict,
    ResultDict,
)

logger = logging.getLogger(__name__)


class CRRConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CRR)

        # fmt: off
        # __sphinx_doc_begin__
        # CRR-specific settings.
        self.weight_type = "bin"
        self.temperature = 1.0
        self.max_weight = 20.0
        self.advantage_type = "mean"
        self.n_action_sample = 4
        self.twin_q = True
        self.target_update_grad_intervals = 100
        # __sphinx_doc_end__
        # fmt: on
        self.replay_buffer_config = {
            "type": MultiAgentReplayBuffer,
            "capacity": 50000,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1000,
            "replay_batch_size": 32,
            # The number of contiguous environment steps to replay at once. This
            # may be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
        }
        self.actor_hiddens = [256, 256]
        self.actor_hidden_activation = "relu"
        self.critic_hiddens = [256, 256]
        self.critic_hidden_activation = "relu"
        self.critic_lr = 3e-4
        self.actor_lr = 3e-4
        self.tau = 5e-3

        # overriding the trainer config default
        self.num_workers = 0  # offline RL does not need rollout workers

    def training(
        self,
        *,
        weight_type: Optional[str] = None,
        temperature: Optional[float] = None,
        max_weight: Optional[float] = None,
        advantage_type: Optional[str] = None,
        n_action_sample: Optional[int] = None,
        twin_q: Optional[bool] = None,
        target_update_grad_intervals: Optional[int] = None,
        replay_buffer_config: Optional[dict] = None,
        actor_hiddens: Optional[List[int]] = None,
        actor_hidden_activation: Optional[str] = None,
        critic_hiddens: Optional[List[int]] = None,
        critic_hidden_activation: Optional[str] = None,
        tau: Optional[float] = None,
        **kwargs,
    ) -> "CRRConfig":

        """
        === CRR configs

        Args:
            weight_type: weight type to use `bin` | `exp`.
            temperature: the exponent temperature used in exp weight type.
            max_weight: the max weight limit for exp weight type.
            advantage_type: The way we reduce q values to v_t values `max` | `mean`.
            n_action_sample: the number of actions to sample for v_t estimation.
            twin_q: if True, uses pessimistic q estimation.
            target_update_grad_intervals: The frequency at which we update the
                target copy of the model in terms of the number of gradient updates
                applied to the main model.
            replay_buffer_config: The config dictionary for replay buffer.
            actor_hiddens: The number of hidden units in the actor's fc network.
            actor_hidden_activation: The activation used in the actor's fc network.
            critic_hiddens: The number of hidden units in the critic's fc network.
            critic_hidden_activation: The activation used in the critic's fc network.
            tau: Polyak averaging coefficient
                (making it 1 is reduces it to a hard update).
            **kwargs: forward compatibility kwargs

        Returns:
            This updated CRRConfig object.
        """
        super().training(**kwargs)

        if weight_type is not None:
            self.weight_type = weight_type
        if temperature is not None:
            self.temperature = temperature
        if max_weight is not None:
            self.max_weight = max_weight
        if advantage_type is not None:
            self.advantage_type = advantage_type
        if n_action_sample is not None:
            self.n_action_sample = n_action_sample
        if twin_q is not None:
            self.twin_q = twin_q
        if target_update_grad_intervals is not None:
            self.target_update_grad_intervals = target_update_grad_intervals
        if replay_buffer_config is not None:
            self.replay_buffer_config = replay_buffer_config
        if actor_hiddens is not None:
            self.actor_hiddens = actor_hiddens
        if actor_hidden_activation is not None:
            self.actor_hidden_activation = actor_hidden_activation
        if critic_hiddens is not None:
            self.critic_hiddens = critic_hiddens
        if critic_hidden_activation is not None:
            self.critic_hidden_activation = critic_hidden_activation
        if tau is not None:
            self.tau = tau

        return self


NUM_GRADIENT_UPDATES = "num_grad_updates"


class CRR(Algorithm):

    # TODO: we have a circular dependency for get
    #  default config. config -> Trainer -> config
    #  defining Config class in the same file for now as a workaround.

    def setup(self, config: PartialAlgorithmConfigDict):
        super().setup(config)
        # added a counter key for keeping track of number of gradient updates
        self._counters[NUM_GRADIENT_UPDATES] = 0
        # if I don't set this here to zero I won't see zero in the logs (defaultdict)
        self._counters[NUM_TARGET_UPDATES] = 0

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return CRRConfig().to_dict()

    @override(Algorithm)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from ray.rllib.algorithms.crr.torch import CRRTorchPolicy

            return CRRTorchPolicy
        else:
            raise ValueError("Non-torch frameworks are not supported yet!")

    @override(Algorithm)
    def training_step(self) -> ResultDict:

        # Sample training batch from replay buffer.
        batch = synchronous_parallel_sample(worker_set=self.workers)
        batch = batch.as_multi_agent()
        # Add batch to replay buffer.
        self.local_replay_buffer.add(batch)
        bsize = self.config["train_batch_size"]
        train_batch = self.local_replay_buffer.sample(bsize)
        # sampling stats doesn't technically mean anything in offline RL.
        # we should instead just replace this with the number of steps that
        # we are going to train on
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Postprocess batch before we learn on it.
        post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
        train_batch = post_fn(train_batch, self.workers, self.config)

        # Learn on training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        if self.config.get("simple_optimizer", False):
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # update target every few gradient updates
        cur_ts = self._counters[NUM_GRADIENT_UPDATES]
        last_update = self._counters[LAST_TARGET_UPDATE_TS]

        if cur_ts - last_update >= self.config["target_update_grad_intervals"]:
            with self._timers[TARGET_NET_UPDATE_TIMER]:
                to_update = self.workers.local_worker().get_policies_to_train()
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        self._counters[NUM_GRADIENT_UPDATES] += 1

        return train_results
