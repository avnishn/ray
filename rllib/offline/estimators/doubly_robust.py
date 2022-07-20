import logging
from typing import Dict, Any
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import SampleBatchType
import numpy as np
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict

from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator

torch, nn = try_import_torch()

logger = logging.getLogger()


@DeveloperAPI
class DoublyRobust(OffPolicyEstimator):
    """The Doubly Robust estimator.

    Let s_t, a_t, and r_t be the state, action, and reward at timestep t.

    This method trains a Q-model for the evaluation policy \pi_e on behavior
    data generated by \pi_b. Currently, RLlib implements this using
    Fitted-Q Evaluation (FQE). You can also implement your own model
    and pass it in as `q_model_config = {"type": your_model_class, **your_kwargs}`.

    For behavior policy \pi_b and evaluation policy \pi_e, define the
    cumulative importance ratio at timestep t as:
    p_t = \sum_{t'=0}^t (\pi_e(a_{t'} | s_{t'}) / \pi_b(a_{t'} | s_{t'})).

    Consider an episode with length T. Let V_T = 0.
    For all t in {0, T - 1}, use the following recursive update:
    V_t^DR = (\sum_{a \in A} \pi_e(a | s_t) Q(s_t, a))
        + p_t * (r_t + \gamma * V_{t+1}^DR - Q(s_t, a_t))

    This estimator computes the expected return for \pi_e for an episode as:
    V^{\pi_e}(s_0) = V_0^DR
    and returns the mean and standard deviation over episodes.

    For more information refer to https://arxiv.org/pdf/1911.06854.pdf"""

    @override(OffPolicyEstimator)
    def __init__(
        self,
        policy: Policy,
        gamma: float,
        q_model_config: Dict = None,
    ):
        """Initializes a Doubly Robust OPE Estimator.

        Args:
            policy: Policy to evaluate.
            gamma: Discount factor of the environment.
            q_model_config: Arguments to specify the Q-model. Must specify
            a `type` key pointing to the Q-model class.
            This Q-model is trained in the train() method and is used
            to compute the state-value and Q-value estimates
            for the DoublyRobust estimator.
            It must implement `train`, `estimate_q`, and `estimate_v`.
            TODO (Rohan138): Unify this with RLModule API.
        """

        super().__init__(policy, gamma)
        model_cls = q_model_config.pop("type")

        self.model = model_cls(
            policy=policy,
            gamma=gamma,
            **q_model_config,
        )
        assert hasattr(
            self.model, "estimate_v"
        ), "self.model must implement `estimate_v`!"
        assert hasattr(
            self.model, "estimate_q"
        ), "self.model must implement `estimate_q`!"

    @override(OffPolicyEstimator)
    def estimate(self, batch: SampleBatchType) -> Dict[str, Any]:
        """Compute off-policy estimates.

        Args:
            batch: The SampleBatch to run off-policy estimation on

        Returns:
            A dict consists of the following metrics:
            - v_behavior: The discounted return averaged over episodes in the batch
            - v_behavior_std: The standard deviation corresponding to v_behavior
            - v_target: The estimated discounted return for `self.policy`,
            averaged over episodes in the batch
            - v_target_std: The standard deviation corresponding to v_target
            - v_gain: v_target / max(v_behavior, 1e-8), averaged over episodes
            - v_gain_std: The standard deviation corresponding to v_gain
        """
        batch = self.convert_ma_batch_to_sample_batch(batch)
        self.check_action_prob_in_batch(batch)
        estimates = {"v_behavior": [], "v_target": [], "v_gain": []}
        # Calculate doubly robust OPE estimates
        for episode in batch.split_by_episode():
            rewards, old_prob = episode["rewards"], episode["action_prob"]
            log_likelihoods = compute_log_likelihoods_from_input_dict(
                self.policy, episode
            )
            new_prob = np.exp(convert_to_numpy(log_likelihoods))

            v_behavior = 0.0
            v_target = 0.0
            q_values = self.model.estimate_q(episode)
            q_values = convert_to_numpy(q_values)
            v_values = self.model.estimate_v(episode)
            v_values = convert_to_numpy(v_values)
            assert q_values.shape == v_values.shape == (episode.count,)

            for t in reversed(range(episode.count)):
                v_behavior = rewards[t] + self.gamma * v_behavior
                v_target = v_values[t] + (new_prob[t] / old_prob[t]) * (
                    rewards[t] + self.gamma * v_target - q_values[t]
                )
            v_target = v_target.item()

            estimates["v_behavior"].append(v_behavior)
            estimates["v_target"].append(v_target)
            estimates["v_gain"].append(v_target / max(v_behavior, 1e-8))
        estimates["v_behavior_std"] = np.std(estimates["v_behavior"])
        estimates["v_behavior"] = np.mean(estimates["v_behavior"])
        estimates["v_target_std"] = np.std(estimates["v_target"])
        estimates["v_target"] = np.mean(estimates["v_target"])
        estimates["v_gain_std"] = np.std(estimates["v_gain"])
        estimates["v_gain"] = np.mean(estimates["v_gain"])
        return estimates

    @override(OffPolicyEstimator)
    def train(self, batch: SampleBatchType) -> Dict[str, Any]:
        """Trains self.model on the given batch.

        Args:
        batch: A SampleBatch or MultiAgentbatch to train on

        Returns:
        A dict with key "loss" and value as the mean training loss.
        """
        batch = self.convert_ma_batch_to_sample_batch(batch)
        losses = self.model.train(batch)
        return {"loss": np.mean(losses)}
