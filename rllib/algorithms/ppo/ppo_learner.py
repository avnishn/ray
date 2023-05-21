from collections import defaultdict
from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Union

from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules.scheduler import Scheduler


LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY = "vf_loss_unclipped"
LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY = "vf_explained_var"
LEARNER_RESULTS_KL_KEY = "mean_kl_loss"
LEARNER_RESULTS_CURR_KL_COEFF_KEY = "curr_kl_coeff"
LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY = "curr_entropy_coeff"


logger = logging.getLogger(__name__)


@dataclass
class PPOLearnerHyperparameters(LearnerHyperparameters):
    """Hyperparameters for the PPOLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the PPOConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.ppo.ppo::PPOConfig::training()` for more details on the
    individual properties.
    """

    kl_coeff: float = None
    kl_target: float = None
    use_critic: bool = None
    clip_param: float = None
    vf_clip_param: float = None
    entropy_coeff: float = None
    entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = None
    vf_loss_coeff: float = None


class PPOLearner(Learner):
    @override(Learner)
    def build(self) -> None:
        super().build()

        # Build entropy coeff scheduling tools.
        self.entropy_coeff_scheduler = Scheduler(
            fixed_value=self.hps.entropy_coeff,
            schedule=self.hps.entropy_coeff_schedule,
            framework=self.framework,
            device=self._device,
        )

        # Set up KL coefficient variables (per module).
        # Note that the KL coeff is not controlled by a schedul, but seeks
        # to stay close to a given kl_target value.
        self.curr_kl_coeffs_per_module = defaultdict(
            lambda: self._get_tensor_variable(self.hps.kl_coeff)
        )

    @override(Learner)
    def additional_update_per_module(
        self, module_id: ModuleID, sampled_kl_values: dict, timestep: int
    ) -> Dict[str, Any]:
        results = super().additional_update_per_module(
            module_id,
            sampled_kl_values=sampled_kl_values,
            timestep=timestep,
        )

        # Update entropy coefficient via our Scheduler.
        new_entropy_coeff = self.entropy_coeff_scheduler.update(
            module_id, timestep=timestep
        )
        results.update({LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY: new_entropy_coeff})

        return results

    @override(Learner)
    def compile_results(
        self,
        batch: MultiAgentBatch,
        fwd_out: Mapping[str, Any],
        postprocessed_loss: Mapping[str, Any],
        postprocessed_gradients: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        for module_id, module_loss_results in postprocessed_loss.items():
            if module_id == self.TOTAL_LOSS_KEY:
                continue
            if math.isinf(module_loss_results[LEARNER_RESULTS_KL_KEY]):
                logger.warning(
                    "KL divergence is non-finite, this will likely destabilize "
                    "your model and the training process. Action(s) in a "
                    "specific state have near-zero probability. "
                    "This can happen naturally in deterministic "
                    "environments where the optimal policy has zero mass "
                    "for a specific action. To fix this issue, consider "
                    "setting `kl_coeff` to 0.0 or increasing `entropy_coeff` in your "
                    "config."
                )
        return super().compile_results(
            batch, fwd_out, postprocessed_loss, postprocessed_gradients
        )
