import abc
from typing import Any, Mapping, Union, Type

from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.optim.rl_optimizer import RLOptimizer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType


class RLTrainer:
    """Base class for rllib algorithm trainers.
    
    Args:
        module_class: The (MA)RLModule class to use.
        module_config: The config for the (MA)RLModule.
        optimizer_class: The optimizer class to use.
        optimizer_config: The config for the optimizer.


    """

    def __init__(
        self,
        module_class: Type[RLModule],
        module_config: Mapping[str, Any],
        optimizer_class: Type[RLOptimizer],
        optimizer_config: Mapping[str, Any],
        distributed: bool = False,
    ):
        self.module_class = module_class
        self.module_config = module_config
        self.optimizer_class = optimizer_class
        self.optimizer_config = optimizer_config
        if distributed:
            self.module, self.optimizer = self.make_distributed()
        else:
            self.module = module_class.from_model_config(**module_config)
            self.optimizer = optimizer_class(optimizer_config)

    @staticmethod
    def on_after_compute_gradients(
        gradients_dict: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Called after gradients have been computed.

        Args:
            gradients_dict (Mapping[str, Any]): A dictionary of gradients.

        Note the relative order of operations looks like this:
            fwd_out = forward_train(batch)
            loss = compute_loss(batch, fwd_out)
            gradients = compute_gradients(loss)
            ---> post_processed_gradients = on_after_compute_gradients(gradients)
            apply_gradients(post_processed_gradients)

        Returns:
            Mapping[str, Any]: A dictionary of gradients.
        """
        return gradients_dict

    def compile_results(
        self,
        batch: NestedDict,
        fwd_out: Mapping[str, Any],
        postprocessed_loss: Mapping[str, Any],
        post_processed_gradients: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Compile results from the update.

        Args:
            batch: The batch that was used for the update.
            fwd_out: The output of the forward train pass.
            postprocessed_loss: The loss after postprocessing.
            post_processed_gradients: The gradients after postprocessing.

        Returns:
            A dictionary of results.
        """
        loss_numpy = convert_to_numpy(postprocessed_loss)
        return {
            "avg_reward": batch["rewards"].mean(),
            **loss_numpy,
        }

    @abc.abstractmethod
    def update(self, batch: SampleBatch) -> Mapping[str, Any]:
        """Perform an update on this Trainer.

        Args:
            batch: A batch of data.

        Returns:
            A dictionary of results.
        """

    @abc.abstractmethod
    def compute_gradients(
        self, loss: Union[TensorType, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Perform an update on self._module.

        For example compute and apply gradients to self._module if
        necessary.

        Args:
            loss: variable(s) used for optimizing self._module.

        Returns:
            A dictionary of extra information and statistics.
        """

    @abc.abstractmethod
    def apply_gradients(self, gradients: Mapping[str, Any]) -> None:
        """Perform an update on self._module"""

    @abc.abstractmethod
    def set_state(self, state: Mapping[str, Any]) -> None:
        """Set the state of the trainer."""

    @abc.abstractmethod
    def get_state(self) -> Mapping[str, Any]:
        """Get the state of the trainer."""

    @abc.abstractmethod
    def make_distributed(self) -> RLModule:
        """Make the module distributed."""
