from typing import Any, Mapping

from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf


_, tf, _ = try_import_tf()


class TFRLModule(RLModule, tf.keras.Model):
    def __init__(self, config: Mapping[str, Any]) -> None:
        tf.keras.Model.__init__(self)
        RLModule.__init__(self, config)

    @override(tf.keras.Model)
    def call(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """forward pass of the module.

        This is aliased to forward_train because Torch DDP requires a forward method to
        be implemented for backpropagation to work.
        """
        return self.forward_train(batch, **kwargs)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        return self.get_weights()

    @override(RLModule)
    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        self.set_weights(state_dict)

    @override(RLModule)
    def make_distributed(self, dist_config: Mapping[str, Any] = None) -> None:
        """Makes the module distributed."""
        # TODO (Avnish): Implement this.
        pass

    @override(RLModule)
    def is_distributed(self) -> bool:
        """Returns True if the module is distributed."""
        # TODO (Avnish): Implement this.
        return False

    def trainable_variables(self) -> Mapping[str, Any]:
        """Returns the trainable variables of the module.

        Example:
            return {"module": module.trainable_variables}

        Note:
            see tensorflow.org/guide/autodiff#gradients_with_respect_to_a_model
            for more details

        """
        raise NotImplementedError

    @override(RLModule)
    def get_multi_agent_class(cls) -> MultiAgentRLModule:
        """Returns the multi-agent wrapper class for this module."""
        raise NotImplementedError("Multi-agent not supported for TFRLModule yet")
