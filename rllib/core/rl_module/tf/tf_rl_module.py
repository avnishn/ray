from typing import Any, Mapping

from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf


_, tf, _ = try_import_tf()


class TfRLModule(RLModule, tf.keras.Model):
    """Base class for RLlib TF RLModules."""

    def __init__(self, *args, **kwargs) -> None:
        tf.keras.Model.__init__(self)
        RLModule.__init__(self, *args, **kwargs)

    @override(tf.keras.Model)
    def call(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """Forward pass of the module.

        Note:
            This is aliased to forward_train to follow the Keras Model API.

        Args:
            batch: The input batch. This input batch should comply with
                input_specs_train().
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass. This output should comply with the
            ouptut_specs_train().

        """
        return self.forward_train(batch)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        return self.get_weights()

    @override(RLModule)
    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        self.set_weights(state_dict)
