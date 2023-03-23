from typing import List


from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.core.models.base import ACTOR
from ray.rllib.core.models.tf.encoder import ENCODER_OUT
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict

_, tf, _ = try_import_tf()

OLD_ACTION_DIST_KEY = "old_action_dist"
OLD_ACTION_DIST_LOGITS_KEY = "old_action_dist_logits"


class APPOTfRLModule(PPOTfRLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        catalog = self.config.get_catalog()
        self.old_pi = catalog.build_pi_head(framework=self.framework)
        self.old_encoder = catalog.build_actor_critic_encoder(framework=self.framework)
        self.old_pi.set_weights(self.pi.get_weights())
        self.old_encoder.set_weights(self.encoder.get_weights())
        self.old_pi.trainable = False
        self.old_encoder.trainable = False

    @override(PPOTfRLModule)
    def target_networks(self):
        return [(self.old_pi, self.pi), (self.old_encoder, self.encoder)]

    @override(PPOTfRLModule)
    def output_specs_train(self) -> List[str]:
        return [
            SampleBatch.ACTION_DIST,
            SampleBatch.VF_PREDS,
            OLD_ACTION_DIST_KEY,
        ]

    def _forward_train(self, batch: NestedDict):
        outs = super()._forward_train(batch)
        old_pi_inputs_encoded = self.old_encoder(batch)[ENCODER_OUT][ACTOR]
        old_action_dist_logits = self.old_pi(old_pi_inputs_encoded)
        old_action_dist = self.action_dist_cls.from_logits(old_action_dist_logits)
        outs[OLD_ACTION_DIST_KEY] = old_action_dist
        outs[OLD_ACTION_DIST_LOGITS_KEY] = old_action_dist_logits
        return outs
