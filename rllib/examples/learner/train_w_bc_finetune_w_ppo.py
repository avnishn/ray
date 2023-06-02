import gymnasium as gym
import shutil
import tempfile
import torch
from typing import Mapping

import ray
from ray import tune
from ray.air import RunConfig, FailureConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.examples.datasets.dataset_utils import convert_json_sample_batch_to_df

GYM_ENV_NAME = "CartPole-v1"
GYM_ENV = gym.make(GYM_ENV_NAME)


class BCActor(torch.nn.Module):
    """A wrapper for the encoder and policy networks of a PPORLModule.

    Args:
        encoder_network: The encoder network of the PPORLModule.
        policy_network: The policy network of the PPORLModule.
        distribution_cls: The distribution class to construct with the logits outputed
            by the policy network.

    """

    def __init__(
        self,
        encoder_network: torch.nn.Module,
        policy_network: torch.nn.Module,
        distribution_cls: torch.distributions.Distribution,
    ):
        super().__init__()
        self.encoder_network = encoder_network
        self.policy_network = policy_network
        self.distribution_cls = distribution_cls

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.distributions.Distribution:
        """Return an action distribution output by the policy network.

        batch: A dict containing the key "obs" mapping to a torch tensor of
            observations.

        """
        encoder_out = self.encoder_network(batch)[ENCODER_OUT][ACTOR]
        action_logits = self.policy_network(encoder_out)
        distribution = self.distribution_cls(logits=action_logits)
        return distribution


def train_ppo_module_with_bc_finetune(
    dataset: ray.data.Dataset, ppo_module_spec: SingleAgentRLModuleSpec
) -> str:
    """Train an Actor with BC finetuning on dataset.

    Args:
        dataset: The dataset to train on.
        module_spec: The module spec of the PPORLModule that will be trained
            after its encoder and policy networks are pretrained with BC.

    Returns:
        The path to the checkpoint of the pretrained PPORLModule.
    """
    batch_size = 512
    learning_rate = 1e-3
    num_epochs = 10

    module = ppo_module_spec.build()
    # We want to pretrain the encoder and policy networks of the RLModule. We don't want
    # to pretrain the value network. The actor will use the Categorical distribution,
    # as its output distribution since we are training on the CartPole environment which
    # has a discrete action space.
    BCActorNetwork = BCActor(module.encoder, module.pi, torch.distributions.Categorical)
    optim = torch.optim.Adam(BCActorNetwork.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in dataset.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.float32
        ):
            action_dist = BCActorNetwork(batch)
            loss = -torch.mean(action_dist.log_prob(batch["actions"]))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch} loss: {loss.detach().item()}")

    checkpoint_dir = tempfile.mkdtemp()
    module.save_to_checkpoint(checkpoint_dir)
    return checkpoint_dir


def train_ppo_agent_from_checkpointed_module(
    module_spec_from_ckpt: SingleAgentRLModuleSpec,
):
    """Train a checkpointed RLModule using PPO.

    Args:
        module_spec_from_ckpt: The module spec of the checkpointed RLModule.

    """

    config = (
        PPOConfig()
        .training(_enable_learner_api=True)
        .rl_module(_enable_rl_module_api=True, rl_module_spec=module_spec_from_ckpt)
        .environment(GYM_ENV_NAME)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            stop={"training_iteration": 10},
            failure_config=FailureConfig(fail_fast="raise"),
            verbose=2,
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    ray.init()

    ray.data.set_progress_bars(False)

    # Read a directory of files in remote storage.
    rllib_path = ray.__path__[0] + "/rllib/"
    cartpole_dataset_path = rllib_path + "tests/data/cartpole/large.json"
    df = convert_json_sample_batch_to_df(cartpole_dataset_path)
    ds = ray.data.from_pandas(df)

    module_spec = SingleAgentRLModuleSpec(
        module_class=PPOTorchRLModule,
        observation_space=GYM_ENV.observation_space,
        action_space=GYM_ENV.action_space,
        model_config_dict={"fcnet_hiddens": [64, 64]},
        catalog_class=PPOCatalog,
    )

    # train a PPO Module with BC finetuning
    module_checkpoint_path = train_ppo_module_with_bc_finetune(ds, module_spec)

    module_spec_from_ckpt = SingleAgentRLModuleSpec(
        module_class=PPOTorchRLModule,
        observation_space=GYM_ENV.observation_space,
        action_space=GYM_ENV.action_space,
        model_config_dict={"fcnet_hiddens": [64, 64]},
        catalog_class=PPOCatalog,
        load_state_path=module_checkpoint_path,
    )

    train_ppo_agent_from_checkpointed_module(module_spec_from_ckpt)
    # clean up the checkpoint directory
    shutil.rmtree(module_checkpoint_path)
