from dataclasses import dataclass, field
from tabnanny import check

from ray.rllib.core.rl_module.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.models.specs.specs_dict import ModelSpecDict, check_specs
from ray.rllib.models.specs.specs_torch import TorchSpecs
from ray.rllib.models.torch.torch_action_dist_v2 import (
    TorchCategorical,
    TorchDeterministic,
    TorchDiagGaussian,
)

import torch
import torch.nn as nn
import gym

from typing import List, Mapping, Any

def get_ppo_loss(fwd_in, fwd_out):
    # this is not exactly a ppo loss, just something to show that the
    # forward train works
    adv = fwd_in["reward"] - fwd_out["vf"]
    actor_loss = -(fwd_out["logp"] * adv).mean()
    critic_loss = (adv ** 2).mean()
    loss = actor_loss + critic_loss

    return loss

def get_shared_encoder_config(env):
    return PPOModuleConfig(
        observation_space=env.observation_space,
        action_space=env.action_space,
        encoder_config=FCConfig(
            hidden_layers=[32],
            activation="ReLU",
        ),
        pi_config=FCConfig(
            hidden_layers=[32],
            activation="ReLU",
        ),
        vf_config=FCConfig(
            hidden_layers=[32],
            activation="ReLU",
        ),
    )

def get_separate_encoder_config(env):
    return PPOModuleConfig(
        observation_space=env.observation_space,
        action_space=env.action_space,
        pi_config=FCConfig(
            hidden_layers=[32],
            activation="ReLU",
        ),
        vf_config=FCConfig(
            hidden_layers=[32],
            activation="ReLU",
        ),
    )


@dataclass
class FCConfig:
    """Configuration for a fully connected network.

    Attributes:
        input_dim: The input dimension of the network. It cannot be None.
        output_dim: The output dimension of the network. if None, the last layer would
            be the last hidden layer.
        hidden_layers: The sizes of the hidden layers.
        activation: The activation function to use after each layer.
    """

    input_dim: int = None
    output_dim: int = None
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    output_activation: str = None


@dataclass
class PPOModuleConfig:
    """Configuration for the PPO module.

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        pi_config: The configuration for the policy network.
        vf_config: The configuration for the value network.
        encoder_config: The configuration for the encoder network.
    """

    observation_space: gym.Space = None
    action_space: gym.Space = None
    pi_config: FCConfig = None
    vf_config: FCConfig = None
    encoder_config: FCConfig = None


class FCNet(nn.Module):
    """A simple fully connected network.

    Attributes:
        input_dim: The input dimension of the network. It cannot be None.
        output_dim: The output dimension of the network. if None, the last layer would
            be the last hidden layer.
        hidden_layers: The sizes of the hidden layers.
        activation: The activation function to use after each layer.
    """

    def __init__(self, config: FCConfig):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_layers = config.hidden_layers
        self.activation = config.activation
        self.output_activation = config.output_activation

        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        for i in range(len(self.hidden_layers) - 1):
            self.layers.append(
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1])
            )
        if config.output_dim is not None:
            self.layers.append(nn.Linear(self.hidden_layers[-1], config.output_dim))

        if config.output_dim is None:
            self.output_dim = config.hidden_layers[-1]
        else:
            self.output_dim = config.output_dim

        self.layers = nn.ModuleList(self.layers)
        self.activation = getattr(nn, self.activation)()
        if self.output_activation is not None:
            self.output_activation = getattr(nn, self.output_activation)()

    def input_specs(self):
        return ModelSpecDict({"input": TorchSpecs("b, h", h=self.input_dim)})

    @check_specs(input_spec="input_specs")
    def forward(self, x):
        x = x["input"]
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class SimplePPOModule(TorchRLModule):
    def __init__(self, config: PPOModuleConfig) -> None:
        super().__init__(config)

        assert (
            isinstance(self.config.observation_space, gym.spaces.Box),
            "This simple PPOModule only supports Box observation space.",
        )

        assert (
            self.config.observation_space.shape[0] == 1,
            "This simple PPOModule only supports 1D observation space.",
        )

        assert (
            isinstance(self.config.action_space, (gym.spaces.Discrete, gym.spaces.Box)),
            "This simple PPOModule only supports Discrete and Box action space.",
        )

        assert self.config.pi_config, "pi_config must be provided."
        assert self.config.vf_config, "vf_config must be provided."

        self.encoder = None
        if self.config.encoder_config:
            encoder_config = self.config.encoder_config
            encoder_config.input_dim = self.config.observation_space.shape[0]
            self.encoder = FCNet(self.config.encoder_config)

        # build pi network
        pi_config = self.config.pi_config
        if not self.encoder:
            pi_config.input_dim = self.config.observation_space.shape[0]
        else:
            pi_config.input_dim = self.encoder.output_dim

        if isinstance(self.config.action_space, gym.spaces.Discrete):
            pi_config.output_dim = self.config.action_space.n
            pi_config.output_activation = "Softmax"
        else:
            pi_config.output_dim = self.config.action_space.shape[0] * 2
        self.pi = FCNet(pi_config)

        # build vf network
        vf_config = self.config.vf_config
        if not self.encoder:
            vf_config.input_dim = self.config.observation_space.shape[0]
        else:
            vf_config.input_dim = self.encoder.output_dim

        vf_config.output_dim = 1
        self.vf = FCNet(vf_config)

        self._is_discrete = isinstance(self.config.action_space, gym.spaces.Discrete)

    @override(RLModule)
    def input_specs_inference(self) -> ModelSpecDict:
        return self.input_specs_exploration()

    @override(RLModule)
    def output_specs_inference(self) -> ModelSpecDict:
        return ModelSpecDict(
            {
                "action_dist": TorchDeterministic,
            }
        )

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        if self.encoder:
            # TODO (kourosh): This {"input": xxx} is very verbose.
            encoded_state = self.encoder({"input": batch["obs"]})
            pi_out = self.pi({"input": encoded_state})
        else:
            pi_out = self.pi({"input": batch["obs"]})

        if self._is_discrete:
            action = torch.argmax(pi_out, dim=-1)
        else:
            action, _ = pi_out.chunk(2, dim=-1)

        action_dist = TorchDeterministic(action)
        return {"action_dist": action_dist}

    @override(RLModule)
    def input_specs_exploration(self):
        return ModelSpecDict(
            {
                "obs": self.encoder.input_specs()["input"]
                if self.encoder
                else self.pi.input_specs()["input"],
            }
        )

    @override(RLModule)
    def output_specs_exploration(self) -> ModelSpecDict:
        return ModelSpecDict(
            {
                "action_dist": TorchCategorical
                if self._is_discrete
                else TorchDiagGaussian,
            }
        )

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        if self.encoder:
            encoded_state = self.encoder({"input": batch["obs"]})
            pi_out = self.pi({"input": encoded_state})
        else:
            pi_out = self.pi({"input": batch["obs"]})

        if self._is_discrete:
            action_dist = TorchCategorical(pi_out)
        else:
            mu, scale = pi_out.chunk(2, dim=-1)
            action_dist = TorchDiagGaussian(mu, scale.exp())

        return {"action_dist": action_dist}

    @override(RLModule)
    def input_specs_train(self) -> ModelSpecDict:
        if self._is_discrete:
            action_dim = 1
        else:
            action_dim = self.config.action_space.shape[0]

        return ModelSpecDict(
            {
                "obs": self.encoder.input_specs
                if self.encoder
                else self.pi.input_specs,
                "action": TorchSpecs("b, da", da=action_dim),
            }
        )

    @override(RLModule)
    def output_specs_train(self) -> ModelSpecDict:
        return ModelSpecDict(
            {
                "action_dist": TorchCategorical
                if self._is_discrete
                else TorchDiagGaussian,
                "logp": TorchSpecs("b", dtype=torch.float32),
                "entropy": TorchSpecs("b", dtype=torch.float32),
                "vf": TorchSpecs("b", dtype=torch.float32),
            }
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        if self.encoder:
            encoded_state = self.encoder({"input": batch["obs"]})
            pi_out = self.pi({"input": encoded_state})
            vf = self.vf({"input": encoded_state})
        else:
            pi_out = self.pi({"input": batch["obs"]})
            vf = self.vf({"input": batch["obs"]})

        if self._is_discrete:
            action_dist = TorchCategorical(pi_out)
        else:
            mu, scale = pi_out.chunk(2, dim=-1)
            action_dist = TorchDiagGaussian(mu, scale.exp())

        logp = action_dist.logp(batch["action"].squeeze(-1))
        entropy = action_dist.entropy()
        return {
            "action_dist": action_dist,
            "logp": logp,
            "entropy": entropy,
            "vf": vf.squeeze(-1),
        }
