
import ray
from collections import defaultdict
import numpy as np
import torch

from ray.rllib.core.trainer_runner import TrainerRunner
from ray.rllib.core.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.torch.torch_sarl_trainer import TorchSARLTrainer
from ray.rllib.utils.numpy import convert_to_numpy


from ray.rllib import SampleBatch


def model_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def model_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def make_dataset():
    size = 1000
    x = np.arange(0, 10, 10 / size, dtype=np.float32)
    a, b = 2, 5
    y = a * x + b
    return x, y


class DummyRLModuleA(TorchRLModule):
    def __init__(self, config):
        """This RL module has 2 networks a and b, and its output is a(x), b(x)"""
        super().__init__(config)
        self.config = config
        self.a = torch.nn.Linear(1, 1)

    def forward_train(self, batch, **kwargs):
        x = torch.reshape(torch.Tensor(batch["x"]), (-1, 1))
        return self.a(x)


class TorchIndependentModulesTrainerA(TorchSARLTrainer):
    def __init__(self, config):
        """Train networks a and b independently."""
        super().__init__(config)

    def make_module(self, module_config):
        # fixing the weights for this test
        torch.manual_seed(0)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0.01)
                m.bias.data.fill_(0.01)

        module = DummyRLModuleA(module_config)
        module.apply(init_weights)
        return module

    def make_optimizer(self, optimizer_config):
        optimizer_a = torch.optim.SGD(self.unwrapped_module.a.parameters(), lr=1e-3)
        optimizer_b = torch.optim.SGD(self.unwrapped_module.b.parameters(), lr=1e-3)
        return [optimizer_a, optimizer_b]

    def compute_loss(self, batch, fwd_out, **kwargs):
        out_a = fwd_out
        y = torch.reshape(torch.Tensor(batch["y"]), (-1, 1))
        loss_a = torch.nn.functional.mse_loss(out_a, y)

        return {
            "total_loss": loss_a,
            "loss_a": loss_a,
        }

    def compute_grads_and_apply_if_needed(self, batch, fwd_out, loss_out, **kwargs):
        loss = loss_out["total_loss"]
        for optimizer in self.optimizer:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizer:
            optimizer.step()
        return {}

    def compile_results(self, batch, fwd_out, loss_out, update_out, **kwargs):
        a_norm = model_norm(self.unwrapped_module.a)
        a_grad_norm = model_grad_norm(self.unwrapped_module.a)

        return {
            "compiled_results": {
                "a_norm": convert_to_numpy(a_norm),
                "a_grad_norm": convert_to_numpy(a_grad_norm),
            },
            "loss_out": convert_to_numpy(loss_out),
            "update_out": convert_to_numpy(update_out),
        }

class DummyRLModuleA(TorchRLModule):
    def __init__(self, config):
        """This RL module has 2 networks a and b, and its output is a(x), b(x)"""
        super().__init__(config)
        self.config = config
        self.a = torch.nn.Linear(1, 1)

    def forward_train(self, batch, **kwargs):
        x = torch.reshape(torch.Tensor(batch["x"]), (-1, 1))
        return self.a(x)


class TorchIndependentModulesTrainerA(TorchSARLTrainer):
    def __init__(self, config):
        """Train networks a and b independently."""
        super().__init__(config)

    def make_module(self, module_config):
        # fixing the weights for this test
        torch.manual_seed(0)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0.01)
                m.bias.data.fill_(0.01)

        module = DummyRLModuleA(module_config)
        module.apply(init_weights)
        return module

    def make_optimizer(self, optimizer_config):
        optimizer_a = torch.optim.SGD(self.unwrapped_module.a.parameters(), lr=1e-3)
        return [optimizer_a]

    def compute_loss(self, batch, fwd_out, **kwargs):
        out_a = fwd_out
        y = torch.reshape(torch.Tensor(batch["y"]), (-1, 1))
        loss_a = torch.nn.functional.mse_loss(out_a, y)

        return {
            "total_loss": loss_a,
            "loss_a": loss_a,
        }

    def compute_grads_and_apply_if_needed(self, batch, fwd_out, loss_out, **kwargs):
        loss = loss_out["total_loss"]
        for optimizer in self.optimizer:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizer:
            optimizer.step()
        return {}

    def compile_results(self, batch, fwd_out, loss_out, update_out, **kwargs):
        a_norm = model_norm(self.unwrapped_module.a)
        a_grad_norm = model_grad_norm(self.unwrapped_module.a)

        return {
            "compiled_results": {
                "a_norm": convert_to_numpy(a_norm),
                "a_grad_norm": convert_to_numpy(a_grad_norm),
            },
            "loss_out": convert_to_numpy(loss_out),
            "update_out": convert_to_numpy(update_out),
        }


if __name__ == "__main__":
    # ray.init(num_gpus=1)
    trainer_modules = [TrainerRunner(TorchIndependentModulesTrainerA, {"module_config": {}, "num_gpus": 0.5}) for _ in range(4)]

    batch_size = 10
    x, y = make_dataset()
    result = defaultdict(list)
    for i in range(3):
        batch = SampleBatch(
                {
                    "x": x[i * batch_size : (i + 1) * batch_size],
                    "y": y[i * batch_size : (i + 1) * batch_size],
                }
            )
        for idx, trainer_module in enumerate(trainer_modules):
            result[idx].append(trainer_module.update(batch))
    print(result)


