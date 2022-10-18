import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import ray

from ray.rllib.core.sarl_trainer import SARLTrainer
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


class TorchSARLTrainer(SARLTrainer):
    def __init__(self, config):
        super().__init__(config)
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) <= 1, (
            "Distributed trainers by default only " "support 1 GPU per worker"
        )
        self._gpu_id = gpu_ids[0] if gpu_ids else None
        # if self._gpu_id:

        self._rank = 0
        self._world_size = 1
        self.gpu_id = gpu_ids[0] if gpu_ids else None

    def _init_model(self):
        # TODO: make this a public method and document where it's called from and why it should not be modified
        """Do not modify this function
        This function gets called by the trainer runner but can't be called
        inside of the init function since DDP has not been set up yet.
        """

        # TODO: move self.make_module call to the __init__ function
        module = self.make_module(self.config["module_config"])
        self._module = self._prepare_module(module)

        # TODO: depending on whether the DDP holds a copy or reference to the module we can move this call to the __init__ function.
        # TODO: question: does the optimizer need ddp.parameters() or just model.parameters()?
        self._optimizer = self.make_optimizer(self.config["module_config"])

    def _prepare_module(self, module):

        # TODO: write this outside to not shadow self
        class DDPWrapper(DDP):
            def forward_train(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        # TODO: share the code a bit
        if self.gpu_id is not None:
            module.to(self.gpu_id)
            pg = torch.distributed.new_group(
                list(range(self.config["_scaling_config"]["world_size"]))
            )
            module = DDPWrapper(module, device_ids=[self.gpu_id], process_group=pg)
        else:
            pg = torch.distributed.new_group(
                list(range(self.config["_scaling_config"]["world_size"]))
            )
            module = DDPWrapper(module, process_group=pg)
        return module

    def _prepare_sample_batch(self, batch):
        return convert_to_torch_tensor(batch, device=self.gpu_id)

    @property
    def unwrapped_module(self):
        # return the unwrapped module from the ddp wrapper
        return self._module.module
