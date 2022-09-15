import torch
import torch.nn as nn

import ray
from ray import train
from ray.air import session, Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

input_size = 1
layer_size = 15
output_size = 1
num_epochs = 50

def weight_init(m):
    nn.init.xavier_normal_(m.weight)
    nn.init.zeros_(m.bias)

def weight_init_zero(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)

class NeuralNetwork(nn.Module):
    def __init__(self, offset=0, zeroed=False):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size+offset)
        self.layer2 = nn.Linear(layer_size+offset, output_size)
        if not zeroed:
            self.layer1.apply(weight_init)
            self.layer2.apply(weight_init)
        else:
            self.layer1.apply(weight_init_zero)
            self.layer2.apply(weight_init_zero)

    def forward(self, input):
        return self.layer2(nn.functional.relu(self.layer1(input)))


def train_loop_per_worker():
    dataset_shard = session.get_dataset_shard("train")
    model = NeuralNetwork()
    model2 = NeuralNetwork(offset=10)
    loss_fn = nn.MSELoss()
    loss_fn2 = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

    model = train.torch.prepare_model(model)
    model2 = train.torch.prepare_model(model2)

    for epoch in range(num_epochs):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float
        ):
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"].to()
            output = model(inputs)
            labels = labels.to(output.get_device())
            out_2 = model2(inputs)
            loss = loss_fn(output, labels)
            loss2 = loss_fn2(out_2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            # print(f"epoch: {epoch}, loss: {loss.item()}")
            print(f"epoch: {epoch}, loss: {loss.item()}, loss2: {loss2.item()}")

        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.state_dict())
            ),
        )


train_dataset = ray.data.from_items([{"x": x, "y": 2 * x + 1} for x in range(200)])
scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
# If using GPUs, use the below scaling config instead.
# scaling_config = ScalingConfig(num_workers=3, use_gpu=True)
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=scaling_config,
    datasets={"train": train_dataset},
)
result = trainer.fit()