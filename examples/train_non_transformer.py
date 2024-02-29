import random
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from examples.data import get_data_electricity, get_loader, get_loader_overfit, get_data_electricity_hourly
from examples.model import QuadraticModel
from examples.util import save_model, mean_variance
from modular_transformer.classical import ClassicalTransformer

from tqdm import tqdm

from modular_transformer.taylor import TaylorTransformer


def moving_average(x, w):
    return [sum(x[i:i + w]) / w for i in range(len(x) - w + 1)]


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
):
    running_loss = 0.0
    losses = []
    len_data = len(training_loader)

    for i, (inputs, labels) in enumerate(tqdm(training_loader)):
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        # inputs = inputs.unsqueeze(-1)
        size_batch = inputs.shape[0]
        assert size_batch <= batch_size
        # assert inputs.shape == (size_batch, input_length, 1)
        labels_masked = labels.clone()
        labels_masked[:, :, 0] = 0.
        outputs = model(inputs).unsqueeze(-1)
        assert outputs.shape == (size_batch, output_length, 1)
        assert not outputs.isnan().any()

        loss = loss_fn(outputs.squeeze(-1), labels[:, :, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 0:
            print(f"Running loss until batch {i}: {running_loss / (i + 1)}")
        losses.append(loss.item())

    epoch_loss = running_loss / len_data
    # print(f"LOSSES TRAIN: {losses}")
    # import matplotlib.pyplot as plt
    # plt.plot(moving_average(losses, 50))
    # plt.title("LOSSES TRAIN")
    # plt.show()
    return epoch_loss


@torch.no_grad()
def plot_samples(model: ClassicalTransformer, test_loader: DataLoader, title: str = ""):
    import matplotlib.pyplot as plt

    n = 0
    for i, (vinputs, vlabels) in enumerate(test_loader):
        if i > 5:
            break
        # vinputs = vinputs.unsqueeze(-1)

        size_batch = vinputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
        # assert vinputs.shape == (size_batch, input_length, 1)
        vlabels_masked = vlabels.clone()
        vlabels_masked[:, :, 0] = 0.
        voutputs = model(vinputs).unsqueeze(-1)

        vinputs = vinputs[0, :, 0]
        voutputs = voutputs[0, :, 0]
        vlabels = vlabels[0, :, 0]
        mse_loss = torch.nn.MSELoss()(voutputs, vlabels)
        # if mse_loss < 1:
        #     continue
        # n += 1
        # if n > 5:
        #     break
        # print("vinputs")
        # print(vinputs)
        # print("voutputs")
        # print(voutputs)
        # print("vlabels")
        # print(vlabels)

        plt.plot(vinputs.numpy(), label="inputs")
        # plot outputs and ground truth behind input sequence
        plt.plot(range(input_length, input_length + output_length), voutputs.numpy(), label="outputs")
        plt.plot(range(input_length, input_length + output_length), vlabels.numpy(), label="ground truth")
        plt.title(f"{title}{' ' if title else ''}mse: {mse_loss:.5f}")
        plt.legend()
        plt.xticks(range(0, input_length + output_length, 2))
        plt.show()
        # time.sleep(5)


@torch.no_grad()
def evaluate_model(model, test_loader: DataLoader):
    losses_mse = []
    losses_mae = []
    model.eval()
    for vinputs, vlabels in test_loader:
        size_batch = vinputs.shape[0]
        assert size_batch <= batch_size
        voutputs = model(vinputs).unsqueeze(-1)
        vloss_mse = torch.nn.MSELoss()(voutputs.squeeze(-1), vlabels[:, :, 0])
        vloss_mae = torch.nn.L1Loss()(voutputs.squeeze(-1), vlabels[:, :, 0])
        losses_mse.append(vloss_mse.item())
        losses_mae.append(vloss_mae.item())

    mean_mse, variance_mse = mean_variance(losses_mse)
    mean_mae, variance_mae = mean_variance(losses_mae)
    return mean_mse, variance_mse, mean_mae, variance_mae


input_length = 20
output_length = 10

model = QuadraticModel(dim_in=input_length, hidden_layers=[], dim_out=output_length)
# model = QuadraticModel(dim_in=input_length, hidden_layers=[10], dim_out=output_length)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

epochs = 50
batch_size = 20
hourly = True

if hourly:
    data_train, data_test = get_data_electricity_hourly()
else:
    data_train, data_test = get_data_electricity()
train_loader = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    positional_encoding=False
)
test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    shuffle=False,
    positional_encoding=False
)
test_loader_shuffle = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    shuffle=True,
    positional_encoding=False
)
# train_loader = get_loader_overfit(
#     data_train,
#     input_length=input_length,
#     output_length=output_length,
#     batch_size=batch_size,
# )
# test_loader = train_loader

learning_rate = 0.0005
# learning_rate = 0.001
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mses_train = []
mses_test_mean = []
mses_test_variance = []
maes_test_mean = []
maes_test_variance = []

start_time = datetime.now()
for epoch in range(epochs):
    print(f"EPOCH {epoch}:")

    model.train(True)
    avg_loss = train_one_epoch(
        model,
        loss_fn,
        optimizer,
        train_loader,
    )
    mses_train.append(avg_loss)

    mean_mse, variance_mse, mean_mae, variance_mae = evaluate_model(model, test_loader)
    mses_test_mean.append(mean_mse)
    mses_test_variance.append(variance_mse)
    maes_test_mean.append(mean_mae)
    maes_test_variance.append(variance_mae)

    # plot_samples(model, test_loader)

    print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mean_mse:.5f}")

save_model(model, "quadratic", str(input_length), str(output_length))

import matplotlib.pyplot as plt
import math
plt.plot(mses_train, label="train mse")
plt.plot(mses_test_mean, label="test mse")
plt.title("losses")
plt.show()

plot_samples(model, test_loader_shuffle, "test")

plot_samples(model, train_loader, "train")
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
duration = datetime.now() - start_time
print(f"Duration: {duration}")

import polars as pl
metadata = pl.DataFrame(
    {
        "model_type": ["quadratic-model"],
        "input_length": [input_length],
        "output_length": [output_length],
        "epochs": [epochs],
        "batch_size": [batch_size],
        "learning_rate": [learning_rate],
        "mses_train": [mses_train],
        "mses_test_mean": [mses_test_mean],
        "mses_test_variance": [mses_test_variance],
        "maes_test_mean": [maes_test_mean],
        "maes_test_variance": [maes_test_variance],
        "num_params": [num_params],
        "duration": [duration],
    }
)
metadata.write_parquet("examples/model-weights/quadratic-metadata.parquet")
