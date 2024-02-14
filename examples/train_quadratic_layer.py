import random
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from examples.data import get_data_electricity, get_loader, get_loader_overfit
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
        # other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        # other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / inputs.shape[1]
        # assert inputs.shape == (size_batch, input_length, 1)
        labels_masked = labels.clone()
        labels_masked[:, :, 0] = 0.
        outputs = model(inputs, labels_masked)
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
    import matplotlib.pyplot as plt
    plt.plot(moving_average(losses, 50))
    plt.title("LOSSES TRAIN")
    plt.show()
    return epoch_loss


@torch.no_grad()
def plot_samples(model: ClassicalTransformer, test_loader: DataLoader, title: str = ""):
    import matplotlib.pyplot as plt

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
        voutputs = model(vinputs, vlabels_masked)

        vinputs = vinputs[0, :, 0]
        voutputs = voutputs[0, :, 0]
        vlabels = vlabels[0, :, 0]
        mse_loss = torch.nn.MSELoss()(voutputs, vlabels)
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
    running_vloss_mse = 0.0
    running_vloss_mae = 0.0
    losses = []
    model.eval()
    for vinputs, vlabels in test_loader:
        size_batch = vinputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
        # assert vinputs.shape == (size_batch, input_length, 1)
        # vlabels_masked = vlabels
        # vlabels_masked[:, :, 0] = 0.
        voutputs = model(vinputs, vlabels)
        vloss_mse = torch.nn.MSELoss()(voutputs.squeeze(-1), vlabels[:, :, 0])
        vloss_mae = torch.nn.L1Loss()(voutputs.squeeze(-1), vlabels[:, :, 0])
        running_vloss_mse += vloss_mse
        running_vloss_mae += vloss_mae
        losses.append(vloss_mse.item())

    mean_losses = sum(losses) / len(losses)
    variance_losses = sum((x - mean_losses) ** 2 for x in losses) / len(losses)

    mse = running_vloss_mse / len(test_loader)
    mae = running_vloss_mae / len(test_loader)
    # print(f"LOSSES TEST: {losses}")
    import matplotlib.pyplot as plt
    plt.plot(moving_average(losses, 50))
    plt.title("LOSSES TEST")
    plt.show()
    return mse, mae


input_length = 20
output_length = 60

model = TaylorTransformer(
    input_features=3,
    output_features=1,
    sequence_length=input_length,
    d_model=5,
    nhead=1,
    dim_feedforward=200,
    num_encoder_layers=1,
    num_decoder_layers=1,
    final_activation=None,
)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

epochs = 3
batch_size = 20

data_train, data_test = get_data_electricity()
train_loader = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
)
test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    shuffle=False,
)
test_loader_shuffle = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    shuffle=True,
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
mses_test = []
maes_test = []

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

    mse_test, mae_test = evaluate_model(model, test_loader)
    mses_test.append(mse_test)
    maes_test.append(mae_test)

    # plot_samples(model, test_loader)

    print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mse_test:.5f}")


import matplotlib.pyplot as plt
import math
plt.plot([math.log(x) for x in mses_train], label="train mse")
plt.plot([math.log(x) for x in mses_test], label="test mse")
plt.title("losses (log scale)")
plt.show()

plt.plot(mses_train, label="train mse")
plt.plot(mses_test, label="test mse")
plt.title("losses")
plt.show()

plot_samples(model, test_loader_shuffle, "test")

plot_samples(model, train_loader, "train")
