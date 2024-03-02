from typing import Literal, Callable

import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.data import get_loader, get_data_electricity, get_data_electricity_hourly


def mean_variance(l: list[float]) -> tuple[float, float]:
    mean = sum(l) / len(l)
    variance = sum((x - mean) ** 2 for x in l) / len(l)
    return mean, variance


def moving_average(x, w):
    return [sum(x[i:i + w]) / w for i in range(len(x) - w + 1)]


def save_model(model, name: str, input_length: str, output_length: str):
    path = f"examples/model-weights/{name}-input{input_length}-output{output_length}.pt"
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
    use_labels: bool,
):
    running_loss = 0.0
    len_data = len(training_loader)
    model.train()

    for inputs, labels in tqdm(training_loader):
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs).unsqueeze(-1)
        assert not outputs.isnan().any()

        loss = loss_fn(outputs.squeeze(-1), labels[:, :, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len_data
    return epoch_loss


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    use_labels: bool,
):
    losses_mse = []
    losses_mae = []
    model.eval()
    for inputs, labels in test_loader:
        assert not inputs.isnan().any()
        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs).unsqueeze(-1)
        loss_mse = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels[:, :, 0])
        loss_mae = torch.nn.functional.l1_loss(outputs.squeeze(-1), labels[:, :, 0])
        losses_mse.append(loss_mse.item())
        losses_mae.append(loss_mae.item())

    mean_mse, variance_mse = mean_variance(losses_mse)
    mean_mae, variance_mae = mean_variance(losses_mae)
    return mean_mse, variance_mse, mean_mae, variance_mae


@torch.no_grad()
def plot_samples(
    model: torch.nn.Module,
    test_loader: DataLoader,
    use_labels: bool,
    num_samples: int = 5,
    title: str = ""
):
    model.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        if i > num_samples:
            break

        _input_length = inputs.shape[1]
        _output_length = labels.shape[1]

        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs).unsqueeze(-1)

        inputs = inputs[0, :, 0]
        outputs = outputs[0, :, 0]
        labels = labels[0, :, 0]
        mse_loss = torch.nn.functional.mse_loss(outputs, labels)

        plt.plot(inputs.numpy(), label="inputs")
        # plot outputs and ground truth behind input sequence
        plt.plot(range(_input_length, _input_length + _output_length), outputs.numpy(), label="outputs")
        plt.plot(range(_input_length, _input_length + _output_length), labels.numpy(), label="ground truth")
        plt.title(f"{title}{' ' if title else ''}mse: {mse_loss:.5f}")
        plt.legend()
        plt.xticks(range(0, _input_length + _output_length, 2))
        plt.show()


Dataset = Literal["electricity", "electricity-hourly"]
LossFunction = Literal["mse", "mae", "mape"]


def get_data_loaders(
    dataset: Dataset,
    input_length: int,
    output_length: int,
    batch_size: int,
    positional_encoding: bool,
):
    match dataset:
        case "electricity":
            data_train, data_test = get_data_electricity()
        case "electricity-hourly":
            data_train, data_test = get_data_electricity_hourly()
        case _:
            raise ValueError(f"Unknown dataset {dataset}")
    train_loader = get_loader(
        data_train,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        shuffle=True,
        positional_encoding=positional_encoding
    )
    test_loader = get_loader(
        data_test,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        shuffle=True,
        positional_encoding=positional_encoding
    )
    return train_loader, test_loader


def get_loss_function(loss_function: LossFunction) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match loss_function:
        case "mse":
            return torch.nn.functional.mse_loss
        case "mae":
            return torch.nn.functional.l1_loss
        case "mape":
            return torchmetrics.functional.mean_absolute_percentage_error
        case _:
            raise ValueError(f"Unknown loss function {loss_function}")
