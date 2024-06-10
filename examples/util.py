import pathlib
from typing import Callable, Literal

import polars as pl
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from examples.data import (
    get_data_electricity,
    get_data_electricity_hourly,
    get_data_ett,
    get_loader,
)

PATH_RESULTS = "examples/model-weights-final2"


Dataset = Literal["electricity", "electricity-hourly", "etth1", "etth2", "ettm1", "ettm2"]
LossFunction = Literal["mse", "mae", "mape"]


def mean_variance(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, variance


def moving_average(x, w):
    return [sum(x[i : i + w]) / w for i in range(len(x) - w + 1)]


def model_path(
    name: str, dataset: Dataset, input_length: int, output_length: int, loss_function: LossFunction
):
    return f"{PATH_RESULTS}/{name}-{dataset}-{loss_function}-input{input_length}-output{output_length}.pt"


def save_model(
    model,
    name: str,
    dataset: Dataset,
    input_length: int,
    output_length: int,
    loss_function: LossFunction,
):
    path = model_path(name, dataset, input_length, output_length, loss_function)
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)


def load_model_weights(
    model,
    name: str,
    dataset: Dataset,
    input_length: int,
    output_length: int,
    loss_function: LossFunction,
):
    path = model_path(name, dataset, input_length, output_length, loss_function)
    model.load_state_dict(torch.load(path))


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

    for inputs, labels in training_loader:
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.0
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs.squeeze(-1)).unsqueeze(-1)
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
    losses_mape = []
    model.eval()
    for inputs, labels in test_loader:
        assert not inputs.isnan().any()
        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.0
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs.squeeze(-1)).unsqueeze(-1)
        loss_mse = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels[:, :, 0])
        loss_mae = torch.nn.functional.l1_loss(outputs.squeeze(-1), labels[:, :, 0])
        loss_mape = torchmetrics.functional.mean_absolute_percentage_error(
            outputs.squeeze(-1), labels[:, :, 0]
        )
        losses_mse.append(loss_mse.item())
        losses_mae.append(loss_mae.item())
        losses_mape.append(loss_mape.item())

    mean_mse, variance_mse = mean_variance(losses_mse)
    mean_mae, variance_mae = mean_variance(losses_mae)
    mean_mape, variance_mape = mean_variance(losses_mape)
    return mean_mse, variance_mse, mean_mae, variance_mae, mean_mape, variance_mape


@torch.no_grad()
def plot_samples(
    model: torch.nn.Module,
    test_loader: DataLoader,
    use_labels: bool,
    num_samples: int = 5,
    title: str = "",
):
    model.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        if i > num_samples:
            break

        _input_length = inputs.shape[1]
        _output_length = labels.shape[1]

        if use_labels:
            labels_masked = labels.clone()
            labels_masked[:, :, 0] = 0.0
            outputs = model(inputs, labels_masked)
        else:
            outputs = model(inputs.squeeze(-1)).unsqueeze(-1)

        inputs_view = inputs[0, :, 0]
        outputs = outputs[0, :, 0]
        labels_view = labels[0, :, 0]
        mse_loss = torch.nn.functional.mse_loss(outputs, labels_view)

        plt.plot(inputs_view.numpy(), label="inputs")
        # plot outputs and ground truth behind input sequence
        plt.plot(
            range(_input_length, _input_length + _output_length), outputs.numpy(), label="outputs"
        )
        plt.plot(
            range(_input_length, _input_length + _output_length),
            labels_view.numpy(),
            label="ground truth",
        )
        plt.title(f"{title}{' ' if title else ''}mse: {mse_loss:.5f}")
        plt.legend()
        plt.xticks(range(0, _input_length + _output_length, 2))
        plt.show()


def get_data_loaders(
    dataset: Dataset,
    input_length: int,
    output_length: int,
    batch_size: int,
    positional_encoding: bool,
    device: torch.device | None = None,
):
    match dataset:
        case "electricity":
            data_train, data_test = get_data_electricity(device=device)
        case "electricity-hourly":
            data_train, data_test = get_data_electricity_hourly(device=device)
        case "etth1":
            data_train, data_test = get_data_ett("h1", device=device)
        case "etth2":
            data_train, data_test = get_data_ett("h2", device=device)
        case "ettm1":
            data_train, data_test = get_data_ett("m1", device=device)
        case "ettm2":
            data_train, data_test = get_data_ett("m2", device=device)
        case _:
            raise ValueError(f"Unknown dataset {dataset}")
    train_loader = get_loader(
        data_train,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        shuffle=True,
        positional_encoding=positional_encoding,
        device=device,
    )
    test_loader = get_loader(
        data_test,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        shuffle=True,
        positional_encoding=positional_encoding,
        device=device,
    )
    return train_loader, test_loader


def get_loss_function(
    loss_function: LossFunction,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match loss_function:
        case "mse":
            return torch.nn.functional.mse_loss
        case "mae":
            return torch.nn.functional.l1_loss
        case "mape":
            return torchmetrics.functional.mean_absolute_percentage_error
        case _:
            raise ValueError(f"Unknown loss function {loss_function}")


def add_to_results(
    model_type: str,
    dataset: Dataset,
    input_length: int,
    output_length: int,
    epochs: int,
    batch_size: int,
    loss_function: LossFunction,
    learning_rate: float,
    num_params: int,
    duration: float,
    loss_train: list[float],
    mses_test_mean: list[float],
    mses_test_variance: list[float],
    maes_test_mean: list[float],
    maes_test_variance: list[float],
    mapes_test_mean: list[float],
    mapes_test_variance: list[float],
):
    results = {
        "model_type": model_type,
        "dataset": dataset,
        "input_length": input_length,
        "output_length": output_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss_function": loss_function,
        "learning_rate": learning_rate,
        "num_params": num_params,
        "duration": duration,
        "loss_train": loss_train,
        "mses_test_mean": mses_test_mean,
        "mses_test_variance": mses_test_variance,
        "maes_test_mean": maes_test_mean,
        "maes_test_variance": maes_test_variance,
        "mapes_test_mean": mapes_test_mean,
        "mapes_test_variance": mapes_test_variance,
    }
    path = f"{PATH_RESULTS}/results.parquet"
    result_df = pl.DataFrame([results])
    with pl.Config(tbl_cols=-1, tbl_width_chars=220):
        print(result_df)

    if not pathlib.Path(path).exists():
        results_df = result_df
    # append row to existing parquet file
    else:
        results_df = pl.read_parquet(path)
        results_df = results_df.vstack(result_df)
    results_df.write_parquet(path)
    return results


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
