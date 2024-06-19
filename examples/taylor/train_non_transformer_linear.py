from datetime import datetime

import matplotlib.pyplot as plt
import torch

from examples.model import get_hidden_size, get_linear_model, size_quadratic
from examples.util import (
    Dataset,
    LossFunction,
    add_to_results,
    evaluate_model,
    get_data_loaders,
    get_device,
    get_loss_function,
    plot_samples,
    save_model,
    train_one_epoch,
)


def train_linear_model(
    input_length: int,
    output_length: int,
    dataset: Dataset,
    loss_function: LossFunction,
    epochs: int,
    batch_size: int = 20,
    plot: bool = True,
):
    quadratic_size = size_quadratic(input_length, output_length, [10])
    hidden_size = get_hidden_size(
        lambda h: (input_length + 1) * h + (h + 1) * output_length, quadratic_size
    )
    # hidden_size = 20
    device = get_device()
    model = get_linear_model(
        input_size=input_length, output_size=output_length, hidden_size=hidden_size, device=device
    )
    train_loader, test_loader = get_data_loaders(
        dataset,
        input_length,
        output_length,
        batch_size=20,
        positional_encoding=False,
        device=device,
    )

    loss_fn = get_loss_function(loss_function)
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.3)

    mses_train = []
    mses_test_mean = []
    mses_test_variance = []
    maes_test_mean = []
    maes_test_variance = []
    mapes_test_mean = []
    mapes_test_variance = []

    start_time = datetime.now()
    for epoch in range(epochs):
        print(f"EPOCH {epoch}:")

        avg_loss = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            use_labels=False,
        )
        mses_train.append(avg_loss)

        mean_mse, variance_mse, mean_mae, variance_mae, mean_mape, variance_mape = evaluate_model(
            model,
            test_loader,
            use_labels=False,
        )
        mses_test_mean.append(mean_mse)
        mses_test_variance.append(variance_mse)
        maes_test_mean.append(mean_mae)
        maes_test_variance.append(variance_mae)
        mapes_test_mean.append(mean_mape)
        mapes_test_variance.append(variance_mape)

        # plot_samples(model, test_loader)

        print(f"Train {loss_function}: {avg_loss:.5f}\nTest MSE:  {mean_mse:.5f}")
    duration = datetime.now() - start_time

    save_model(model, "linear", dataset, input_length, output_length, loss_function)

    if plot:
        plt.plot(mses_train, label="train mse")
        plt.plot(mses_test_mean, label="test mse")
        plt.title("losses")
        plt.legend()
        plt.show()

        plot_samples(model, test_loader, use_labels=False, num_samples=2, title="test")
        plot_samples(model, train_loader, use_labels=False, num_samples=1, title="train")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print(f"Duration: {duration}")

    add_to_results(
        "linear",
        dataset,
        input_length,
        output_length,
        epochs,
        batch_size,
        loss_function,
        learning_rate,
        num_params,
        duration.total_seconds(),
        mses_train,
        mses_test_mean,
        mses_test_variance,
        maes_test_mean,
        maes_test_variance,
        mapes_test_mean,
        mapes_test_variance,
    )


if __name__ == "__main__":
    _input_length = 12
    _output_length = 24
    _epochs = 100
    _dataset: Dataset = "electricity-hourly"
    _loss_function: LossFunction = "mse"
    train_linear_model(
        _input_length,
        _output_length,
        _dataset,
        _loss_function,
        _epochs,
    )
