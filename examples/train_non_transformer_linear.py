from datetime import datetime

import matplotlib.pyplot as plt
import torch

from examples.model import get_linear_model, get_hidden_size
from examples.util import (
    Dataset,
    LossFunction,
    evaluate_model,
    get_data_loaders,
    get_loss_function,
    plot_samples,
    save_model,
    train_one_epoch, add_to_results, get_device,
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
    quadratic_size = (input_length + (input_length ** 2 + input_length) // 2 + 1) * output_length
    hidden_size = get_hidden_size(
        lambda h: (input_length + 1) * h + (h + 1) * output_length,
        quadratic_size
    )
    device = get_device()
    model = get_linear_model(input_size=input_length, output_size=output_length, hidden_size=hidden_size, device=device)
    train_loader, test_loader = get_data_loaders(
        dataset,
        input_length,
        output_length,
        batch_size=20,
        positional_encoding=False,
        device=device,
    )

    learning_rate = 0.005
    loss_fn = get_loss_function(loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mean_mse:.5f}")
    duration = datetime.now() - start_time

    save_model(model, "linear", str(input_length), str(output_length))

    if plot:
        plt.plot(mses_train, label="train mse")
        plt.plot(mses_test_mean, label="test mse")
        plt.title("losses")
        plt.show()

        plot_samples(model, test_loader, use_labels=False, num_samples=2, title="test")
        plot_samples(model, train_loader, use_labels=False, num_samples=1, title="train")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    print(f"Duration: {duration}")

    add_to_results(
        "quadratic-model",
        input_length,
        output_length,
        epochs,
        batch_size,
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
    _input_length = 20
    _output_length = 10
    _epochs = 5
    _dataset: Dataset = "electricity-hourly"
    _loss_function: LossFunction = "mse"
    train_linear_model(
        _input_length,
        _output_length,
        _dataset,
        _loss_function,
        _epochs,
    )
