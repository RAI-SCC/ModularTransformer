from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl
import torch

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
from modular_transformer.taylor import TaylorTransformer


def train_quadratic_transformer(
    input_length: int,
    output_length: int,
    dataset: Dataset,
    loss_function: LossFunction,
    epochs: int,
    batch_size: int = 20,
    plot: bool = True,
):
    device = get_device()
    model = TaylorTransformer(
        input_features=1,
        output_features=1,
        sequence_length=input_length,
        sequence_length_decoder=output_length,
        d_model=1,
        nhead=1,
        dim_feedforward=200,
        num_encoder_layers=1,
        num_decoder_layers=1,
        final_activation=None,
        device=device,
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
    print(optimizer)

    losses_train = []
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
            use_labels=True,
        )
        losses_train.append(avg_loss)

        mean_mse, variance_mse, mean_mae, variance_mae, mean_mape, variance_mape = evaluate_model(
            model,
            test_loader,
            use_labels=True,
        )
        mses_test_mean.append(mean_mse)
        mses_test_variance.append(variance_mse)
        maes_test_mean.append(mean_mae)
        maes_test_variance.append(variance_mae)
        mapes_test_mean.append(mean_mape)
        mapes_test_variance.append(variance_mape)

        print(f"Train {loss_function}: {avg_loss:.5f}\nTest MSE:  {mean_mse:.5f}")

    save_model(model, "taylor-transformer", dataset, input_length, output_length, loss_function)

    if plot:
        plt.plot(losses_train, label=f"train {loss_function}")
        plt.plot(mses_test_mean, label=f"test {loss_function}")
        plt.title("losses")
        plt.legend()
        plt.show()

        plt.plot(mses_test_variance, label="q test mse variance")
        plt.title("q mse variance")
        plt.legend()
        plt.show()

        plot_samples(model, test_loader, use_labels=True, num_samples=2, title="test")
        plot_samples(model, train_loader, use_labels=True, num_samples=1, title="train")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    duration = datetime.now() - start_time
    print(f"Duration: {duration}")

    add_to_results(
        "taylor-transformer",
        dataset,
        input_length,
        output_length,
        epochs,
        batch_size,
        loss_function,
        learning_rate,
        num_params,
        duration.total_seconds(),
        losses_train,
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
    train_quadratic_transformer(
        _input_length,
        _output_length,
        _dataset,
        _loss_function,
        _epochs,
    )
