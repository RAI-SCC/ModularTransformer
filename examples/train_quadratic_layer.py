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
    train_one_epoch,
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
    )

    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    train_loader, test_loader = get_data_loaders(
        dataset,
        input_length,
        output_length,
        batch_size=20,
        positional_encoding=False,
    )

    learning_rate = 0.0005
    loss_fn = get_loss_function(loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mses_train = []
    mses_test_mean = []
    mses_test_variance = []
    maes_test_mean = []
    maes_test_variance = []

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
        mses_train.append(avg_loss)

        mean_mse, variance_mse, mean_mae, variance_mae = evaluate_model(
            model,
            test_loader,
            use_labels=True,
        )
        mses_test_mean.append(mean_mse)
        mses_test_variance.append(variance_mse)
        maes_test_mean.append(mean_mae)
        maes_test_variance.append(variance_mae)

        print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mean_mse:.5f}")

    save_model(model, "taylor-transformer", str(input_length), str(output_length))

    if plot:
        plt.plot(mses_train, label="q train mse")
        plt.plot(mses_test_mean, label="q test mse")
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

    metadata = pl.DataFrame(
        {
            "model_type": ["taylor-transformer"],
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
    metadata.write_parquet("examples/model-weights/taylor-transformer-metadata.parquet")


if __name__ == "__main__":
    _input_length = 20
    _output_length = 10
    _epochs = 5
    _dataset: Dataset = "electricity-hourly"
    _loss_function: LossFunction = "mse"
    train_quadratic_transformer(
        _input_length,
        _output_length,
        _dataset,
        _loss_function,
        _epochs,
    )
