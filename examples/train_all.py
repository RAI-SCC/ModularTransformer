from itertools import product

from examples.train import train_transformer
from examples.train_non_transformer import train_quadratic_model
from examples.train_non_transformer_linear import train_linear_model
from examples.train_quadratic_layer import train_quadratic_transformer


def main():
    epochs = 20
    for input_length, output_length, dataset, loss_function in product(
        [10, 20, 50],
        [10, 20, 50],
        ["electricity", "electricity-hourly"],
        ["mse", "mae", "mape"],
    ):
        print(f"Training transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}")
        train_transformer(
            input_length,
            output_length,
            dataset,
            loss_function,
            epochs,
            plot=False,
        )
        print(f"Training quadratic transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}")
        train_quadratic_transformer(
            input_length,
            output_length,
            dataset,
            loss_function,
            epochs,
            plot=False,
        )
        print(f"Training quadratic model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}")
        train_quadratic_model(
            input_length,
            output_length,
            dataset,
            loss_function,
            epochs,
            plot=False,
        )
        print(f"Training linear model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}")
        train_linear_model(
            input_length,
            output_length,
            dataset,
            loss_function,
            epochs,
            plot=False,
        )


if __name__ == "__main__":
    main()
