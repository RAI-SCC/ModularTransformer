import os
from itertools import product

from examples.train import train_transformer
from examples.train_non_transformer import train_quadratic_model
from examples.train_non_transformer_cubic import train_cubic_model
from examples.train_non_transformer_double import train_quadratic_model_double
from examples.train_non_transformer_linear import train_linear_model
from examples.train_quadratic_layer import train_quadratic_transformer
from examples.util import PATH_RESULTS, model_path


def main():  # noqa
    breakages = []
    epochs = 100
    lengths = [(12, 24), (12, 6), (24, 24)]
    datasets = ["electricity-hourly", "etth2"]
    for (input_length, output_length), dataset, loss_function in product(
        lengths, datasets, ["mse", "mae"]
    ):
        print(
            f"Training for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
        )
        # stop if `shutdown` file exists
        if os.path.exists("shutdown"):
            print("PREMATURELY STOPPING TRAINING")
            break

        # skip if sample already exists
        # use linear since it was the last in the list
        if os.path.exists(
            model_path("linear", dataset, input_length, output_length, loss_function)
        ):
            print(f"Skipping {input_length}, {output_length}, {dataset}, {loss_function}")
            continue
        try:
            print(
                f"Training transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_transformer(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append((input_length, output_length, dataset, loss_function, "transformer"))
        try:
            print(
                f"Training quadratic transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_quadratic_transformer(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in quadratic transformer for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append(
                (input_length, output_length, dataset, loss_function, "quadratic transformer")
            )
        try:
            print(
                f"Training quadratic model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_quadratic_model(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in quadratic model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append((input_length, output_length, dataset, loss_function, "quadratic"))
        try:
            print(
                f"Training quadratic model double for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_quadratic_model_double(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in quadratic model double for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append(
                (input_length, output_length, dataset, loss_function, "quadratic double")
            )
        try:
            print(
                f"Training cubic model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_cubic_model(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in cubic model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append((input_length, output_length, dataset, loss_function, "cubic"))
        try:
            print(
                f"Training linear model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            train_linear_model(
                input_length,
                output_length,
                dataset,
                loss_function,
                epochs,
                plot=False,
            )
        except Exception as e:
            print(
                f"Error in linear model for input_length={input_length}, output_length={output_length}, dataset={dataset}, loss_function={loss_function}"
            )
            print(e)
            breakages.append((input_length, output_length, dataset, loss_function, "linear"))
    print("Breakages:")
    print(breakages)


if __name__ == "__main__":
    os.makedirs(PATH_RESULTS, exist_ok=True)
    main()
