import os

import matplotlib.pyplot as plt
import numpy as np

from examples.model import (
    CubicModelLarge,
    QuadraticModel,
    get_hidden_size,
    get_linear_model,
    size_quadratic,
)
from examples.util import Dataset, LossFunction, load_model_weights


def plot_weights(
    weights: np.array,
    name: str,
    dataset: Dataset,
    input_length: int,
    output_length: int,
    loss_function: LossFunction,
):
    plt.imshow(weights)
    plt.yticks([])
    plt.xlabel("input")
    plt.ylabel("output")
    # plt.title(name)
    plt.savefig(
        f"examples/plots/model-weights/{name}-{dataset}-{input_length}-{output_length}-{loss_function}.pdf"
    )
    plt.show()
    os.makedirs("examples/plots/model-weights", exist_ok=True)


if __name__ == "__main__":
    # model weights density matrix
    input_length = 12
    output_length = 24
    loss_function: LossFunction = "mse"
    dataset: Dataset = "electricity-hourly"

    quadratic_model = QuadraticModel(dim_in=input_length, hidden_layers=[10], dim_out=output_length)
    load_model_weights(
        quadratic_model, "quadratic", dataset, input_length, output_length, loss_function
    )
    weights_q_1 = quadratic_model.lin0.weight.detach().numpy().__abs__()
    plot_weights(
        weights_q_1, "quadratic-layer-1", dataset, input_length, output_length, loss_function
    )
    weights_q_2 = quadratic_model.lin1.weight.detach().numpy().__abs__()
    plot_weights(
        weights_q_2, "quadratic-layer-2", dataset, input_length, output_length, loss_function
    )

    cubic_model = CubicModelLarge(dim_in=input_length, hidden_layers=[], dim_out=output_length)
    load_model_weights(cubic_model, "cubic", dataset, input_length, output_length, loss_function)
    weights_c_1 = cubic_model.lin0.weight.detach().numpy().__abs__()[:, :180]
    plot_weights(weights_c_1, "cubic-layer-1", dataset, input_length, output_length, loss_function)

    _hidden_size = get_hidden_size(
        lambda h: (input_length + 1) * h + (h + 1) * output_length,
        size_quadratic(input_length, output_length, [10]),
    )
    linear_model = get_linear_model(
        input_size=input_length, output_size=output_length, hidden_size=_hidden_size
    )
    load_model_weights(linear_model, "linear", dataset, input_length, output_length, loss_function)
    weights_l_1 = linear_model._modules["0"].weight.detach().numpy().__abs__()
    plot_weights(weights_l_1, "linear-layer-1", dataset, input_length, output_length, loss_function)
    weights_l_2 = linear_model._modules["2"].weight.detach().numpy().__abs__()
    plot_weights(weights_l_2, "linear-layer-2", dataset, input_length, output_length, loss_function)
