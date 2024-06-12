import matplotlib.pyplot as plt
import torch

from examples.model import (
    CubicModelLarge,
    QuadraticModel,
    get_hidden_size,
    get_linear_model,
    size_quadratic,
)
from examples.util import Dataset, LossFunction, get_data_loaders, load_model_weights
from modular_transformer import ClassicalTransformer
from modular_transformer.taylor import TaylorTransformer

input_length = 12
output_length = 24
dataset: Dataset = "electricity-hourly"
loss_function: LossFunction = "mse"

transformer = ClassicalTransformer(
    input_features=3,
    output_features=1,
    d_model=16,
    nhead=1,
    dim_feedforward=200,
    num_encoder_layers=1,
    num_decoder_layers=1,
    final_activation=None,
)
load_model_weights(transformer, "transformer", dataset, input_length, output_length, loss_function)
taylor_transformer = TaylorTransformer(
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
load_model_weights(
    taylor_transformer, "taylor-transformer", dataset, input_length, output_length, loss_function
)

quadratic_model = QuadraticModel(dim_in=input_length, hidden_layers=[10], dim_out=output_length)
load_model_weights(
    quadratic_model, "quadratic", dataset, input_length, output_length, loss_function
)
cubic_model = CubicModelLarge(dim_in=input_length, hidden_layers=[], dim_out=output_length)
load_model_weights(cubic_model, "cubic", dataset, input_length, output_length, loss_function)

quadratic_size = size_quadratic(input_length, output_length, [10])
hidden_size = get_hidden_size(
    lambda h: (input_length + 1) * h + (h + 1) * output_length, quadratic_size
)
linear_model = get_linear_model(
    input_size=input_length, output_size=output_length, hidden_size=hidden_size
)
load_model_weights(linear_model, "linear", dataset, input_length, output_length, loss_function)

# quadratic_model_solved = QuadraticModel(dim_in=input_length, hidden_layers=[], dim_out=output_length)
# quadratic_model_solved.load_state_dict(
#     torch.load(f"{model_path}/quadratic-solved-input{input_length}-output{output_length}.pt")
# )

bs = 20
_, test_loader = get_data_loaders(
    dataset,
    input_length,
    output_length,
    batch_size=20,
    positional_encoding=True,
)

torch.no_grad().__enter__()

for i, (inputs, labels) in enumerate(test_loader):
    taylor_transformer.eval()
    transformer.eval()
    quadratic_model.eval()

    assert inputs.shape == (bs, input_length, 3)
    assert labels.shape == (bs, output_length, 3)
    labels_masked = labels.clone()
    labels_masked[:, :, 0] = 0.0
    inputs_no_pos_encoding = inputs[:, :, 0].unsqueeze(-1)

    outputs_taylor_transformer = taylor_transformer(
        inputs_no_pos_encoding, torch.zeros((bs, output_length, 1))
    )
    outputs_transformer = transformer(inputs, labels_masked)
    outputs_quadratic_model = quadratic_model(inputs_no_pos_encoding)
    outputs_cubic_model = cubic_model(inputs_no_pos_encoding)
    outputs_linear_model = linear_model(inputs_no_pos_encoding.squeeze(-1))
    # outputs_quadratic_model_solved = quadratic_model_solved(inputs_no_pos_encoding)

    inputs_view = inputs[0, :, 0]
    outputs_taylor_transformer = outputs_taylor_transformer[0, :, 0]
    outputs_transformer = outputs_transformer[0, :, 0]
    outputs_quadratic_model = outputs_quadratic_model[0, :]
    # outputs_quadratic_model_solved = outputs_quadratic_model_solved[0, :]
    outputs_cubic_model = outputs_cubic_model[0, :]
    outputs_linear_model = outputs_linear_model[0, :]
    labels_view = labels[0, :, 0]
    mse_taylor_transformer = torch.nn.MSELoss()(outputs_taylor_transformer, labels)
    mse_transformer = torch.nn.MSELoss()(outputs_transformer, labels)
    mse_quadratic_model = torch.nn.MSELoss()(outputs_quadratic_model, labels)
    # mse_quadratic_model_solved = torch.nn.MSELoss()(outputs_quadratic_model_solved, labels)
    print("MSEs:")
    print(f"Taylor Transformer: {mse_taylor_transformer:.5f}")
    print(f"Transformer: {mse_transformer:.5f}")
    print(f"Quadratic Model: {mse_quadratic_model:.5f}")
    print(f"Cubic Model: {torch.nn.MSELoss()(outputs_cubic_model, labels):.5f}")
    print(f"Linear Model: {torch.nn.MSELoss()(outputs_linear_model, labels):.5f}")
    # print(f"Quadratic Model solved: {mse_quadratic_model_solved:.5f}")

    outputs_taylor_transformer = outputs_taylor_transformer.detach().numpy()
    outputs_transformer = outputs_transformer.detach().numpy()
    outputs_quadratic_model = outputs_quadratic_model.detach().numpy()
    # outputs_quadratic_model_solved = outputs_quadratic_model_solved.detach().numpy()
    outputs_cubic_model = outputs_cubic_model.detach().numpy()
    outputs_linear_model = outputs_linear_model.detach().numpy()
    labels_view = labels_view.detach().numpy()
    inputs_view = inputs_view.detach().numpy()

    plt.plot(inputs_view, label="inputs")
    plt.plot(range(input_length, input_length + output_length), labels_view, label="ground truth")
    # plot outputs and ground truth behind input sequence
    plt.plot(
        range(input_length, input_length + output_length),
        outputs_taylor_transformer,
        label="taylor transformer",
    )
    plt.plot(
        range(input_length, input_length + output_length), outputs_transformer, label="transformer"
    )

    # plt.title(f"mse: {mse_loss:.5f}")
    plt.legend()
    plt.xticks(range(0, input_length + output_length, 2))
    plt.savefig(
        f"examples/plots/predictions/transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.png"
    )
    plt.savefig(
        f"examples/plots/predictions/transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.pdf"
    )
    plt.savefig(
        f"examples/plots/predictions/transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.svg"
    )
    plt.show()

    plt.plot(inputs_view, label="inputs")
    plt.plot(range(input_length, input_length + output_length), labels_view, label="ground truth")
    plt.plot(
        range(input_length, input_length + output_length),
        outputs_linear_model,
        label="linear model",
    )
    plt.plot(
        range(input_length, input_length + output_length),
        outputs_quadratic_model,
        label="quadratic model",
    )
    # plt.plot(range(input_length, input_length + output_length), outputs_quadratic_model_solved, label="quadratic model solved")
    plt.plot(
        range(input_length, input_length + output_length), outputs_cubic_model, label="cubic model"
    )
    plt.legend()
    plt.xticks(range(0, input_length + output_length, 2))
    plt.savefig(
        f"examples/plots/predictions/non-transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.png"
    )
    plt.savefig(
        f"examples/plots/predictions/non-transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.pdf"
    )
    plt.savefig(
        f"examples/plots/predictions/non-transformers-{input_length}-{output_length}-{dataset}-{loss_function}-{i}.svg"
    )
    plt.show()
