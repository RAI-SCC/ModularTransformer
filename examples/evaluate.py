import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from examples.data import get_data_electricity_hourly, get_loader
from examples.model import QuadraticModel
from modular_transformer import ClassicalTransformer
from modular_transformer.taylor import TaylorTransformer

input_length = 20
output_length = 10

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
taylor_transformer.load_state_dict(
    torch.load(f"examples/model-weights/taylor-transformer-input{input_length}-output{output_length}.pt")
)
quadratic_model = QuadraticModel(dim_in=input_length, hidden_layers=[], dim_out=output_length)
quadratic_model.load_state_dict(
    torch.load(f"examples/model-weights/quadratic-input{input_length}-output{output_length}.pt")
)
quadratic_model_solved = QuadraticModel(dim_in=input_length, hidden_layers=[], dim_out=output_length)
quadratic_model_solved.load_state_dict(
    torch.load(f"examples/model-weights/quadratic-solved-input{input_length}-output{output_length}.pt")
)
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
transformer.load_state_dict(
    torch.load(f"examples/model-weights/transformer-input{input_length}-output{output_length}.pt")
)

hourly = True
if hourly:
    data_train, data_test = get_data_electricity_hourly()
else:
    data_train, data_test = get_data_electricity_hourly()

bs = 20
test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=bs,
    shuffle=True,
    positional_encoding=True
)

torch.no_grad().__enter__()

for i, (inputs, labels) in enumerate(test_loader):
    taylor_transformer.eval()
    transformer.eval()
    quadratic_model.eval()
    if i > 5:
        break

    assert inputs.shape == (bs, input_length, 3)
    assert labels.shape == (bs, output_length, 3)
    labels_masked = labels.clone()
    labels_masked[:, :, 0] = 0.
    inputs_no_pos_encoding = inputs[:, :, 0].unsqueeze(-1)

    outputs_taylor_transformer = taylor_transformer(inputs_no_pos_encoding, torch.zeros((bs, output_length, 1)))
    outputs_transformer = transformer(inputs, labels_masked)
    outputs_quadratic_model = quadratic_model(inputs_no_pos_encoding)
    outputs_quadratic_model_solved = quadratic_model_solved(inputs_no_pos_encoding)

    inputs = inputs[0, :, 0]
    outputs_taylor_transformer = outputs_taylor_transformer[0, :, 0]
    outputs_transformer = outputs_transformer[0, :, 0]
    outputs_quadratic_model = outputs_quadratic_model[0, :]
    outputs_quadratic_model_solved = outputs_quadratic_model_solved[0, :]
    labels = labels[0, :, 0]
    mse_taylor_transformer = torch.nn.MSELoss()(outputs_taylor_transformer, labels)
    mse_transformer = torch.nn.MSELoss()(outputs_transformer, labels)
    mse_quadratic_model = torch.nn.MSELoss()(outputs_quadratic_model, labels)
    mse_quadratic_model_solved = torch.nn.MSELoss()(outputs_quadratic_model_solved, labels)
    print("MSEs:")
    print(f"Taylor Transformer: {mse_taylor_transformer:.5f}")
    print(f"Transformer: {mse_transformer:.5f}")
    print(f"Quadratic Model: {mse_quadratic_model:.5f}")
    print(f"Quadratic Model solved: {mse_quadratic_model_solved:.5f}")

    outputs_taylor_transformer = outputs_taylor_transformer.detach().numpy()
    outputs_transformer = outputs_transformer.detach().numpy()
    outputs_quadratic_model = outputs_quadratic_model.detach().numpy()
    outputs_quadratic_model_solved = outputs_quadratic_model_solved.detach().numpy()
    labels = labels.detach().numpy()
    inputs = inputs.detach().numpy()

    plt.plot(inputs, label="inputs")
    # plot outputs and ground truth behind input sequence
    plt.plot(range(input_length, input_length + output_length), outputs_taylor_transformer, label="taylor transformer")
    plt.plot(range(input_length, input_length + output_length), outputs_transformer, label="transformer")
    plt.plot(range(input_length, input_length + output_length), outputs_quadratic_model, label="quadratic model")
    plt.plot(range(input_length, input_length + output_length), outputs_quadratic_model_solved, label="quadratic model solved")
    plt.plot(range(input_length, input_length + output_length), labels, label="ground truth")
    # plt.title(f"mse: {mse_loss:.5f}")
    plt.legend()
    plt.xticks(range(0, input_length + output_length, 2))
    plt.show()


import polars as pl
metadata_taylor_transformer = pl.read_parquet("examples/model-weights/taylor-transformer-metadata.parquet")
metadata_transformer = pl.read_parquet("examples/model-weights/transformer-metadata.parquet")
metadata_quadratic = pl.read_parquet("examples/model-weights/quadratic-metadata.parquet")
