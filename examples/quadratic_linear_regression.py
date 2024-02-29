from datetime import datetime

import torch

from examples.data import get_data_electricity_hourly, get_loader
from examples.model import QuadraticModel
from examples.util import save_model

input_length = 20
output_length = 20
batch_size = 200

model = QuadraticModel(dim_in=input_length, hidden_layers=[], dim_out=output_length)
data_train, data_test = get_data_electricity_hourly()

train_loader = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    positional_encoding=False
)
train_loader_all = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=1_000_000,  # use all data at once
    positional_encoding=False
)

test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    positional_encoding=False
)
test_loader_all = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=1_000_000,
    positional_encoding=False
)

assert len(train_loader_all) == 1
assert len(test_loader_all) == 1


def square_inputs(inputs):
    x = inputs.squeeze(-1)
    _batch_size, _dim_in = x.shape
    layer = model.lin0
    x_squared = torch.empty(_batch_size, (_dim_in ** 2 + _dim_in) // 2)
    for j in range(_batch_size):
        triu_indices = torch.triu_indices(_dim_in, _dim_in)
        assert triu_indices.shape[1] == x_squared.shape[1]
        x_squared[j] = torch.outer(x[j], x[j])[triu_indices[0], triu_indices[1]]
    x = torch.cat([x, x_squared], dim=-1)
    # x.shape = (batch_size, dim_in + dim_in ** 2)
    return x


def optimal_weights(inputs, labels):
    x_squared = square_inputs(inputs)

    x_sq_plus_one = torch.cat([torch.ones(x_squared.shape[0], 1), x_squared], dim=-1)
    weight_optimal_plus_one = torch.linalg.lstsq(x_sq_plus_one, labels[:, :, 0]).solution
    weight_optimal = weight_optimal_plus_one[1:]
    bias_optimal = weight_optimal_plus_one[0]
    return weight_optimal, bias_optimal


start_time = datetime.now()
# generate linear regression problem with quadratic features
inputs, labels = next(iter(train_loader_all))
vinputs, vlabels = next(iter(test_loader_all))

outputs = model(inputs)
voutputs = model(vinputs)

loss = torch.nn.functional.mse_loss(outputs, labels[:, :, 0])
vloss = torch.nn.functional.mse_loss(voutputs, vlabels[:, :, 0])

weight_optimal, bias_optimal = optimal_weights(inputs, labels)
state_dict = model.lin0.state_dict()
state_dict['weight'] = weight_optimal.T
state_dict['bias'] = bias_optimal
model.lin0.load_state_dict(state_dict)

outputs_optimal = model(inputs)
voutputs_optimal = model(vinputs)

loss_optimal = torch.nn.functional.mse_loss(outputs_optimal, labels[:, :, 0])
vloss_optimal = torch.nn.functional.mse_loss(voutputs_optimal, vlabels[:, :, 0])

duration = datetime.now() - start_time
print(f"Duration: {duration}")

print(f"Loss before optimal weights: {loss.item():.4f}")
print(f"Loss after optimal weights: {loss_optimal.item():.4f}")
print(f"Validation loss before optimal weights: {vloss.item():.4f}")
print(f"Validation loss after optimal weights: {vloss_optimal.item():.4f}")
save_model(model, f"quadratic-solved", str(input_length), str(output_length))
