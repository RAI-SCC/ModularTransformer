import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from examples.data import get_data_electricity, get_loader, get_loader_overfit
from modular_transformer.classical import ClassicalTransformer

from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
):
    running_loss = 0.0
    len_data = len(training_loader)

    for inputs, labels in tqdm(training_loader):
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        inputs = inputs.unsqueeze(-1)
        size_batch = inputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / inputs.shape[1]
        assert inputs.shape == (size_batch, input_length, 1)
        outputs = model(inputs, other)
        assert not outputs.isnan().any()

        loss = loss_fn(outputs.squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len_data
    print(f"{epoch_loss=}")
    return epoch_loss


def evaluate_model(model, test_loader: DataLoader):
    running_vloss_mse = 0.0
    running_vloss_mae = 0.0
    model.eval()
    with torch.no_grad():
        for _, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.unsqueeze(-1)

            size_batch = vinputs.shape[0]
            assert size_batch <= batch_size
            # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
            other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
            assert vinputs.shape == (size_batch, input_length, 1)
            voutputs = model(vinputs, other)
            vloss_mse = torch.nn.MSELoss()(voutputs.squeeze(-1), vlabels)
            vloss_mae = torch.nn.L1Loss()(voutputs.squeeze(-1), vlabels)
            running_vloss_mse += vloss_mse
            running_vloss_mae += vloss_mae

    mse = running_vloss_mse / len(test_loader)
    mae = running_vloss_mae / len(test_loader)
    return mse, mae


@torch.no_grad()
def plot_samples(model: ClassicalTransformer, test_loader: DataLoader):
    n = 0
    import matplotlib.pyplot as plt
    for vinputs, vlabels in test_loader:
        vinputs = vinputs.unsqueeze(-1)

        size_batch = vinputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
        assert vinputs.shape == (size_batch, input_length, 1)
        voutputs = model(vinputs, other)
        vinputs = vinputs.squeeze(-1)
        voutputs = voutputs.squeeze(-1)
        # print("vinputs")
        # print(vinputs)
        # print("voutputs")
        # print(voutputs)
        # print("vlabels")
        # print(vlabels)

        for i in range(vlabels.shape[0]):
            if n > 3:
                return
            plt.plot(vinputs[i, :].numpy(), label="inputs")
            # plot outputs and ground truth behind input sequence
            plt.plot(range(input_length, input_length + output_length), voutputs[i, :].numpy(), label="outputs")
            plt.plot(range(input_length, input_length + output_length), vlabels[i, :].numpy(), label="ground truth")
            plt.legend()
            plt.show()
            # time.sleep(5)
            n += 1


input_length = 10
output_length = 10

model = ClassicalTransformer(
    input_features=1,
    output_features=1,
    d_model=10,
    nhead=1,
    dim_feedforward=10,
    num_encoder_layers=4,
    num_decoder_layers=1,
    final_activation=None,
)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

epochs = 10
batch_size = 20

data_train, data_test = get_data_electricity()
train_loader = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
)
test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=batch_size,
    shuffle=False,
)
# train_loader = get_loader_overfit(
#     data_train,
#     input_length=input_length,
#     output_length=output_length,
#     batch_size=batch_size,
# )
# test_loader = train_loader

learning_rate = 0.01
momentum = 0.3
# learning_rate = 0.001
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

losses = []

start_time = datetime.now()
for epoch in range(epochs):
    print(f"EPOCH {epoch}:")

    model.train(True)
    avg_loss = train_one_epoch(
        model,
        loss_fn,
        optimizer,
        train_loader,
    )
    losses.append(avg_loss)

    mse_test, mae_test = evaluate_model(model, test_loader)

    # plot_samples(model, test_loader)

    print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mse_test:.5f}")

import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("losses")
plt.show()

plot_samples(model, test_loader)
