import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from examples.data import get_data_electricity, get_loader
from modular_transformer.classical import ClassicalTransformer


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
):
    running_loss = 0.0
    len_data = len(training_loader)

    for inputs, labels in training_loader:
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        outputs = model(inputs, torch.zeros_like(inputs))
        assert not outputs.isnan().any()

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len_data


def evaluate_model(model, test_loader: DataLoader):
    running_vloss_mse = 0.0
    running_vloss_mae = 0.0
    model.eval()
    with torch.no_grad():
        for _, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs, torch.zeros_like(vinputs))
            vloss_mse = torch.nn.MSELoss()(voutputs, vlabels)
            vloss_mae = torch.nn.L1Loss()(voutputs, vlabels)
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
    # vinputs, vlabels = next(iter(test_loader))
        voutputs = model(vinputs, torch.zeros_like(vinputs))
        print("vinputs")
        print(vinputs)
        print("voutputs")
        print(voutputs)
        print("vlabels")
        print(vlabels)

        for i in range(vlabels.shape[0]):
            if n > 5:
                return
            plt.plot(vlabels[i, :].numpy(), label="ground truth")
            plt.plot(voutputs[i, :].numpy(), label="outputs")
            plt.legend()
            plt.show()
            time.sleep(5)
            n += 1


input_length = 4
output_length = 10

model = ClassicalTransformer(
    input_features=input_length,
    output_features=output_length,
    d_model=10,
    nhead=1,
    dim_feedforward=5,
    num_encoder_layers=1,
    num_decoder_layers=1,
)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

epochs = 20

data_train, data_test = get_data_electricity()
train_loader = get_loader(
    data_train,
    input_length=input_length,
    output_length=output_length,
    batch_size=20,
)
test_loader = get_loader(
    data_test,
    input_length=input_length,
    output_length=output_length,
    batch_size=20,
    shuffle=False,
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.3)

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

    mse_test, mae_test = evaluate_model(model, test_loader)

    plot_samples(model, test_loader)

    print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mse_test:.5f}")
