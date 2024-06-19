import math
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.data import get_data_electricity, get_loader
from modular_transformer.classical_mc import ClassicalMCDTransformer


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
    sample_num=5,
):
    running_loss = 0.0
    len_data = len(training_loader)
    check = 999

    for i, (inputs, labels) in enumerate(tqdm(training_loader)):
        optimizer.zero_grad()
        assert not inputs.isnan().any()
        # inputs = inputs.unsqueeze(-1)
        size_batch = inputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        # other = torch.range(0, inputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / inputs.shape[1]
        # assert inputs.shape == (size_batch, input_length, 1)
        labels_masked = labels.clone()
        labels_masked[:, :, 0] = 0.0
        outputs = []
        s = 0
        while s < sample_num:
            new_out = model(inputs, labels_masked)
            assert new_out.shape == (size_batch, output_length, 1)
            assert not new_out.isnan().any()
            outputs.append(new_out)
            s += 1

        outputs = torch.mean(torch.stack(outputs), dim=0)
        loss = loss_fn(outputs.squeeze(-1), labels[:, :, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == check:
            print(f"Running loss until batch {i}: {running_loss / (i + 1)}")

    epoch_loss = running_loss / len_data
    return epoch_loss


@torch.no_grad()
def plot_samples(model: ClassicalMCDTransformer, test_loader: DataLoader, sample_num=5):
    n = 0

    import matplotlib.pyplot as plt

    for vinputs, vlabels in test_loader:
        # vinputs = vinputs.unsqueeze(-1)

        size_batch = vinputs.shape[0]
        assert size_batch <= batch_size
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
        # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
        # assert vinputs.shape == (size_batch, input_length, 1)
        vlabels_masked = vlabels.clone()
        vlabels_masked[:, :, 0] = 0.0
        vout_list = []
        i = 0
        while i < sample_num:
            vout_list.append(model(vinputs, vlabels_masked))
            i += 1
        voutputs = torch.mean(torch.stack(vout_list), dim=0)
        voutput_std = torch.squeeze(torch.std(torch.stack(vout_list), dim=0))
        # voutputs = model(vinputs, vlabels_masked)
        v_inputs = vinputs[:, :, 0]
        v_outputs = voutputs[:, :, 0]
        v_labels = vlabels[:, :, 0]
        # print("vinputs")
        # print(vinputs)
        # print("voutputs")
        # print(voutputs)
        # print("vlabels")
        # print(vlabels)

        for i in range(v_labels.shape[0]):
            if n > 1:
                return
            plt.plot(v_inputs[i, :].numpy(), color="blue", label="inputs")
            # plot outputs and ground truth behind input sequence
            plt.plot(
                range(input_length, input_length + output_length),
                v_outputs[i, :].numpy(),
                color="orange",
                label="outputs",
            )
            plt.plot(
                range(input_length, input_length + output_length),
                v_labels[i, :].numpy(),
                color="green",
                label="ground truth",
            )
            plt.legend()
            plt.xticks(range(0, input_length + output_length, 2))
            plt.fill_between(
                range(input_length, input_length + output_length),
                (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=1)).numpy(),
                (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=-1)).numpy(),
                color="orange",
                alpha=0.1,
            )
            # plt.plot(range(input_length, input_length + output_length), (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=1)).numpy(), color='orange', alpha=.3, label="upper conf bound")
            # plt.plot(range(input_length, input_length + output_length), (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=-1)).numpy(), color='orange', alpha=.3, label="lower conf bound")
            plt.show()
            # time.sleep(5)
            n += 1


def evaluate_model(model, test_loader: DataLoader):
    running_vloss_mse = 0.0
    running_vloss_mae = 0.0
    model.eval()
    with torch.no_grad():
        for _, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata

            size_batch = vinputs.shape[0]
            assert size_batch <= batch_size
            # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1)
            # other = torch.range(0, vinputs.shape[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(size_batch, 1, 1) / vinputs.shape[1]
            # assert vinputs.shape == (size_batch, input_length, 1)
            # vlabels_masked = vlabels
            # vlabels_masked[:, :, 0] = 0.
            voutputs = model(vinputs, vlabels)
            vloss_mse = torch.nn.MSELoss()(voutputs.squeeze(-1), vlabels[:, :, 0])
            vloss_mae = torch.nn.L1Loss()(voutputs.squeeze(-1), vlabels[:, :, 0])
            running_vloss_mse += vloss_mse
            running_vloss_mae += vloss_mae

    mse = running_vloss_mse / len(test_loader)
    mae = running_vloss_mae / len(test_loader)
    return mse, mae


if __name__ == "__main__":
    input_length = 50
    output_length = 150

    model = ClassicalMCDTransformer(
        input_features=3,
        output_features=1,
        d_model=16,
        nhead=1,
        dim_feedforward=200,
        num_encoder_layers=1,
        num_decoder_layers=1,
        final_activation=None,
        layer_norm=True,
        dropout=0.0,
        gaussian=True,
        weight_drop=True,
        rate=0.5,
        std_dev=0.5,
        istrainablesigma=True,
    )

    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    epochs = 50
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

    learning_rate = 0.005
    # learning_rate = 0.001
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mses_train = []
    mses_test = []
    maes_test = []

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
        mses_train.append(avg_loss)

        mse_test, mae_test = evaluate_model(model, test_loader)
        mses_test.append(mse_test)
        maes_test.append(mae_test)

        # plot_samples(model, test_loader)

        print(f"Train MSE: {avg_loss:.5f}\nTest MSE:  {mse_test:.5f}")

    plt.plot([math.log(x) for x in mses_train], label="train mse")
    plt.plot([math.log(x) for x in mses_test], label="test mse")
    plt.title("losses (log scale)")
    plt.show()

    plt.plot(mses_train, label="train mse")
    plt.plot(mses_test, label="test mse")
    plt.title("losses")
    plt.show()

    plot_samples(model, test_loader)
