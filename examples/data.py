import polars as pl
import torch
from torch.utils.data import DataLoader


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, input_seq_len: int, output_seq_len: int):
        self.x = x
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return len(self.x) - (self.input_seq_len + self.output_seq_len)

    def __getitem__(self, index: int):
        sequence = self.x[index : index + self.input_seq_len]
        next_elements = self.x[
            index + self.input_seq_len : index + self.input_seq_len + self.output_seq_len
        ]
        assert sequence.shape[0] == self.input_seq_len
        assert next_elements.shape[0] == self.output_seq_len
        return (sequence, next_elements)


class TimeseriesDatasetMultivariate(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, dimension: int, input_seq_len: int, output_seq_len: int):
        self.x = x
        self.dimension = dimension
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return (len(self.x) - (self.input_seq_len + self.output_seq_len)) / self.dimension

    def __getitem__(self, index: int):
        sequence = self.x[index : index + self.input_seq_len * self.dimension]
        next_elements = self.x[
            index + self.input_seq_len : index + self.input_seq_len + self.output_seq_len
        ]
        assert sequence.shape[0] == self.input_seq_len
        assert next_elements.shape[0] == self.output_seq_len
        return (sequence, next_elements)


def get_data_electricity() -> tuple[torch.Tensor, torch.Tensor]:
    data = pl.read_csv(
        "./data/western-europe-power-consumption/de.csv",
        dtypes={"start": pl.Datetime, "end": pl.Datetime, "load": pl.Float32},
    )
    x = data["load"]
    # replace all nan (7 values) with neighboring values
    print(f"{x.is_null().sum()} null values in data")
    x = x.fill_null(strategy="backward")
    normalized_x = (x - x.mean()) / x.std()
    x_np = normalized_x.to_numpy()
    x_tensor = torch.Tensor(x_np)
    x_len = len(x)
    return x_tensor[: int(x_len * 0.8)], x_tensor[int(x_len * 0.8) :]


def get_loader(
    data: torch.Tensor,
    input_length: int,
    output_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TimeseriesDataset(data, input_length, output_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loader_overfit(
    data: torch.Tensor,
    input_length: int,
    output_length: int,
    batch_size: int,
) -> DataLoader:
    dataset = TimeseriesDataset(data[:input_length * 2 + 1], input_length, output_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_loader_multivariate(
    data: torch.Tensor,
    dimension: int,
    input_length: int,
    output_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TimeseriesDatasetMultivariate(data, dimension, input_length, output_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
