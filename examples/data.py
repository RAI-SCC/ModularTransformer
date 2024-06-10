import polars as pl
import torch
import math
from torch.utils.data import DataLoader
from typing import Literal


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, input_seq_len: int, output_seq_len: int, device: torch.device | None = None):
        self.x = x
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.device = device

    def __len__(self):
        return len(self.x) - (self.input_seq_len + self.output_seq_len)

    @staticmethod
    def _index_to_annotations(index: int):
        # 15 minute intervals -> 96 per day
        day = index // 96
        time = index % 96
        hour = time / 4  # float
        x = math.cos(hour * 2 * 3.1415 / 24)
        y = math.sin(hour * 2 * 3.1415 / 24)
        # return [day, time, hour, x, y]
        return [x, y]

    def __getitem__(self, index: int):
        sequence = self.x[index : index + self.input_seq_len]
        next_elements = self.x[
            index + self.input_seq_len : index + self.input_seq_len + self.output_seq_len
        ]
        assert sequence.shape[0] == self.input_seq_len
        assert next_elements.shape[0] == self.output_seq_len
        annotations = [self._index_to_annotations(i) for i in range(index, index + self.input_seq_len)]
        annotations = torch.Tensor(annotations).to(self.device)
        annotations_next_elements = [
            self._index_to_annotations(i)
            for i in range(index + self.input_seq_len, index + self.input_seq_len + self.output_seq_len)
        ]
        annotations_next_elements = torch.Tensor(annotations_next_elements).to(self.device)
        sequence = sequence.unsqueeze(-1)
        next_elements = next_elements.unsqueeze(-1)

        return (
            torch.cat([sequence, annotations], dim=1),
            torch.cat([next_elements, annotations_next_elements], dim=1)
        )


class TimeseriesDatasetMultivariate(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, dimension: int, input_seq_len: int, output_seq_len: int):
        self.x = x
        self.dimension = dimension
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return (len(self.x) - (self.input_seq_len + self.output_seq_len)) // self.dimension

    def __getitem__(self, index: int):
        sequence = self.x[index : index + self.input_seq_len * self.dimension]
        next_elements = self.x[
            index + self.input_seq_len : index + self.input_seq_len + self.output_seq_len
        ]
        next_elements_converted = next_elements.view(self.output_seq_len, self.dimension)
        assert sequence.shape[0] == self.input_seq_len
        assert next_elements.shape[0] == self.output_seq_len
        return (sequence, next_elements)


def get_data_electricity(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
    x_tensor = torch.from_numpy(x_np).to(device)
    x_len = len(x)
    return x_tensor[: int(x_len * 0.7)], x_tensor[int(x_len * 0.7):]


def get_data_electricity_hourly(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x_train, x_test = get_data_electricity(device=device)
    # skip every 4 values
    x_train = x_train[::4]
    x_test = x_test[::4]
    return x_train, x_test


def get_ett_time_series(i: Literal['h1', 'h2', 'm1', 'm2'] = 'h1', device: torch.device | None = None) -> torch.Tensor:
    schema = {
        'date': pl.Datetime,
        'HUFL': pl.Float64,
        'HULL': pl.Float64,
        'MUFL': pl.Float64,
        'MULL': pl.Float64,
        'LUFL': pl.Float64,
        'LULL': pl.Float64,
        'OT': pl.Float64,
    }
    data = pl.read_csv(
        f"data/ett-small/ETT{i}.csv",
        schema=schema,
    )
    x = data[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]]
    assert all(not x[col].is_null().any() for col in x.columns)
    x_np = x.to_numpy()
    x_normalized = (x_np - x_np.mean(axis=0)) / x_np.std(axis=0)
    x_tensor = torch.Tensor(x_normalized)
    return x_tensor

def get_data_ett(i: Literal['h1', 'h2', 'm1', 'm2'] = 'h1', device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x = get_ett_time_series(i)
    x_len = len(x)
    # only take a look at target value
    return x[: int(x_len * 0.8), -1], x[int(x_len * 0.8):, -1]




class NoPositionalEncoding(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0][:, 0].unsqueeze(-1), self.dataset[i][1][:, 0].unsqueeze(-1)


def get_loader(
    data: torch.Tensor,
    input_length: int,
    output_length: int,
    batch_size: int,
    shuffle: bool = True,
    positional_encoding: bool = True,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = TimeseriesDataset(data, input_length, output_length, device=device)

    if not positional_encoding:
        dataset = NoPositionalEncoding(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loader_overfit(
    data: torch.Tensor,
    input_length: int,
    output_length: int,
    batch_size: int,
) -> DataLoader:
    dataset = TimeseriesDataset(data[:input_length * 2 + 1], input_length, output_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    assert len(loader) == 1
    return loader


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
