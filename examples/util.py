import torch


def mean_variance(l: list[float]) -> tuple[float, float]:
    mean = sum(l) / len(l)
    variance = sum((x - mean) ** 2 for x in l) / len(l)
    return mean, variance


def moving_average(x, w):
    return [sum(x[i:i + w]) / w for i in range(len(x) - w + 1)]


def save_model(model, name: str, input_length: str, output_length: str):
    path = f"examples/model-weights/{name}-input{input_length}-output{output_length}.pt"
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)
