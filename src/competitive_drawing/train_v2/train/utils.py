import torch


def accuracy_score(labels: torch.Tensor, outputs: torch.Tensor) -> float:
    return (torch.sum(labels == outputs) / len(outputs)).item()
