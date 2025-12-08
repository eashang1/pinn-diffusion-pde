import torch
import torch.nn as nn
from typing import List


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def create_pinn_network(
    input_dim: int = 2,
    output_dim: int = 1,
    hidden_layers: List[int] = [64, 64, 64],
    device: torch.device = None
) -> FeedForwardNetwork:
    if device is None:
        device = get_device()
    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    return FeedForwardNetwork(layer_sizes).to(device)
