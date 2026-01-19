"""Base neural network components.

Contains fundamental building blocks used across different architectures.
"""

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fourier features for time encoding."""

    def __init__(self, output_size=64, learnable=False):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable

        if self.learnable:
            self.w = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)
        else:
            half_dim = output_size // 2
            f = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f)
            self.register_buffer('f', f)

    def forward(self, x):
        """Apply Fourier features to input."""
        if self.learnable:
            f = 2 * torch.pi * x @ self.w.T
        else:
            f = x * self.f
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class Identity(nn.Module):
    """Identity module."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=(512, 512, 512, 512),
        activations=nn.GELU,
        activate_final=False,
        layer_norm=False,
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activations())
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim))

        if activate_final:
            layers.append(activations())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)
