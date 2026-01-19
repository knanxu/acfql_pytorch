"""Network utility functions and helper modules.

Includes GroupNorm1d and other utilities needed for network architectures.
"""

import torch
import torch.nn as nn


class GroupNorm1d(nn.Module):
    """GroupNorm for 1D inputs (used in UNet architectures)."""

    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) tensor
        Returns:
            Normalized tensor
        """
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)


class Downsample1d(nn.Module):
    """Downsample 1D signal by factor of 2."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsample 1D signal by factor of 2."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


def unsqueeze_expand_at(x, size, dim):
    """Unsqueeze and expand tensor at specified dimension."""
    x = x.unsqueeze(dim)
    expand_shape = [-1] * x.ndim
    expand_shape[dim] = size
    return x.expand(*expand_shape)


def flatten(x, begin_axis=-2):
    """Flatten tensor from begin_axis onwards."""
    if begin_axis < 0:
        begin_axis = x.ndim + begin_axis
    return x.reshape(*x.shape[:begin_axis], -1)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """Reshape dimensions between begin_axis and end_axis to target_dims."""
    if begin_axis < 0:
        begin_axis = x.ndim + begin_axis
    if end_axis < 0:
        end_axis = x.ndim + end_axis

    new_shape = list(x.shape[:begin_axis]) + list(target_dims) + list(x.shape[end_axis + 1:])
    return x.reshape(*new_shape)


def join_dimensions(x, begin_axis, end_axis):
    """Join dimensions between begin_axis and end_axis."""
    if begin_axis < 0:
        begin_axis = x.ndim + begin_axis
    if end_axis < 0:
        end_axis = x.ndim + end_axis

    # Calculate the product of dimensions to join
    join_size = 1
    for i in range(begin_axis, end_axis + 1):
        join_size *= x.shape[i]

    new_shape = list(x.shape[:begin_axis]) + [join_size] + list(x.shape[end_axis + 1:])
    return x.reshape(*new_shape)
