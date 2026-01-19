"""Timestep embedding modules for flow matching.

Ported from much-ado-about-noising repository.
Author: Chaoyi Pan (original), adapted for acfql_pytorch
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class FourierEmbedding(nn.Module):
    """Random Fourier features for timestep embedding."""

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.dim = dim
        # Register as buffer so it moves with the model
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        x = x[:, None] * self.freqs[None, :] * 2 * math.pi
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.embedding = nn.Embedding(max_positions, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) tensor of timesteps in [0, 1]
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        # Convert continuous timesteps to discrete positions
        positions = (x * (self.max_positions - 1)).long().clamp(0, self.max_positions - 1)
        return self.embedding(positions)


# Registry of supported timestep embeddings
SUPPORTED_TIMESTEP_EMBEDDING = {
    "sinusoidal": SinusoidalEmbedding,
    "fourier": FourierEmbedding,
    "positional": SinusoidalEmbedding,  # Default to sinusoidal for "positional"
}


def get_timestep_embedding(emb_type: str, dim: int, **kwargs) -> nn.Module:
    """Factory function to create timestep embeddings.

    Args:
        emb_type: Type of embedding ("sinusoidal", "fourier", "positional")
        dim: Embedding dimension
        **kwargs: Additional arguments for the embedding

    Returns:
        Timestep embedding module
    """
    if emb_type not in SUPPORTED_TIMESTEP_EMBEDDING:
        raise ValueError(
            f"Unknown embedding type: {emb_type}. "
            f"Supported types: {list(SUPPORTED_TIMESTEP_EMBEDDING.keys())}"
        )
    return SUPPORTED_TIMESTEP_EMBEDDING[emb_type](dim, **kwargs)
