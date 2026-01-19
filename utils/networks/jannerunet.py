"""JannerUNet architecture for flow matching.

Ported from much-ado-about-noising repository.
U-Net architecture based on Janner et al.'s diffusion policy implementation.
"""

import numpy as np
import torch
import torch.nn as nn

from utils.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from utils.network_utils import GroupNorm1d


def get_norm(dim: int, norm_type: str = "groupnorm"):
    """Get normalization layer."""
    if norm_type == "groupnorm":
        return GroupNorm1d(dim, 8, 4)
    elif norm_type == "layernorm":
        return LayerNorm(dim)
    else:
        return nn.Identity()


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


class LayerNorm(nn.Module):
    """Layer normalization for 1D inputs."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ResidualBlock(nn.Module):
    """Residual block with conditioning."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        emb_dim: int,
        kernel_size: int = 3,
        norm_type: str = "groupnorm",
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.emb_mlp = nn.Sequential(nn.Mish(), nn.Linear(emb_dim, out_dim))
        self.residual_conv = (
            nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, emb):
        out = self.conv1(x) + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        return out + self.residual_conv(x)


class LinearAttention(nn.Module):
    """Linear attention mechanism."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)

        # Reshape for multi-head attention
        q, k, v = qkv
        b, c, d = q.shape
        q = q.view(b, self.heads, c // self.heads, d)
        k = k.view(b, self.heads, c // self.heads, d)
        v = v.view(b, self.heads, c // self.heads, d)

        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)

        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, -1, d)
        out = self.to_out(out)
        return out + x


class JannerUNet(nn.Module):
    """JannerUNet architecture for action prediction."""
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        model_dim: int = 32,
        emb_dim: int = 32,
        kernel_size: int = 3,
        dim_mult: list[int] | None = None,
        norm_type: str = "groupnorm",
        attention: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        super().__init__()

        # Default dim_mult if not provided
        if dim_mult is None:
            dim_mult = [1, 2, 2, 2]

        self.Ta = Ta
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model_dim = model_dim
        self.emb_dim = emb_dim
        self.disable_time_embedding = disable_time_embedding

        # Use act_dim as in_dim for the U-Net
        in_dim = act_dim
        dims = [in_dim] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:], strict=False))

        # Time embedding mappings for s and t
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                emb_dim // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                emb_dim // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Map combined embeddings to model dimension
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim * 2, model_dim * 4),
            nn.Mish(),
            nn.Linear(model_dim * 4, model_dim),
        )

        # Condition encoder
        self.cond_encoder = nn.Linear(obs_dim * To, emb_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            dim_in, dim_out, model_dim, kernel_size, norm_type
                        ),
                        ResidualBlock(
                            dim_out, dim_out, model_dim, kernel_size, norm_type
                        ),
                        LinearAttention(dim_out) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, norm_type
        )
        self.mid_attn = LinearAttention(mid_dim) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, norm_type
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            dim_out * 2, dim_in, model_dim, kernel_size, norm_type
                        ),
                        ResidualBlock(
                            dim_in, dim_in, model_dim, kernel_size, norm_type
                        ),
                        LinearAttention(dim_in) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 5, padding=2),
            get_norm(model_dim, norm_type),
            nn.Mish(),
            nn.Conv1d(model_dim, in_dim, 1),
        )

        # Scalar output head
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mid_dim, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

        # Zero-out scalar head
        nn.init.constant_(self.scalar_head[-1].weight, 0)
        nn.init.constant_(self.scalar_head[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            x: (batch, Ta, act_dim) action sequence
            s: (batch,) source time parameter
            t: (batch,) target time parameter
            condition: (batch, To, obs_dim) observation condition

        Returns:
            y: (batch, Ta, act_dim) predicted action sequence
            scalar: (batch, 1) predicted scalar value
        """
        # Check Ta dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        batch_size = x.shape[0]
        device = x.device

        x = x.permute(0, 2, 1)  # (batch, act_dim, Ta)

        # Process time embeddings
        if not self.disable_time_embedding:
            emb_s = self.map_s(s)
            emb_t = self.map_t(t)
            time_emb = torch.cat([emb_s, emb_t], dim=-1)
        else:
            time_emb = torch.zeros((batch_size, self.emb_dim), device=device)

        # Process condition
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)
            cond_emb = self.cond_encoder(cond_flat)
        else:
            cond_emb = torch.zeros((batch_size, self.emb_dim), device=device)

        # Combine time and condition embeddings
        combined_emb = torch.cat([time_emb, cond_emb], dim=-1)
        emb = self.map_emb(combined_emb)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # Get scalar output from bottleneck
        scalar_out = self.scalar_head(x)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # (batch, Ta, act_dim)

        return x, scalar_out
