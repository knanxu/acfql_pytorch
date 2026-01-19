"""ChiUNet architecture for flow matching.

Ported from much-ado-about-noising repository.
U-Net architecture from Diffusion Policy by Chi et al.
"""

from typing import Final

import numpy as np
import torch
import torch.nn as nn

from utils.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from utils.network_utils import GroupNorm1d, Downsample1d, Upsample1d


class ChiResidualBlock(nn.Module):
    """Residual block with conditioning."""

    out_dim: Final[int]
    cond_predict_scale: Final[bool]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        emb_dim: int,
        kernel_size: int = 3,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4),
            nn.GELU(),
        )

        cond_dim = 2 * out_dim if cond_predict_scale else out_dim
        self.cond_predict_scale = cond_predict_scale
        self.out_dim = out_dim
        self.cond_encoder = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, cond_dim))

        self.residual_conv = (
            nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, emb):
        out = self.conv1(x)
        embed = self.cond_encoder(emb)
        if self.cond_predict_scale:
            embed = embed.view(embed.shape[0], 2, -1, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed.unsqueeze(-1)
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class ChiUNet(nn.Module):
    """ChiUNet architecture for action prediction."""

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        model_dim: int = 256,
        emb_dim: int = 256,
        kernel_size: int = 5,
        cond_predict_scale: bool = True,
        obs_as_global_cond: bool = True,
        dim_mult: list[int] | None = None,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        super().__init__()

        # Default dim_mult if not provided
        if dim_mult is None:
            dim_mult = [1, 2, 2]

        self.Ta = Ta
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_as_global_cond = obs_as_global_cond
        self.model_dim = model_dim
        self.emb_dim = emb_dim
        self.disable_time_embedding = disable_time_embedding
        original_emb_dim = emb_dim

        dims = [act_dim] + [model_dim * m for m in np.cumprod(dim_mult)]

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), nn.GELU(), nn.Linear(emb_dim * 4, emb_dim)
        )

        if obs_as_global_cond:
            self.global_cond_encoder = nn.Linear(To * obs_dim, emb_dim)
            emb_dim = emb_dim * 2  # cat obs and emb
            self.local_cond_encoder = None
        else:
            self.global_cond_encoder = None
            emb_dim = emb_dim
            self.local_cond_encoder = nn.ModuleList(
                [
                    ChiResidualBlock(
                        obs_dim, model_dim, emb_dim, kernel_size, cond_predict_scale
                    ),
                    ChiResidualBlock(
                        obs_dim, model_dim, emb_dim, kernel_size, cond_predict_scale
                    ),
                    Downsample1d(model_dim),
                ]
            )

        in_out = list(zip(dims[:-1], dims[1:], strict=False))
        mid_dim = dims[-1]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ChiResidualBlock(
                            dim_in, dim_out, emb_dim, kernel_size, cond_predict_scale
                        ),
                        ChiResidualBlock(
                            dim_out, dim_out, emb_dim, kernel_size, cond_predict_scale
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mids = nn.ModuleList(
            [
                ChiResidualBlock(
                    mid_dim, mid_dim, emb_dim, kernel_size, cond_predict_scale
                ),
                ChiResidualBlock(
                    mid_dim, mid_dim, emb_dim, kernel_size, cond_predict_scale
                ),
            ]
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ChiResidualBlock(
                            dim_out * 2,
                            dim_in,
                            emb_dim,
                            kernel_size,
                            cond_predict_scale,
                        ),
                        ChiResidualBlock(
                            dim_in, dim_in, emb_dim, kernel_size, cond_predict_scale
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(model_dim, 8, 4),
            nn.GELU(),
            nn.Conv1d(model_dim, act_dim, 1),
        )

        # Time embedding mappings for s and t
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                original_emb_dim // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                original_emb_dim // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Scalar output head
        mid_channels = model_dim * np.prod(dim_mult)
        self.scalar_output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mid_channels, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Zero-out scalar head
        nn.init.constant_(self.scalar_output_head[-1].weight, 0)
        nn.init.constant_(self.scalar_output_head[-1].bias, 0)

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
        assert x.shape[1] & (x.shape[1] - 1) == 0, (
            f"Ta dimension must be 2^n, current shape: {x.shape}"
        )

        x = x.permute(0, 2, 1)  # (B, act_dim, Ta)

        if not self.disable_time_embedding:
            emb_s = self.map_s(s)
            emb_t = self.map_t(t)
            emb = torch.cat([emb_s, emb_t], dim=-1)
        else:
            emb = torch.zeros((x.shape[0], self.emb_dim), device=x.device)
        emb = self.map_emb(emb)

        # If obs_as_global_cond, concatenate obs and emb
        if self.obs_as_global_cond:
            if condition is not None:
                condition_emb = self.global_cond_encoder(torch.flatten(condition, 1))
                emb = torch.cat([emb, condition_emb], dim=-1)
            else:
                emb = torch.cat([emb, torch.zeros_like(emb[:, : self.emb_dim])], dim=-1)
            h_local = None
        else:
            condition = (
                condition.permute(0, 2, 1)
                if condition is not None
                else torch.zeros(
                    (x.shape[0], self.obs_dim, x.shape[-1]), device=x.device
                )
            )
            assert x.shape[-1] == condition.shape[-1]
            if self.local_cond_encoder is not None:
                resnet1, resnet2, downsample = self.local_cond_encoder
                h_local = [resnet1(condition, emb), downsample(resnet2(condition, emb))]
            else:
                h_local = None

        h = []

        for idx, (resnet1, resnet2, downsample) in enumerate(self.downs):
            x = resnet1(x, emb)
            if idx == 0 and h_local is not None:
                x = x + h_local[0]
            x = resnet2(x, emb)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mids:
            x = mid_module(x, emb)

        # Get scalar output after midmodule
        scalar_out = self.scalar_output_head(x)

        for idx, (resnet1, resnet2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, emb)
            if idx == (len(self.ups) - 1) and h_local is not None:
                x = x + h_local[1]
            x = resnet2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # (B, Ta, act_dim)

        return x, scalar_out
