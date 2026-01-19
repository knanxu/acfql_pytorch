"""MLP-based flow matching networks.

Contains ActorVectorField and MeanActorVectorField for flow matching.
"""

import torch
import torch.nn as nn

from utils.base_networks import MLP, FourierFeatures


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Supports optional time conditioning (times=None means no time input).
    This matches JAX behavior where actor_onestep_flow doesn't use time.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims=(512, 512, 512, 512),
        encoder=None,
        use_fourier_features=False,
        fourier_feature_dim=64,
        use_time=True,
    ):
        super(ActorVectorField, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_fourier_features = use_fourier_features
        self.use_time = use_time
        self.encoder = encoder

        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = obs_dim

        # Add action dimension
        self.input_dim += self.action_dim

        # Only add time dimension if use_time=True
        if self.use_time:
            if self.use_fourier_features:
                self.ff = FourierFeatures(fourier_feature_dim)
                self.input_dim += fourier_feature_dim
            else:
                self.ff = None
                self.input_dim += 1
        else:
            self.ff = None

        self.mlp = MLP(self.input_dim, self.action_dim, hidden_dim=hidden_dims)

    def forward(self, o, x_t, t=None, is_encoded=False):
        """Forward pass.

        Args:
            o: observations
            x_t: noisy actions
            t: time (optional, only used if use_time=True)
            is_encoded: whether observations are already encoded
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(o)
        else:
            observations = o

        if self.use_time and t is not None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=observations.device).float()

            if self.use_fourier_features:
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                time_emb = self.ff(t)
            else:
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                time_emb = t

            inputs = torch.cat([observations, x_t, time_emb], dim=-1)
        else:
            # No time input - just concat observations and actions
            inputs = torch.cat([observations, x_t], dim=-1)

        v = self.mlp(inputs)
        return v


class MeanActorVectorField(nn.Module):
    """Actor vector field with t_begin and t_end time conditioning (for JVP-based flow matching)."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims=(512, 512, 512, 512),
        encoder=None,
        use_fourier_features=False,
        fourier_feature_dim=64,
    ):
        super(MeanActorVectorField, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_fourier_features = use_fourier_features
        self.encoder = encoder

        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = obs_dim

        if self.use_fourier_features:
            self.ff = FourierFeatures(fourier_feature_dim)
            self.input_dim += self.action_dim + fourier_feature_dim * 2  # two time embeddings
        else:
            self.ff = None
            self.input_dim += self.action_dim + 2  # t_begin and t_end

        self.mlp = MLP(self.input_dim, self.action_dim, hidden_dim=hidden_dims)

    def forward(self, observations, x_t, t_begin, t_end, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if not isinstance(t_begin, torch.Tensor):
            t_begin = torch.tensor(t_begin, device=observations.device).float()
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor(t_end, device=observations.device).float()

        if self.use_fourier_features:
            t_begin = self.ff(t_begin)
            t_end = self.ff(t_end)

        # Ensure all tensors have the same number of dimensions for concatenation
        # observations: (B, obs_dim) or (B, seq_len, obs_dim)
        # x_t: (B, action_dim)
        # t_begin, t_end: (B, 1) or (B, time_dim)

        # If observations is 3D (has sequence dimension), flatten it
        if observations.dim() == 3:
            # (B, seq_len, obs_dim) -> (B, seq_len * obs_dim)
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)

        # Ensure t_begin and t_end are 2D
        if t_begin.dim() == 1:
            t_begin = t_begin.unsqueeze(-1)
        if t_end.dim() == 1:
            t_end = t_end.unsqueeze(-1)

        inputs = torch.cat([observations, x_t, t_begin, t_end], dim=-1)

        v = self.mlp(inputs)
        return v
