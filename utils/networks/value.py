"""Value/Critic networks for Q-learning.

Contains Value network for Q(s,a) or V(s) estimation with ensemble support.
"""

import torch
import torch.nn as nn

from utils.base_networks import MLP


class Value(nn.Module):
    """Value/Critic Network.

    Can be used as V(s) network (when action_dim=0 or None)
    or as Q(s, a) network (when action_dim > 0).

    Features:
    - Contains num_ensembles independent MLP networks (Ensemble).
    - Output shape is (num_ensembles, batch_size) for easy min/mean computation.
    """
    def __init__(self,
                 observation_dim,
                 action_dim=None,
                 hidden_dim=(512, 512, 512, 512),
                 num_ensembles=2,
                 encoder=None,
                 layer_norm=True,
                ):
        super(Value, self).__init__()

        self.num_ensembles = num_ensembles
        self.encoder = encoder
        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = observation_dim

        if action_dim is not None and action_dim > 0:
            self.input_dim += action_dim

        # Build Ensemble
        # Use nn.ModuleList to create multiple independent MLP instances
        # Each MLP outputs dimension 1 (representing Value)
        self.nets = nn.ModuleList([
            MLP(input_dim=self.input_dim,
                action_dim=1,
                hidden_dim=hidden_dim,
                activations=nn.GELU,
                activate_final=False,
                layer_norm=layer_norm,
                )
            for _ in range(num_ensembles)
        ])

    def forward(self, observations, actions=None):
        """Forward pass.

        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim) [optional]

        Returns:
            values: (num_ensembles, batch_size)
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations

        if actions is not None:
            inputs = torch.cat([inputs, actions], dim=-1)

        outputs = []
        for net in self.nets:
            out = net(inputs)
            outputs.append(out)

        # [batch_size, 1] * N -> [num_ensembles, batch_size, 1]
        outputs = torch.stack(outputs, dim=0)

        # Remove last dimension -> [num_ensembles, batch_size]
        return outputs.squeeze(-1)
