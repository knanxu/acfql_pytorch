"""
Improved Mean Flow Behavior Cloning Agent (PyTorch Implementation)

Enhanced JVP-based flow matching with improved tangent vector calculation.
Config is passed as a dictionary (similar to JAX version).
"""

import copy
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.model import MeanActorVectorField, Value, ImpalaEncoder


class IMFBCAgent:
    """
    Improved Mean Flow Behavior Cloning Agent.
    
    Features:
    - Enhanced JVP-based flow matching (uses network output as tangent)
    - Action chunking support (predict multiple future actions)
    - Visual encoder support (IMPALA-CNN)
    - Critic network structure (for future RL fine-tuning)
    
    Config is passed as a dictionary (similar to JAX version).
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        config: Dict[str, Any],
    ):
        """Initialize IMFBC Agent."""
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config = config
        
        self.device = torch.device(config['device'])
        
        # Save encoder reference from actor (for external encoding in inference)
        self.encoder = actor.encoder if hasattr(actor, 'encoder') else None
        
        # Move models to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        
        # Training step counter
        self.step = 0
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BC flow matching loss using improved JVP-based formulation.
        
        Key difference from MFBC: uses network output at (t_end, t_end) as tangent
        instead of (x_1 - x_0).
        """
        observations = batch['observations']
        actions = batch['actions']
        
        batch_size = observations.shape[0]
        
        # Handle action chunking
        if self.config['action_chunking']:
            if actions.ndim == 3:
                batch_actions = actions.reshape(batch_size, -1)
            else:
                batch_actions = actions
        else:
            if actions.ndim == 3:
                batch_actions = actions[:, 0, :]
            else:
                batch_actions = actions
        
        action_dim = batch_actions.shape[-1]
        
        # JVP-based flow matching
        x_1 = torch.randn_like(batch_actions)
        x_0 = batch_actions
        
        # Sample time pair using logit-normal distribution
        mu = self.config['time_logit_mu']
        sigma = self.config['time_logit_sigma']
        time_pair = mu + sigma * torch.randn(batch_size, 2, device=self.device)
        time_pair = torch.sigmoid(time_pair)
        sorted_pair, _ = torch.sort(time_pair, dim=-1)
        t_begin = sorted_pair[:, :1]
        t_end = sorted_pair[:, 1:]
        
        # Apply instant mask
        instant_mask = torch.bernoulli(
            torch.full((batch_size, 1), self.config['time_instant_prob'], device=self.device)
        )
        t_begin = torch.where(instant_mask.bool(), t_end, t_begin)
        
        # Interpolate at t_end
        x_t = (1 - t_end) * x_0 + t_end * x_1
        
        # Compute JVP with improved tangent (use network output at t_end, t_end)
        def cond_mean_flow(actions_input, t_begin_input, t_end_input):
            return self.actor(observations, actions_input, t_begin_input, t_end_input)
        
        # Key improvement: tangent_actions = network(obs, x_t, t_end, t_end)
        tangent_actions = self.actor(observations, x_t, t_end, t_end)
        tangent_t_begin = torch.zeros_like(t_begin)
        tangent_t_end = torch.ones_like(t_end)
        
        primals = (x_t, t_begin, t_end)
        tangents = (tangent_actions, tangent_t_begin, tangent_t_end)
        
        u, dudt = torch.autograd.functional.jvp(cond_mean_flow, primals, tangents)
        
        # Apply mask for action chunking
        if self.config['action_chunking'] and 'valid' in batch:
            valid = batch['valid']
            if valid.dim() == 3:
                valid = valid.squeeze(-1)
            
            total_flat_dim = batch_actions.shape[-1]
            horizon = self.config['horizon_length']
            single_step_dim = total_flat_dim // horizon
            
            mask_expanded = valid.unsqueeze(-1).repeat(1, 1, single_step_dim)
            mask_flat = mask_expanded.reshape(batch_size, -1)
            dudt = dudt * mask_flat
        
        # Prediction with stop_gradient on dudt
        pred = self.actor(observations, x_t, t_begin, t_end) + (t_end - t_begin) * dudt.detach()
        
        # Loss target: (pred - x_1 + x_0)^2
        if self.config['action_chunking'] and 'valid' in batch:
            valid = batch['valid']
            if valid.dim() == 3:
                valid = valid.squeeze(-1)
            loss_per_element = (pred - x_1 + x_0) ** 2
            loss_per_element = loss_per_element.reshape(
                batch_size, self.config['horizon_length'], self.config['action_dim']
            )
            bc_flow_loss = (loss_per_element * valid.unsqueeze(-1)).mean()
        else:
            bc_flow_loss = ((pred - x_1 + x_0) ** 2).mean()
        
        total_loss = self.config['bc_weight'] * bc_flow_loss
        
        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            't_begin_mean': t_begin.mean().item(),
            't_end_mean': t_end.mean().item(),
        }
        
        return total_loss, info
    
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic loss (placeholder for future RL)."""
        if not self.config.get('use_critic', False):
            return torch.tensor(0.0, device=self.device), {
                'critic_loss': 0.0,
                'q_mean': 0.0,
                'q_max': 0.0,
                'q_min': 0.0,
            }
        raise NotImplementedError("Critic loss for RL not implemented yet")
    
    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss (actor + critic)."""
        info = {}
        
        actor_loss, actor_info = self.actor_loss(batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        total = actor_loss + critic_loss
        info['total_loss'] = total.item()
        
        return total, info
    
    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step."""
        loss, info = self.total_loss(batch)
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        if self.config.get('use_critic', False):
            self.critic_optimizer.zero_grad()
            self.critic_optimizer.step()
            self.target_update()
        
        self.step += 1
        return info
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one update step."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        return self._update(batch)
    
    def batch_update(
        self, 
        batches: Dict[str, torch.Tensor]
    ) -> Tuple['IMFBCAgent', Dict[str, float]]:
        """Perform multiple updates (one per batch)."""
        num_batches = next(iter(batches.values())).shape[0]
        
        all_info = []
        for i in range(num_batches):
            batch = {k: v[i] for k, v in batches.items()}
            info = self.update(batch)
            all_info.append(info)
        
        avg_info = {}
        for key in all_info[0].keys():
            avg_info[key] = np.mean([info[key] for info in all_info])
        
        return self, avg_info
    
    def target_update(self):
        """Soft update of target critic network."""
        if not self.config.get('use_critic', False):
            return
        
        tau = self.config['tau']
        for target_param, param in zip(
            self.target_critic.parameters(), 
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    @torch.no_grad()
    def compute_meanflow_actions(
        self,
        observations: torch.Tensor,
        noises: torch.Tensor,
    ):
        """Compute actions using mean flow: action = noise - u(obs, noise, 0, 1)."""
        if self.encoder is not None:
            observations = self.encoder(observations)
        times_begin = torch.zeros((*noises.shape[:-1], 1), device=self.device)
        times_end = torch.ones_like(times_begin)
        actions = noises - self.actor(observations, noises, times_begin, times_end, is_encoded=True)
        actions = torch.clamp(actions, -1, 1)
        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        observations: torch.Tensor,
        temperature: float = 0.0,
    ) -> np.ndarray:
        """Sample actions using the trained flow model."""
        if not isinstance(observations, torch.Tensor):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(self.device)
        
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        if self.config['action_chunking']:
            action_dim *= self.config['horizon_length']
        
        actions = torch.randn(batch_size, action_dim, device=self.device)
        if temperature > 0:
            actions = actions * temperature
        
        actions = self.compute_meanflow_actions(observations, actions)
        return actions.cpu().numpy()
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.step = checkpoint.get('step', 0)
        print(f"Agent loaded from {path} (step {self.step})")
    
    @classmethod
    def create(
        cls,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
    ) -> 'IMFBCAgent':
        """Create a new IMFBC agent."""
        config['action_dim'] = action_dim
        
        is_visual = len(observation_shape) == 3
        
        if is_visual:
            config['observation_dim'] = observation_shape
        else:
            config['observation_dim'] = observation_shape[0]
        
        if 'device' not in config:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(config['device'])
        
        # Create encoder if visual
        encoder = None
        if is_visual and config.get('encoder') is not None:
            if config['encoder'] == 'impala_small':
                encoder = ImpalaEncoder(
                    input_shape=observation_shape,
                    width=1,
                    stack_sizes=(16, 32, 32),
                    num_blocks=2,
                    mlp_hidden_dims=(512,),
                    layer_norm=config.get('layer_norm', False),
                )
            else:
                raise ValueError(f"Unknown encoder: {config['encoder']}")
        
        # Determine input dimension
        if encoder is not None:
            network_input_dim = encoder.output_dim
        else:
            network_input_dim = config['observation_dim'] if not is_visual else np.prod(observation_shape)
        
        # Full action dimension
        full_action_dim = action_dim
        if config.get('action_chunking', True):
            full_action_dim = action_dim * config.get('horizon_length', 5)
        
        # Create actor (MeanActorVectorField for JVP-based flow)
        actor = MeanActorVectorField(
            observation_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dim=config.get('actor_hidden_dims', (512, 512, 512, 512)),
            encoder=encoder,
            use_fourier_features=config.get('use_fourier_features', False),
            fourier_feature_dim=config.get('fourier_feature_dim', 64),
        )
        
        # Create critic
        critic_encoder = None
        if is_visual and config.get('encoder') is not None:
            if config['encoder'] == 'impala_small':
                critic_encoder = ImpalaEncoder(
                    input_shape=observation_shape,
                    width=1,
                    stack_sizes=(16, 32, 32),
                    num_blocks=2,
                    mlp_hidden_dims=(512,),
                    layer_norm=config.get('layer_norm', False),
                )
        
        critic = Value(
            observation_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dim=config.get('value_hidden_dims', (512, 512, 512, 512)),
            num_ensembles=config.get('num_qs', 2),
            encoder=critic_encoder,
        )
        
        target_critic = copy.deepcopy(critic)
        
        # Create optimizers
        lr = config.get('lr', 3e-4)
        weight_decay = config.get('weight_decay', 0.0)
        
        if weight_decay > 0:
            actor_optimizer = optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
            critic_optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
        
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=config,
        )


def get_config():
    """Get default configuration for IMFBC agent (ml_collections.ConfigDict)."""
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='imfbc',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            flow_steps=10,
            encoder=None,
            horizon_length=5,
            action_chunking=True,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,
            use_critic=False,
            bc_weight=1.0,
            # IMFBC specific
            time_logit_mu=-0.4,
            time_logit_sigma=1.0,
            time_instant_prob=0.2,
        )
    )
