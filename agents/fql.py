"""
Flow Q-Learning (FQL) Agent (PyTorch Implementation)

Implements FQL algorithm with:
- Flow-based action modeling (continuous normalizing flow)
- Action chunking support
- Critic network for Q-learning
- Distillation policy for one-step action generation

Supports:
- Encoder types: 'identity', 'mlp', 'image' (ResNet-based), 'impala'
- Policy types: 'mlp' (ActorVectorField)
- Multi-modal observations (multiple cameras + proprioceptive data)
- Action chunking with flexible horizon lengths

Config is passed as a dictionary (similar to JAX version).
"""

import copy
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.network_factory import get_encoder
from utils.networks import ActorVectorField, Value


class FQLAgent:
    """
    Flow Q-Learning Agent.

    Features:
    - Flow-based action modeling (continuous normalizing flow)
    - Action chunking support (predict multiple future actions)
    - Visual encoders: ResNet-based, IMPALA, MLP, Identity
    - Multi-modal observations (multiple cameras + proprioceptive data)
    - Critic network for Q-learning
    - Distillation policy for one-step action generation

    Config is passed as a dictionary (similar to JAX version).
    """

    def __init__(
        self,
        actor: nn.Module,
        actor_onestep: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        config: Dict[str, Any],
        encoder: Optional[nn.Module] = None,
        critic_encoder: Optional[nn.Module] = None,
    ):
        """Initialize FQL Agent."""
        self.actor = actor  # Flow network (like BCAgent)
        self.actor_onestep = actor_onestep  # One-step distillation policy
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config = config
        self.encoder = encoder
        self.critic_encoder = critic_encoder

        self.device = torch.device(config['device'])

        # Move models to device
        self.actor.to(self.device)
        self.actor_onestep.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
        if self.critic_encoder is not None:
            self.critic_encoder.to(self.device)

        # Training step counter
        self.step = 0

    def _encode_observations(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode observations using the encoder if present."""
        if self.encoder is not None:
            return self.encoder(observations)
        return observations
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FQL actor loss.
        
        Includes:
        - BC flow loss (flow matching)
        - Distillation loss (train one-step to match flow)
        - Q loss (maximize Q value)
        
        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim) or (B, C, H, W)
                - actions: (B, action_dim) or (B, T, action_dim) if chunking
                - valid: (B, T) mask for action chunks (optional)
        
        Returns:
            loss: Scalar loss
            info: Dictionary of logging information
        """
        observations = batch['observations']
        actions = batch['actions']

        # Get batch size
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            batch_size = observations[first_key].shape[0]
        else:
            batch_size = observations.shape[0]

        # Encode observations if using separate encoder
        obs_emb = self._encode_observations(observations)
        
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
        
        # === BC Flow Loss ===
        # Flow matching: x_t = (1-t) * x_0 + t * x_1
        x_0 = torch.randn_like(batch_actions)  # noise
        x_1 = batch_actions  # target action
        t = torch.rand(batch_size, 1, device=self.device)
        
        x_t = (1 - t) * x_0 + t * x_1
        target_vel = x_1 - x_0  # target velocity

        if self.encoder is not None:
            pred_vel = self.actor(obs_emb, x_t, t, is_encoded=True)
        else:
            pred_vel = self.actor(observations, x_t, t)
        
        if self.config['action_chunking'] and 'valid' in batch:
            valid = batch['valid']
            if valid.ndim == 2:
                loss_per_element = (pred_vel - target_vel) ** 2
                loss_per_element = loss_per_element.reshape(
                    batch_size, self.config['horizon_length'], self.config['action_dim']
                )
                bc_flow_loss = (loss_per_element * valid.unsqueeze(-1)).mean()
            else:
                bc_flow_loss = ((pred_vel - target_vel) ** 2).mean()
        else:
            bc_flow_loss = ((pred_vel - target_vel) ** 2).mean()
        
        # === Distillation Loss ===
        # Train one-step actor to match flow actions
        noises = torch.randn(batch_size, action_dim, device=self.device)

        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(observations, noises)

        # One-step actor (no time input - matches JAX version)
        if self.encoder is not None:
            actor_actions = self.actor_onestep(obs_emb, noises, is_encoded=True)
        else:
            actor_actions = self.actor_onestep(observations, noises)
        distill_loss = ((actor_actions - target_flow_actions) ** 2).mean()

        # === Q Loss ===
        # Maximize Q value for one-step actions
        actor_actions_clipped = torch.clamp(actor_actions, -1, 1)

        # Encode observations for critic if needed
        if self.critic_encoder is not None:
            obs_emb_critic = self.critic_encoder(observations)
            qs = self.critic(obs_emb_critic, actor_actions_clipped)
        else:
            qs = self.critic(observations, actor_actions_clipped)
        q = qs.mean(dim=0)  # Mean over ensemble
        q_loss = -q.mean()
        
        # Total actor loss
        total_loss = self.config['bc_weight'] * bc_flow_loss + self.config['alpha'] * distill_loss + q_loss
        
        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
        }
        
        return total_loss, info
    
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FQL critic loss (TD loss).
        
        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim)
                - actions: (B, action_dim)
                - next_observations: (B, obs_dim)
                - rewards: (B,)
                - masks: (B,) - 1 if not done, 0 if done
        
        Returns:
            loss: Scalar critic loss
            info: Dictionary of logging information
        """
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch.get('next_observations', observations)
        rewards = batch['rewards']
        masks = batch.get('masks', 1.0 - batch.get('terminals', torch.zeros_like(rewards)))

        # Get batch size
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            batch_size = observations[first_key].shape[0]
        else:
            batch_size = observations.shape[0]

        # Encode observations for critic if needed
        if self.critic_encoder is not None:
            obs_emb = self.critic_encoder(observations)
            next_obs_emb = self.critic_encoder(next_observations)
        else:
            obs_emb = observations
            next_obs_emb = next_observations
        
        # Handle action chunking
        if self.config['action_chunking']:
            if actions.ndim == 3:
                batch_actions = actions.reshape(batch_size, -1)
            else:
                batch_actions = actions
            
            if next_observations.ndim == 3:
                next_obs = next_observations[:, -1, :]
            else:
                next_obs = next_observations
            
            if rewards.ndim == 2:
                last_reward = rewards[:, -1]
                last_mask = masks[:, -1] if masks.ndim == 2 else masks
            else:
                last_reward = rewards
                last_mask = masks
        else:
            if actions.ndim == 3:
                batch_actions = actions[:, 0, :]
            else:
                batch_actions = actions
            next_obs = next_observations
            last_reward = rewards if rewards.ndim == 1 else rewards[:, 0]
            last_mask = masks if masks.ndim == 1 else masks[:, 0]
        
        action_dim = batch_actions.shape[-1]
        
        # Compute target Q value
        with torch.no_grad():
            # Sample next actions using one-step policy (no time input - matches JAX)
            noises = torch.randn(batch_size, action_dim, device=self.device)

            # Encode next observations for actor if needed
            if self.encoder is not None:
                next_obs_emb_actor = self._encode_observations(next_obs)
                next_actions = self.actor_onestep(next_obs_emb_actor, noises, is_encoded=True)
            else:
                next_actions = self.actor_onestep(next_obs, noises)
            next_actions = torch.clamp(next_actions, -1, 1)

            # Compute target Q values
            if self.critic_encoder is not None:
                next_qs = self.target_critic(next_obs_emb, next_actions)  # (num_ensembles, B)
            else:
                next_qs = self.target_critic(next_obs, next_actions)  # (num_ensembles, B)
            
            if self.config['q_agg'] == 'min':
                next_q = next_qs.min(dim=0)[0]
            else:
                next_q = next_qs.mean(dim=0)
            
            # Compute TD target
            discount_factor = self.config['discount'] ** (self.config['horizon_length'] if self.config['action_chunking'] else 1)
            target_q = last_reward + discount_factor * last_mask * next_q
        
        # Compute current Q values
        if self.critic_encoder is not None:
            current_qs = self.critic(obs_emb, batch_actions)  # (num_ensembles, B)
        else:
            current_qs = self.critic(observations, batch_actions)  # (num_ensembles, B)
        
        # Get valid mask (critical for action chunking - masks out cross-episode samples)
        if 'valid' in batch:
            valid = batch['valid']
            if valid.ndim == 2:
                # Use last timestep's validity for TD target
                last_valid = valid[:, -1]
            else:
                last_valid = valid
        else:
            last_valid = torch.ones(batch_size, device=self.device)
        
        # Compute critic loss with valid mask (MSE for each ensemble, then mean)
        # This is critical to avoid Q-value explosion from cross-episode samples
        td_errors = (current_qs - target_q.unsqueeze(0)) ** 2  # (num_ensembles, B)
        critic_loss = (td_errors * last_valid.unsqueeze(0)).mean()
        
        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': current_qs.mean().item(),
            'q_max': current_qs.max().item(),
            'q_min': current_qs.min().item(),
            'target_q_mean': target_q.mean().item(),
        }
        
        return critic_loss, info
    
    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute total loss (actor + critic)."""
        info = {}
        
        # Actor loss
        actor_loss, actor_info = self.actor_loss(batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Critic loss
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        info['total_loss'] = (actor_loss + critic_loss).item()
        
        return actor_loss, critic_loss, info
    
    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step."""
        # Compute losses
        actor_loss, critic_loss, info = self.total_loss(batch)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update target network
        self.target_update()
        
        self.step += 1
        
        return info
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step.
        
        Args:
            batch: Dictionary of batched data
        
        Returns:
            info: Dictionary of logging information
        """
        # Move batch to device if needed
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        return self._update(batch)
    
    def batch_update(
        self, 
        batches: Dict[str, torch.Tensor]
    ) -> Tuple['FQLAgent', Dict[str, float]]:
        """
        Perform multiple updates (one per batch).
        
        Args:
            batches: Dictionary where each value has shape (num_batches, batch_size, ...)
        
        Returns:
            agent: Self (for compatibility with JAX-style API)
            info: Averaged logging information
        """
        num_batches = next(iter(batches.values())).shape[0]
        
        all_info = []
        for i in range(num_batches):
            batch = {k: v[i] for k, v in batches.items()}
            info = self.update(batch)
            all_info.append(info)
        
        # Average info across batches
        avg_info = {}
        for key in all_info[0].keys():
            avg_info[key] = np.mean([info[key] for info in all_info])
        
        return self, avg_info
    
    def target_update(self):
        """Soft update of target critic network."""
        tau = self.config['tau']
        for target_param, param in zip(
            self.target_critic.parameters(), 
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    @torch.no_grad()
    def compute_flow_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noises: torch.Tensor,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        # Encode observations if needed
        obs_emb = self._encode_observations(observations)

        # Get batch size
        if isinstance(obs_emb, dict):
            first_key = list(obs_emb.keys())[0]
            batch_size = obs_emb[first_key].shape[0]
        else:
            batch_size = obs_emb.shape[0]

        flow_steps = self.config.get('flow_steps', 10)

        actions = noises
        for i in range(flow_steps):
            t = torch.full((batch_size, 1), i / flow_steps, device=self.device)
            if self.encoder is not None:
                vels = self.actor(obs_emb, actions, t, is_encoded=True)
            else:
                vels = self.actor(observations, actions, t)
            actions = actions + vels / flow_steps
        actions = torch.clamp(actions, -1, 1)

        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor], np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Sample actions using the trained policy.

        Uses the one-step distillation policy for fast inference.

        Args:
            observations: Observations
            temperature: Sampling temperature (for noise scaling)

        Returns:
            actions: numpy array
        """
        # Helper to convert to torch
        def to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(self.device)
            elif isinstance(x, dict):
                return {k: to_torch(v) for k, v in x.items()}
            elif isinstance(x, torch.Tensor):
                return x.to(self.device)
            return x

        observations = to_torch(observations)

        # Ensure batch dimension exists
        if isinstance(observations, dict):
            first_key = list(observations.keys())[0]
            if observations[first_key].ndim in [1, 3]:  # (dim,) or (C, H, W)
                observations = {k: v.unsqueeze(0) for k, v in observations.items()}
            batch_size = observations[first_key].shape[0]
        else:
            if observations.ndim == 1:
                observations = observations.unsqueeze(0)
            batch_size = observations.shape[0]

        action_dim = self.config['action_dim']
        if self.config['action_chunking']:
            action_dim *= self.config['horizon_length']

        # Sample noise
        noises = torch.randn(batch_size, action_dim, device=self.device)
        if temperature != 1.0:
            noises = noises * temperature

        # Use one-step policy for fast inference
        obs_emb = self._encode_observations(observations)
        if self.encoder is not None:
            actions = self.actor_onestep(obs_emb, noises, is_encoded=True)
        else:
            actions = self.actor_onestep(observations, noises)
        actions = torch.clamp(actions, -1, 1)

        return actions.cpu().numpy()

    def save(self, path: str):
        """Save agent checkpoint."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_onestep': self.actor_onestep.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
        }
        if self.encoder is not None:
            save_dict['encoder'] = self.encoder.state_dict()
        if self.critic_encoder is not None:
            save_dict['critic_encoder'] = self.critic_encoder.state_dict()

        torch.save(save_dict, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_onestep.load_state_dict(checkpoint['actor_onestep'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.encoder is not None and 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if self.critic_encoder is not None and 'critic_encoder' in checkpoint:
            self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.step = checkpoint.get('step', 0)
        print(f"Agent loaded from {path} (step {self.step})")
    
    @classmethod
    def create(
        cls,
        observation_shape: Union[Tuple[int, ...], Dict],
        action_dim: int,
        config: Dict[str, Any],
    ) -> 'FQLAgent':
        """Create a new FQL agent."""
        config = dict(config)  # Copy to avoid mutation
        config['action_dim'] = action_dim
        config['act_dim'] = action_dim  # For factory functions

        # Set device if not specified
        if 'device' not in config:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        device = torch.device(config['device'])

        # Determine observation type
        is_multi_image = isinstance(observation_shape, dict)
        is_visual = is_multi_image or (isinstance(observation_shape, tuple) and len(observation_shape) == 3)

        if is_multi_image:
            config['shape_meta'] = observation_shape
            config['obs_type'] = 'image'
        elif is_visual:
            config['obs_dim'] = int(np.prod(observation_shape))
            config['obs_type'] = 'image'
        else:
            config['obs_dim'] = observation_shape[0]
            config['obs_type'] = 'state'

        # Get configuration parameters
        horizon_length = config.get('horizon_length', 5)
        config['Ta'] = horizon_length  # Action sequence length
        config['To'] = config.get('obs_steps', 1)  # Observation sequence length

        # Create encoder using factory
        encoder = get_encoder(config)
        network_input_dim = encoder.output_dim if hasattr(encoder, 'output_dim') else config.get('emb_dim', 256)

        # Full action dimension
        full_action_dim = action_dim
        if config.get('action_chunking', True):
            full_action_dim = action_dim * horizon_length

        # Create actor (flow network)
        actor = ActorVectorField(
            obs_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dims=config.get('actor_hidden_dims', (512, 512, 512, 512)),
            encoder=None,  # Encoder is separate
            use_fourier_features=config.get('use_fourier_features', False),
            fourier_feature_dim=config.get('fourier_feature_dim', 64),
        )

        # Create one-step actor (for distillation)
        # NOTE: One-step actor does NOT use time input (matches JAX version)
        actor_onestep = ActorVectorField(
            obs_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dims=config.get('actor_hidden_dims', (512, 512, 512, 512)),
            encoder=None,  # Encoder is separate
            use_fourier_features=False,
            use_time=False,  # Critical: one-step actor doesn't use time (matches JAX)
        )

        # Create critic encoder (separate from actor encoder)
        critic_encoder = get_encoder(config)

        # Create critic
        critic = Value(
            observation_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dim=config.get('value_hidden_dims', (512, 512, 512, 512)),
            num_ensembles=config.get('num_qs', 2),
            encoder=None,  # Encoder is separate
            layer_norm=config.get('layer_norm', True),
        )

        # Create target critic
        target_critic = copy.deepcopy(critic)

        # Create optimizers
        lr = config.get('lr', 3e-4)
        weight_decay = config.get('weight_decay', 0.0)

        # Collect parameters for actor optimizer (includes both actors and encoder)
        actor_params = list(actor.parameters()) + list(actor_onestep.parameters())
        if encoder is not None:
            actor_params += list(encoder.parameters())

        # Collect parameters for critic optimizer
        critic_params = list(critic.parameters())
        if critic_encoder is not None:
            critic_params += list(critic_encoder.parameters())

        if weight_decay > 0:
            actor_optimizer = optim.AdamW(actor_params, lr=lr, weight_decay=weight_decay)
            critic_optimizer = optim.AdamW(critic_params, lr=lr, weight_decay=weight_decay)
        else:
            actor_optimizer = optim.Adam(actor_params, lr=lr)
            critic_optimizer = optim.Adam(critic_params, lr=lr)

        return cls(
            actor=actor,
            actor_onestep=actor_onestep,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=config,
            encoder=encoder,
            critic_encoder=critic_encoder,
        )


def get_config():
    """Get default configuration for FQL agent (ml_collections.ConfigDict)."""
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='fql',
            lr=3e-4,
            batch_size=256,

            # Encoder configuration
            encoder='mlp',  # Encoder type: 'identity', 'mlp', 'image', 'impala'
            obs_type='state',  # Observation type: 'state', 'image', 'keypoint'
            emb_dim=256,  # Encoder output dimension
            encoder_hidden_dims=[256, 256],  # For MLP encoder
            encoder_dropout=0.25,  # Encoder dropout

            # Image encoder specific
            rgb_model_name='resnet18',  # ResNet model: 'resnet18', 'resnet34', 'resnet50'
            resize_shape=None,  # Optional image resize (H, W)
            crop_shape=None,  # Optional crop for augmentation (H, W)
            random_crop=True,  # Enable random crop augmentation
            share_rgb_model=False,  # Share encoder across multiple cameras
            use_group_norm=True,  # Use GroupNorm in ResNet
            imagenet_norm=False,  # Use ImageNet normalization

            # Network configuration
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,

            # Training
            discount=0.99,
            tau=0.005,
            num_qs=2,
            flow_steps=10,
            weight_decay=0.0,

            # Action chunking
            horizon_length=5,
            action_chunking=True,
            obs_steps=1,  # Observation context length

            # Time embedding
            use_fourier_features=False,
            fourier_feature_dim=64,

            # FQL specific
            bc_weight=1.0,
            alpha=100.0,
            q_agg='mean',  # Q aggregation method (min or mean)
        )
    )
