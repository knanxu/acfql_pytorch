"""
Mean Flow Behavior Cloning Agent (PyTorch Implementation)

Uses JVP-based flow matching with t_begin, t_end time parameters.

Supports:
- Encoder types: 'identity', 'mlp', 'image' (ResNet-based), 'impala'
- Policy types: 'mlp', 'chiunet', 'chitransformer', 'jannerunet'
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

from utils.network_factory import get_encoder, get_network
from utils.networks import MeanActorVectorField, Value


class MFBCAgent:
    """
    Mean Flow Behavior Cloning Agent.

    Features:
    - JVP-based flow matching with t_begin, t_end
    - Action chunking support (predict multiple future actions)
    - Visual encoders: ResNet-based, IMPALA, MLP, Identity
    - Policy networks: MLP, ChiUNet, ChiTransformer, JannerUNet
    - Multi-modal observations (multiple cameras + proprioceptive data)
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
        encoder: Optional[nn.Module] = None,
        critic_encoder: Optional[nn.Module] = None,
    ):
        """Initialize MFBC Agent."""
        self.actor = actor
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
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
        if self.critic_encoder is not None:
            self.critic_encoder.to(self.device)

        # Get policy type from config (check both network_type and policy_type for compatibility)
        self.policy_type = config.get('network_type', config.get('policy_type', 'mlp'))

        # Training step counter
        self.step = 0

    def _encode_observations(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode observations using the encoder if present."""
        if self.encoder is not None:
            return self.encoder(observations)
        return observations
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BC flow matching loss using JVP-based formulation.

        Uses (x_1 - x_0) as tangent vector for JVP computation.

        Supports both MLP-based (MeanActorVectorField) and advanced networks
        (ChiUNet, ChiTransformer, JannerUNet).
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

        # Handle action chunking - determine format based on policy type
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 1)

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies expect (B, T, act_dim)
            if actions.ndim == 2:
                # Reshape flat actions to (B, T, act_dim)
                batch_actions = actions.reshape(batch_size, horizon_length, action_dim)
            else:
                batch_actions = actions

            # JVP-based flow matching
            x_1 = torch.randn_like(batch_actions)
            x_0 = batch_actions

            # Sample time pair using logit-normal distribution
            mu = self.config['time_logit_mu']
            sigma = self.config['time_logit_sigma']
            time_pair = mu + sigma * torch.randn(batch_size, 2, device=self.device)
            time_pair = torch.sigmoid(time_pair)
            sorted_pair, _ = torch.sort(time_pair, dim=-1)
            t_begin = sorted_pair[:, 0]  # (B,)
            t_end = sorted_pair[:, 1]    # (B,)

            # Apply instant mask
            instant_mask = torch.bernoulli(
                torch.full((batch_size,), self.config['time_instant_prob'], device=self.device)
            )
            t_begin = torch.where(instant_mask.bool(), t_end, t_begin)

            # Interpolate at t_end
            x_t = (1 - t_end.unsqueeze(-1).unsqueeze(-1)) * x_0 + t_end.unsqueeze(-1).unsqueeze(-1) * x_1

            # Compute JVP
            def cond_mean_flow(actions_input, t_begin_input, t_end_input):
                if self.encoder is not None:
                    return self.actor(actions_input, t_begin_input, t_end_input, obs_emb)
                else:
                    return self.actor(actions_input, t_begin_input, t_end_input, observations)

            tangent_actions = x_1 - x_0
            tangent_t_begin = torch.zeros_like(t_begin)
            tangent_t_end = torch.ones_like(t_end)

            primals = (x_t, t_begin, t_end)
            tangents = (tangent_actions, tangent_t_begin, tangent_t_end)

            u, dudt = torch.autograd.functional.jvp(cond_mean_flow, primals, tangents)

            # Handle scalar output from JannerUNet
            if isinstance(u, tuple):
                u, _ = u
            if isinstance(dudt, tuple):
                dudt, _ = dudt

            # Compute target
            u_tgt = (x_1 - x_0 - (t_end - t_begin).unsqueeze(-1).unsqueeze(-1) * dudt).detach()

            # Predict
            if self.encoder is not None:
                pred = self.actor(x_t, t_begin, t_end, obs_emb)
            else:
                pred = self.actor(x_t, t_begin, t_end, observations)

            # Handle scalar output from JannerUNet
            if isinstance(pred, tuple):
                pred, _ = pred

            # Compute MSE loss
            if 'valid' in batch:
                valid = batch['valid']  # (B, T)
                loss_per_element = (pred - u_tgt) ** 2
                bc_flow_loss = (loss_per_element * valid.unsqueeze(-1)).mean()
            else:
                bc_flow_loss = ((pred - u_tgt) ** 2).mean()
        else:
            # MLP-based policy expects flat (B, act_dim * T)
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

            # Compute JVP
            def cond_mean_flow(actions_input, t_begin_input, t_end_input):
                if self.encoder is not None:
                    return self.actor(obs_emb, actions_input, t_begin_input, t_end_input, is_encoded=True)
                else:
                    return self.actor(observations, actions_input, t_begin_input, t_end_input)

            tangent_actions = x_1 - x_0
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

            # Compute target
            u_tgt = (x_1 - x_0 - (t_end - t_begin) * dudt).detach()

            # Predict
            if self.encoder is not None:
                pred = self.actor(obs_emb, x_t, t_begin, t_end, is_encoded=True)
            else:
                pred = self.actor(observations, x_t, t_begin, t_end)

            # Compute MSE loss
            if self.config['action_chunking'] and 'valid' in batch:
                valid = batch['valid']
                if valid.dim() == 3:
                    valid = valid.squeeze(-1)
                loss_per_element = (pred - u_tgt) ** 2
                loss_per_element = loss_per_element.reshape(
                    batch_size, self.config['horizon_length'], self.config['action_dim']
                )
                bc_flow_loss = (loss_per_element * valid.unsqueeze(-1)).mean()
            else:
                bc_flow_loss = ((pred - u_tgt) ** 2).mean()

        total_loss = self.config['bc_weight'] * bc_flow_loss

        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            't_begin_mean': t_begin.mean().item() if t_begin.numel() > 0 else 0.0,
            't_end_mean': t_end.mean().item() if t_end.numel() > 0 else 0.0,
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
    ) -> Tuple['MFBCAgent', Dict[str, float]]:
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
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noises: torch.Tensor,
    ):
        """
        Compute actions using mean flow: action = noise - u(obs, noise, 0, 1).

        Supports both MLP-based and advanced network types.
        """
        # Encode observations if needed
        obs_emb = self._encode_observations(observations)

        # Get batch size
        if isinstance(obs_emb, dict):
            first_key = list(obs_emb.keys())[0]
            batch_size = obs_emb[first_key].shape[0]
        else:
            batch_size = obs_emb.shape[0]

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies work with (B, T, act_dim)
            # noises should already be in (B, T, act_dim) format
            times_begin = torch.zeros(batch_size, device=self.device)
            times_end = torch.ones(batch_size, device=self.device)

            if self.encoder is not None:
                u = self.actor(noises, times_begin, times_end, obs_emb)
            else:
                u = self.actor(noises, times_begin, times_end, observations)

            # Handle scalar output from JannerUNet
            if isinstance(u, tuple):
                u, _ = u

            actions = noises - u
        else:
            # MLP-based policy expects flat (B, act_dim * T)
            times_begin = torch.zeros((*noises.shape[:-1], 1), device=self.device)
            times_end = torch.ones_like(times_begin)

            if self.encoder is not None:
                u = self.actor(obs_emb, noises, times_begin, times_end, is_encoded=True)
            else:
                u = self.actor(observations, noises, times_begin, times_end)

            actions = noises - u

        actions = torch.clamp(actions, -1, 1)
        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor], np.ndarray],
        temperature: float = 0.0,
    ) -> np.ndarray:
        """
        Sample actions using the trained flow model.

        Supports both MLP-based and advanced network types.
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
        horizon_length = self.config.get('horizon_length', 1)

        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # U-Net/Transformer policies work with (B, T, act_dim)
            actions = torch.randn(batch_size, horizon_length, action_dim, device=self.device)
        else:
            # MLP-based policy expects flat (B, act_dim * T)
            if self.config.get('action_chunking', True):
                full_action_dim = action_dim * horizon_length
            else:
                full_action_dim = action_dim
            actions = torch.randn(batch_size, full_action_dim, device=self.device)

        if temperature > 0:
            actions = actions * temperature

        actions = self.compute_meanflow_actions(observations, actions)

        # Reshape to flat format if needed
        if self.policy_type in ['chiunet', 'chitransformer', 'jannerunet']:
            # (B, T, act_dim) -> (B, T * act_dim)
            actions = actions.reshape(batch_size, -1)

        return actions.cpu().numpy()
    
    def save(self, path: str):
        """Save agent checkpoint."""
        save_dict = {
            'actor': self.actor.state_dict(),
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
    ) -> 'MFBCAgent':
        """Create a new MFBC agent."""
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

        # Get network type from policy_type or network_type
        network_type = config.get('network_type', config.get('policy_type', 'mlp'))
        config['network_type'] = network_type

        # Create encoder using factory
        encoder = get_encoder(config)
        network_input_dim = encoder.output_dim if hasattr(encoder, 'output_dim') else config.get('emb_dim', 256)

        # Create policy network using factory
        actor = get_network(config)

        # Create critic encoder (separate from actor encoder)
        critic_encoder = get_encoder(config)

        # Create critic
        full_action_dim = action_dim * horizon_length if config.get('action_chunking', True) else action_dim
        critic = Value(
            observation_dim=network_input_dim,
            action_dim=full_action_dim,
            hidden_dim=config.get('value_hidden_dims', (512, 512, 512, 512)),
            num_ensembles=config.get('num_qs', 2),
            encoder=None,  # Encoder is separate
            layer_norm=config.get('layer_norm', True),
        )

        target_critic = copy.deepcopy(critic)

        # Create optimizers
        lr = config.get('lr', 3e-4)
        weight_decay = config.get('weight_decay', 0.0)

        # Collect parameters for actor optimizer (includes encoder)
        actor_params = list(actor.parameters())
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
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=config,
            encoder=encoder,
            critic_encoder=critic_encoder,
        )


def get_config():
    """Get default configuration for MFBC agent (ml_collections.ConfigDict)."""
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='mfbc',
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
            network_type='mlp',  # Network: 'mlp', 'chiunet', 'chitransformer', 'jannerunet'
            actor_hidden_dims=(512, 512, 512, 512),  # For MLP network

            # ChiUNet-specific
            model_dim=256,
            kernel_size=5,
            cond_predict_scale=True,
            obs_as_global_cond=True,
            dim_mult=[1, 2, 2],

            # ChiTransformer-specific
            d_model=256,
            nhead=4,
            num_layers=8,
            attn_dropout=0.3,
            n_cond_layers=0,

            # JannerUNet-specific
            norm_type='groupnorm',
            attention=False,

            # Common network settings
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,

            # Training
            discount=0.99,
            tau=0.005,
            num_qs=2,
            flow_steps=10,
            weight_decay=0.0,
            use_critic=False,
            bc_weight=1.0,

            # Action chunking
            horizon_length=5,
            action_chunking=True,
            obs_steps=1,  # Observation context length

            # Time embedding
            use_fourier_features=False,
            fourier_feature_dim=64,
            timestep_emb_type='positional',
            timestep_emb_params=None,
            disable_time_embedding=False,

            # MFBC specific
            time_logit_mu=-0.4,
            time_logit_sigma=1.0,
            time_instant_prob=0.2,
        )
    )
