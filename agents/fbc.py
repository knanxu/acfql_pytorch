"""
Behavior Cloning Agent with Flow Matching (PyTorch Implementation)

Adapted from JAX/Flax ACFQLAgent to pure BC with flow matching.
Keeps critic structure for future RL experiments.

Supports:
- Encoder types: 'impala' or 'impala_small' (visual), or None (state-based)
- Policy types: 'mlp' only (ChiUNet and ChiTransformer not implemented yet)
- Single image observations (multi-image not implemented yet)

Config is passed as a dictionary (similar to JAX version).
"""

import copy
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.model import ActorVectorField, Value, ImpalaEncoder


class BCAgent:
    """
    Behavior Cloning Agent using Flow Matching.
    
    Features:
    - Flow-based action modeling (continuous normalizing flow)
    - Action chunking support (predict multiple future actions)
    - Visual encoder: IMPALA (single image only)
    - Policy network: MLP-based ActorVectorField
    - Critic network structure (for future RL fine-tuning)
    
    Note: ChiUNet, ChiTransformer, and multi-image encoders are not yet implemented.
    
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
        """Initialize BC Agent.
        
        Args:
            actor: Policy network (flow model)
            critic: Critic network (for RL fine-tuning)
            target_critic: Target critic network
            actor_optimizer: Optimizer for actor (and encoder if present)
            critic_optimizer: Optimizer for critic
            config: Configuration dictionary
            encoder: Optional visual encoder for actor (separate from actor's internal encoder)
            critic_encoder: Optional visual encoder for critic
        """
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
        
        # Get policy type from config
        self.policy_type = config.get('policy_type', 'mlp')
        
        # Training step counter
        self.step = 0
    
    def _encode_observations(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode observations using the encoder if present.
        
        Args:
            observations: Raw observations (tensor or dict for multi-image)
        
        Returns:
            Encoded observations tensor
        """
        if self.encoder is not None:
            return self.encoder(observations)
        return observations
    
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BC flow matching loss.
        
        Args:
            batch: Dictionary containing:
                - observations: (B, obs_dim) or (B, C, H, W) or Dict[str, Tensor]
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
        
        # Handle action chunking - determine format based on policy type
        action_dim = self.config['action_dim']
        horizon_length = self.config.get('horizon_length', 1)
        
        if self.policy_type in ['chiunet', 'chitransformer']:
            raise NotImplementedError(f"Policy type '{self.policy_type}' not implemented yet. Use 'mlp' policy type.")
        else:
            # MLP-based policy expects flat (B, act_dim * T)
            if self.config.get('action_chunking', True):
                if actions.ndim == 3:
                    batch_actions = actions.reshape(batch_size, -1)
                else:
                    batch_actions = actions
            else:
                if actions.ndim == 3:
                    batch_actions = actions[:, 0, :]
                else:
                    batch_actions = actions
            
            # Flow matching
            x_0 = torch.randn_like(batch_actions)
            x_1 = batch_actions
            
            t = torch.rand(batch_size, 1, device=self.device)
            x_t = (1 - t) * x_0 + t * x_1
            target_vel = x_1 - x_0
            
            # Predict velocity (pass encoded observations)
            if self.encoder is not None:
                pred_vel = self.actor(obs_emb, x_t, t, is_encoded=True)
            else:
                pred_vel = self.actor(observations, x_t, t)
            
            # Compute MSE loss
            if self.config.get('action_chunking', True) and 'valid' in batch:
                valid = batch['valid']  # (B, T)
                loss_per_element = (pred_vel - target_vel) ** 2
                loss_per_element = loss_per_element.reshape(
                    batch_size, horizon_length, action_dim
                )
                bc_flow_loss = (loss_per_element * valid.unsqueeze(-1)).mean()
            else:
                bc_flow_loss = ((pred_vel - target_vel) ** 2).mean()
        
        total_loss = self.config.get('bc_weight', 1.0) * bc_flow_loss
        
        info = {
            'actor_loss': total_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
        }
        
        return total_loss, info
    
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute critic loss (placeholder for future RL).
        
        For BC training, this returns zero loss.
        """
        if not self.config.get('use_critic', False):
            # Return zero loss for BC-only training
            return torch.tensor(0.0, device=self.device), {
                'critic_loss': 0.0,
                'q_mean': 0.0,
                'q_max': 0.0,
                'q_min': 0.0,
            }
        
        # TODO: Implement for RL fine-tuning
        raise NotImplementedError("Critic loss for RL not implemented yet")
    
    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss (actor + critic)."""
        info = {}
        
        # Actor loss (BC flow matching)
        actor_loss, actor_info = self.actor_loss(batch)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Critic loss (zero for BC, can be activated for RL)
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        total = actor_loss + critic_loss
        info['total_loss'] = total.item()
        
        return total, info
    
    def _update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step."""
        # Compute loss
        loss, info = self.total_loss(batch)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        # Update critic (if used)
        if self.config.get('use_critic', False):
            self.critic_optimizer.zero_grad()
            self.critic_optimizer.step()
            self.target_update()
        
        self.step += 1
        
        return info
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one update step.
        
        Args:
            batch: Dictionary of batched data (already on device)
        
        Returns:
            info: Dictionary of logging information
        """
        # Helper function to move to device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            return x
        
        # Move batch to device if needed
        batch = {k: to_device(v) for k, v in batch.items()}
        
        return self._update(batch)
    
    def batch_update(
        self, 
        batches: Dict[str, torch.Tensor]
    ) -> Tuple['BCAgent', Dict[str, float]]:
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
    def compute_flow_actions(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noises: torch.Tensor,
    ):
        """Compute actions from the BC flow model using the Euler method.
        
        Args:
            observations: Raw or encoded observations
            noises: Initial noise tensor
        
        Returns:
            Predicted actions
        """
        # Encode observations if needed
        obs_emb = self._encode_observations(observations)
        
        # Get batch size
        if isinstance(obs_emb, dict):
            first_key = list(obs_emb.keys())[0]
            batch_size = obs_emb[first_key].shape[0]
        else:
            batch_size = obs_emb.shape[0]
        
        flow_steps = self.config.get('flow_steps', 10)
        
        if self.policy_type in ['chiunet', 'chitransformer']:
            raise NotImplementedError(f"Policy type '{self.policy_type}' not implemented yet. Use 'mlp' policy type.")
        else:
            # MLP-based policy
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
        temperature: float = 0.0,
    ) -> np.ndarray:
        """
        Sample actions using the trained flow model.
        
        Args:
            observations: (B, obs_dim) or (B, C, H, W) or Dict[str, Tensor]
            temperature: Sampling temperature (0 = deterministic)
        
        Returns:
            actions: (B, action_dim * horizon_length) flattened
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
        
        if self.policy_type in ['chiunet', 'chitransformer']:
            raise NotImplementedError(f"Policy type '{self.policy_type}' not implemented yet. Use 'mlp' policy type.")
        else:
            # Start with noise in flat format
            full_action_dim = action_dim * horizon_length if self.config.get('action_chunking', True) else action_dim
            actions = torch.randn(batch_size, full_action_dim, device=self.device)
        
        if temperature > 0:
            actions = actions * temperature
        
        # Integrate flow using Euler method
        actions = self.compute_flow_actions(observations, actions)
        
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
    ) -> 'BCAgent':
        """
        Create a new BC agent.
        
        Args:
            observation_shape: Shape of observations
                - For state: (obs_dim,) tuple
                - For images: (C, H, W) tuple
                - For multi-image: shape_meta dict (from robomimic_image_utils)
            action_dim: Dimension of action space
            config: Agent configuration dictionary with keys:
                - encoder_type: 'impala', 'impala_small', or None (only IMPALA supported)
                - policy_type: 'mlp' (only MLP supported currently)
                - emb_dim: Encoder output dimension (default 256)
                - horizon_length: Action chunking length
                - action_chunking: Enable action chunking
                - ... other standard config options
        
        Returns:
            agent: Initialized BC agent
        """
        config = dict(config)  # Copy to avoid mutation
        config['action_dim'] = action_dim
        
        # Set device if not specified
        if 'device' not in config:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(config['device'])
        
        # Determine observation type
        is_multi_image = isinstance(observation_shape, dict)
        is_visual = is_multi_image or (isinstance(observation_shape, tuple) and len(observation_shape) == 3)
        
        if is_multi_image:
            config['observation_dim'] = observation_shape  # shape_meta dict
        elif is_visual:
            config['observation_dim'] = observation_shape  # (C, H, W)
        else:
            config['observation_dim'] = observation_shape[0]  # scalar obs_dim
        
        # Get configuration parameters
        encoder_type = config.get('encoder_type', config.get('encoder'))  # Support both names
        policy_type = config.get('policy_type', 'mlp')
        emb_dim = config.get('emb_dim', 256)
        horizon_length = config.get('horizon_length', 5)
        
        # Create encoder
        encoder = None
        network_input_dim = emb_dim  # Default
        
        if is_visual and encoder_type is not None:
            if is_multi_image:
                raise NotImplementedError("Multi-image encoder not implemented yet. Use single image observations.")
            else:
                # Single image encoder - only support IMPALA for now
                if encoder_type == 'impala' or encoder_type == 'impala_small':
                    encoder = ImpalaEncoder(
                        input_shape=observation_shape,
                        width=config.get('impala_width', 1),
                        stack_sizes=config.get('impala_stack_sizes', (16, 32, 32)),
                        num_blocks=config.get('impala_num_blocks', 2),
                        mlp_hidden_dims=config.get('impala_mlp_hidden_dims', (512,)),
                        layer_norm=config.get('layer_norm', False),
                    )
                    network_input_dim = encoder.output_dim
                else:
                    raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'impala' or 'impala_small' are supported.")
        elif not is_visual:
            network_input_dim = observation_shape[0]
        else:
            # Visual but no encoder specified - flatten
            network_input_dim = int(np.prod(observation_shape))
        
        # Full action dimension (with chunking)
        full_action_dim = action_dim
        if config.get('action_chunking', True):
            full_action_dim = action_dim * horizon_length
        
        # Create policy network based on type
        if policy_type == 'chiunet':
            raise NotImplementedError("ChiUNet policy not implemented yet. Use 'mlp' policy type.")
        elif policy_type == 'chitransformer':
            raise NotImplementedError("ChiTransformer policy not implemented yet. Use 'mlp' policy type.")
        else:
            # MLP-based ActorVectorField
            # If using separate encoder, don't pass it to actor
            actor = ActorVectorField(
                observation_dim=network_input_dim,
                action_dim=full_action_dim,
                hidden_dim=config.get('actor_hidden_dims', (512, 512, 512, 512)),
                encoder=None,  # Encoder is separate
                use_fourier_features=config.get('use_fourier_features', False),
                fourier_feature_dim=config.get('fourier_feature_dim', 64),
            )
        
        # Create critic encoder (separate from actor encoder)
        critic_encoder = None
        if is_visual and encoder_type is not None:
            if is_multi_image:
                raise NotImplementedError("Multi-image encoder not implemented yet. Use single image observations.")
            else:
                # Single image encoder - only support IMPALA for now
                if encoder_type == 'impala' or encoder_type == 'impala_small':
                    critic_encoder = ImpalaEncoder(
                        input_shape=observation_shape,
                        width=config.get('impala_width', 1),
                        stack_sizes=config.get('impala_stack_sizes', (16, 32, 32)),
                        num_blocks=config.get('impala_num_blocks', 2),
                        mlp_hidden_dims=config.get('impala_mlp_hidden_dims', (512,)),
                        layer_norm=config.get('layer_norm', False),
                    )
                else:
                    raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'impala' or 'impala_small' are supported.")
        
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
    """Get default configuration for BC agent (ml_collections.ConfigDict)."""
    import ml_collections
    return ml_collections.ConfigDict(
        dict(
            agent_name='fbc',  # Agent name
            lr=3e-4,  # Learning rate
            batch_size=256,  # Batch size
            
            # Encoder configuration
            encoder_type=None,  # Visual encoder: None, 'impala', 'resnet18', 'resnet34', 'vit_b_16'
            emb_dim=256,  # Encoder output dimension
            resize_shape=None,  # Optional image resize (H, W)
            crop_shape=None,  # Optional crop for augmentation (H, W)
            random_crop=True,  # Enable random crop augmentation
            share_rgb_model=False,  # Share encoder across multiple cameras
            
            # Impala-specific
            impala_width=1,
            impala_stack_sizes=(16, 32, 32),
            impala_num_blocks=2,
            impala_mlp_hidden_dims=(512,),
            
            # Policy configuration
            policy_type='mlp',  # Policy network: 'mlp', 'chiunet', 'chitransformer'
            actor_hidden_dims=(512, 512, 512, 512),  # For MLP policy
            
            # ChiUNet-specific
            model_dim=256,
            kernel_size=5,
            cond_predict_scale=True,
            dim_mult=[1, 2, 2],
            
            # ChiTransformer-specific
            d_model=256,
            nhead=4,
            num_layers=8,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            n_cond_layers=0,
            
            # Time embedding
            timestep_emb_type='positional',
            time_emb_dim=256,
            use_fourier_features=False,
            fourier_feature_dim=64,
            
            # Critic configuration
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            num_qs=2,
            
            # Training
            discount=0.99,
            tau=0.005,
            weight_decay=0.0,
            use_critic=False,
            bc_weight=1.0,
            
            # Action chunking
            horizon_length=5,
            action_chunking=True,
            obs_steps=1,  # Observation context length
            
            # Flow
            flow_steps=10,
        )
    )
