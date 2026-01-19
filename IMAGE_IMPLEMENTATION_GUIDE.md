# Image-Based Training Implementation Guide

This guide documents the implementation of image-based training support for the acfql_pytorch project, based on the much-ado-about-noising repository.

## What Has Been Implemented

### Phase 1: Core Infrastructure ✅

1. **Timestep Embeddings** (`utils/embeddings.py`)
   - `SinusoidalEmbedding`: Sinusoidal positional embeddings
   - `FourierEmbedding`: Random Fourier features
   - `PositionalEmbedding`: Learnable positional embeddings
   - Registry system: `SUPPORTED_TIMESTEP_EMBEDDING`

2. **Visual Encoders** (`utils/encoders.py`)
   - `IdentityEncoder`: Pass-through encoder with dropout
   - `MLPEncoder`: MLP encoder for low-dimensional observations
   - `MultiImageObsEncoder`: Multi-modal encoder supporting:
     - ResNet-based visual encoders (ResNet18, ResNet34, ResNet50)
     - Multiple RGB cameras
     - Low-dimensional proprioceptive data
     - Image preprocessing (resize, crop, normalization)
     - Sequence handling for temporal observations

3. **Network Utilities** (`utils/network_utils.py`)
   - `GroupNorm1d`: Group normalization for 1D inputs
   - `Downsample1d` / `Upsample1d`: Sampling layers for U-Net
   - Helper functions: `unsqueeze_expand_at`, `flatten`, `reshape_dimensions`, `join_dimensions`

### Phase 2: Network Architectures ✅

4. **ChiUNet** (`utils/networks/chiunet.py`)
   - U-Net architecture from Diffusion Policy
   - Features:
     - Residual blocks with conditioning
     - Multi-scale processing (encoder-decoder)
     - Global or local observation conditioning
     - Scalar output head for auxiliary predictions
     - ~20M parameters (configurable)
   - Best for: Image-based manipulation tasks

5. **ChiTransformer** (`utils/networks/chitransformer.py`)
   - Transformer architecture from Diffusion Policy
   - Features:
     - Causal self-attention for actions
     - Cross-attention for observations
     - Positional embeddings
     - Scalar output head
     - ~30M parameters (configurable)
   - Best for: Sequential reasoning tasks

6. **Network Factory** (`utils/network_factory.py`)
   - `get_network()`: Create networks based on config
   - `get_encoder()`: Create encoders based on config
   - `get_network_and_encoder()`: Create both together
   - Supports: 'mlp', 'chiunet', 'chitransformer'

## What Needs To Be Done

### Phase 3: Agent Integration (HIGH PRIORITY)

The existing agents (FBC, MFBC, IMFBC, FQL) need to be updated to support image observations. Here's what needs to be modified:

#### 3.1 Update Agent Initialization

**File**: `agents/fbc.py` (and similarly for mfbc.py, imfbc.py, fql.py)

**Current structure**:
```python
class BCAgent:
    def __init__(self, actor, critic, target_critic, actor_optimizer, critic_optimizer, config, encoder=None, critic_encoder=None):
        self.actor = actor
        self.encoder = encoder  # Currently only supports IMPALA
        # ...
```

**What to change**:
1. Update `create()` classmethod to use `get_network_and_encoder()`:
```python
@classmethod
def create(cls, observation_shape, action_dim, config):
    # Determine if using images
    obs_type = config.get('obs_type', 'state')

    if obs_type == 'image':
        # Use new network factory
        from utils.network_factory import get_network_and_encoder

        # Update config with necessary parameters
        config['act_dim'] = action_dim
        config['obs_dim'] = config.get('emb_dim', 256)  # Use embedding dim for images

        actor, encoder = get_network_and_encoder(config)
    else:
        # Existing state-based logic
        encoder = None
        actor = ActorVectorField(...)

    # Create critic (can also use encoder)
    critic = Value(...)

    # Create optimizers
    if encoder is not None:
        actor_params = list(actor.parameters()) + list(encoder.parameters())
    else:
        actor_params = actor.parameters()

    actor_optimizer = optim.Adam(actor_params, lr=config['lr'])
    # ...
```

2. Update `actor_loss()` to handle different network types:
```python
def actor_loss(self, batch):
    observations = batch['observations']
    actions = batch['actions']

    # Encode observations
    if self.encoder is not None:
        obs_emb = self.encoder(observations)  # Can be dict for multi-image
    else:
        obs_emb = observations

    # Get network type
    network_type = self.config.get('network_type', 'mlp')

    if network_type in ['chiunet', 'chitransformer']:
        # These networks expect (B, Ta, act_dim) and (B, To, obs_dim)
        # Sample time parameters
        batch_size = actions.shape[0]
        s = torch.rand(batch_size, device=self.device)
        t = torch.rand(batch_size, device=self.device)

        # Interpolate between noise and target
        noise = torch.randn_like(actions)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * actions

        # Predict velocity
        v_pred, scalar = self.actor(x_t, s, t, obs_emb)
        v_target = actions - noise

        # Flow matching loss
        loss = F.mse_loss(v_pred, v_target)

    else:  # mlp
        # Existing MLP-based flow matching logic
        # Flatten observations and actions
        obs_flat = obs_emb.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        # ... existing code ...

    return loss, info
```

3. Update `sample_actions()` for inference:
```python
def sample_actions(self, observations, num_steps=10):
    # Encode observations
    if self.encoder is not None:
        obs_emb = self.encoder(observations)
    else:
        obs_emb = observations

    network_type = self.config.get('network_type', 'mlp')

    if network_type in ['chiunet', 'chitransformer']:
        # Start from noise
        batch_size = obs_emb.shape[0] if not isinstance(obs_emb, dict) else list(obs_emb.values())[0].shape[0]
        Ta = self.config['horizon_length']
        act_dim = self.config['action_dim']

        x = torch.randn(batch_size, Ta, act_dim, device=self.device)

        # Euler integration
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=self.device)
            s = torch.zeros(batch_size, device=self.device)

            v, _ = self.actor(x, s, t, obs_emb)
            x = x + v * dt

        actions = torch.clamp(x, -1, 1)

    else:  # mlp
        # Existing MLP-based sampling
        # ...

    return actions
```

#### 3.2 Update Config System

**File**: `agents/fbc.py` (in `get_config()` function)

Add image-specific config parameters:
```python
def get_config():
    config = ml_collections.ConfigDict()

    # Existing parameters
    config.agent_name = 'fbc'
    config.lr = 3e-4
    config.batch_size = 256
    # ...

    # NEW: Network architecture selection
    config.network_type = 'mlp'  # 'mlp', 'chiunet', 'chitransformer'
    config.emb_dim = 256
    config.num_layers = 4
    config.model_dim = 256  # For ChiUNet
    config.n_heads = 4  # For ChiTransformer

    # NEW: Observation type
    config.obs_type = 'state'  # 'state' or 'image'
    config.obs_steps = 2  # Number of observation frames

    # NEW: Encoder settings (for images)
    config.encoder = None  # None, 'mlp', 'image'
    config.encoder_dropout = 0.25
    config.rgb_model_name = 'resnet18'
    config.use_group_norm = True
    config.random_crop = True
    config.resize_shape = None
    config.crop_shape = None

    # NEW: Shape metadata (for images)
    config.shape_meta = {}  # Will be populated from dataset

    return config
```

### Phase 4: Dataset Support (HIGH PRIORITY)

#### 4.1 Add Image Dataset Loading

**File**: `envs/robomimic_utils.py`

Add function to load image-based datasets:
```python
def get_image_dataset(env, env_name, obs_keys=None):
    """Load Robomimic image dataset.

    Args:
        env: Robomimic environment
        env_name: Environment name (e.g., 'lift-mh-low_dim')
        obs_keys: List of observation keys to include

    Returns:
        Dataset object with image observations
    """
    import h5py
    from utils.datasets import Dataset

    # Parse environment name
    task, dataset_type, obs_type = parse_env_name(env_name)

    # Get dataset path
    dataset_path = get_dataset_path(task, dataset_type, obs_type='image')

    # Load dataset
    with h5py.File(dataset_path, 'r') as f:
        demos = f['data']

        # Collect data
        observations = {}
        actions = []
        rewards = []
        terminals = []

        for demo_id in demos.keys():
            demo = demos[demo_id]

            # Load images (multiple cameras)
            for key in obs_keys:
                if key not in observations:
                    observations[key] = []

                if 'image' in key:
                    # Load image data (H, W, C) format
                    img_data = demo['obs'][key][:]
                    observations[key].append(img_data)
                else:
                    # Load low-dim data
                    obs_data = demo['obs'][key][:]
                    observations[key].append(obs_data)

            # Load actions
            actions.append(demo['actions'][:])
            rewards.append(demo['rewards'][:])
            terminals.append(demo['dones'][:])

        # Concatenate
        for key in observations:
            observations[key] = np.concatenate(observations[key], axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)

    # Create Dataset
    dataset = Dataset.create(
        observations=observations,  # Dict of observations
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        next_observations=observations,  # Simplified
        masks=1.0 - terminals,
    )

    # Set pytorch format for images
    dataset.pytorch_format = True

    return dataset
```

#### 4.2 Update `make_env_and_datasets()`

**File**: `envs/env_utils.py`

Update to detect and load image datasets:
```python
def make_env_and_datasets(env_name, **kwargs):
    """Make environment and datasets.

    Automatically detects if environment uses images or state observations.
    """
    if is_robomimic_env(env_name):
        from envs import robomimic_utils

        # Parse environment name to detect observation type
        if 'image' in env_name or kwargs.get('obs_type') == 'image':
            # Load image-based dataset
            env = robomimic_utils.make_env(env_name)

            # Define observation keys based on task
            obs_keys = kwargs.get('obs_keys', [
                'agentview_image',
                'robot0_eye_in_hand_image',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
            ])

            dataset = robomimic_utils.get_image_dataset(env, env_name, obs_keys)

            # Create shape_meta for encoder
            shape_meta = {
                'obs': {},
                'action': {'shape': [env.action_space.shape[0]]}
            }

            for key in obs_keys:
                if 'image' in key:
                    # Assuming 84x84 RGB images
                    shape_meta['obs'][key] = {
                        'shape': [3, 84, 84],
                        'type': 'rgb'
                    }
                else:
                    # Low-dim observations
                    obs_dim = dataset[key].shape[-1]
                    shape_meta['obs'][key] = {
                        'shape': [obs_dim],
                        'type': 'low_dim'
                    }

            # Attach shape_meta to dataset
            dataset.shape_meta = shape_meta

        else:
            # Existing state-based loading
            env = robomimic_utils.make_env(env_name)
            dataset = robomimic_utils.get_dataset(env, env_name)

        eval_env = env
        train_dataset = dataset
        val_dataset = None

    elif is_d4rl_env(env_name):
        # Existing D4RL logic
        # ...

    return env, eval_env, train_dataset, val_dataset
```

### Phase 5: Training Script Updates (MEDIUM PRIORITY)

#### 5.1 Update `train_bc.py`

**Changes needed**:

1. Add command-line arguments for image training:
```python
# Add to FLAGS
flags.DEFINE_string('obs_type', 'state', 'Observation type: state or image')
flags.DEFINE_string('network_type', 'mlp', 'Network type: mlp, chiunet, chitransformer')
flags.DEFINE_integer('obs_steps', 2, 'Number of observation frames')
```

2. Update agent creation to pass shape_meta:
```python
# After loading dataset
if hasattr(train_dataset, 'shape_meta'):
    config['shape_meta'] = train_dataset.shape_meta
    config['obs_type'] = 'image'
else:
    config['obs_type'] = 'state'

# Update observation shape
if config['obs_type'] == 'image':
    observation_shape = None  # Not used for images
    config['obs_dim'] = config['emb_dim']  # Use embedding dimension
else:
    observation_shape = (env.observation_space.shape[0],)
    config['obs_dim'] = observation_shape[0]
```

3. Update batch preprocessing:
```python
# In training loop
if config['obs_type'] == 'image':
    # Batch is already in correct format from dataset
    batch_tensor = {
        'observations': batch['obs'],  # Dict of tensors
        'actions': torch.from_numpy(batch['actions']).float(),
    }
else:
    # Existing state-based preprocessing
    batch_tensor = {
        'observations': torch.from_numpy(batch['observations']).float(),
        'actions': torch.from_numpy(batch['actions']).float(),
    }
```

### Phase 6: Example Configurations (LOW PRIORITY)

Create example config files for image-based training:

**File**: `agents/fbc_image.py`
```python
"""FBC agent config for image-based tasks."""
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # Agent settings
    config.agent_name = 'fbc'
    config.lr = 1e-4  # Lower LR for images
    config.batch_size = 128  # Smaller batch for memory
    config.weight_decay = 1e-5

    # Network architecture
    config.network_type = 'chiunet'  # or 'chitransformer'
    config.emb_dim = 256
    config.model_dim = 128
    config.kernel_size = 5
    config.dim_mult = [1, 2, 2]
    config.cond_predict_scale = True
    config.obs_as_global_cond = True

    # Observation settings
    config.obs_type = 'image'
    config.obs_steps = 2
    config.horizon_length = 16  # Must be power of 2 for ChiUNet

    # Encoder settings
    config.encoder = 'image'
    config.rgb_model_name = 'resnet18'
    config.use_group_norm = True
    config.random_crop = True
    config.use_seq = True
    config.keep_horizon_dims = True

    # Flow matching settings
    config.flow_steps = 10
    config.discount = 0.99
    config.action_chunking = True

    # Device
    config.device = 'cuda'

    return config
```

### Phase 7: Testing (HIGH PRIORITY)

Create test script to verify image training:

**File**: `test_image_training.py`
```python
"""Test script for image-based training."""
import torch
from envs.env_utils import make_env_and_datasets
from agents.fbc import BCAgent, get_config

# Load image dataset
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('lift-mh-image')

print(f"Dataset size: {len(train_dataset)}")
print(f"Shape meta: {train_dataset.shape_meta}")

# Create agent config
config = get_config()
config['obs_type'] = 'image'
config['network_type'] = 'chiunet'
config['shape_meta'] = train_dataset.shape_meta
config['obs_steps'] = 2
config['horizon_length'] = 16

# Create agent
agent = BCAgent.create(
    observation_shape=None,
    action_dim=env.action_space.shape[0],
    config=config
)

# Test forward pass
batch = train_dataset.sample_sequence(
    batch_size=4,
    sequence_length=16
)

batch_tensor = {
    'observations': batch['obs'],  # Dict of tensors
    'actions': torch.from_numpy(batch['actions']).float(),
}

# Test loss computation
loss, info = agent.actor_loss(batch_tensor)
print(f"Loss: {loss.item()}")
print(f"Info: {info}")

# Test sampling
obs_sample = {k: v[:1] for k, v in batch['obs'].items()}
actions = agent.sample_actions(obs_sample)
print(f"Sampled actions shape: {actions.shape}")
```

## Summary of Implementation Status

### ✅ Completed (Phase 1-2)
- Timestep embeddings
- Visual encoders (ResNet-based)
- Network utilities
- ChiUNet architecture
- ChiTransformer architecture
- Network factory functions

### 🔄 In Progress (Phase 3-4)
- Agent integration (needs manual updates)
- Dataset support (needs implementation)

### ⏳ Pending (Phase 5-7)
- Training script updates
- Example configurations
- Testing and validation

## Next Steps

1. **Update FBC agent** (`agents/fbc.py`):
   - Modify `create()` to use network factory
   - Update `actor_loss()` for different network types
   - Update `sample_actions()` for inference

2. **Implement image dataset loading** (`envs/robomimic_utils.py`):
   - Add `get_image_dataset()` function
   - Update `make_env_and_datasets()` to detect image datasets

3. **Update training script** (`train_bc.py`):
   - Add image-specific command-line arguments
   - Update batch preprocessing
   - Handle shape_meta from dataset

4. **Test on Robomimic**:
   - Run `test_image_training.py`
   - Train on lift-mh-image task
   - Verify convergence and performance

## Usage Example (After Full Implementation)

```bash
# Train ChiUNet on Robomimic lift task with images
python train_bc.py \
    --env_name=lift-mh-image \
    --agent=agents/fbc_image.py \
    --agent.network_type=chiunet \
    --agent.obs_steps=2 \
    --agent.horizon_length=16 \
    --agent.batch_size=128 \
    --offline_steps=500000 \
    --eval_interval=50000

# Train ChiTransformer
python train_bc.py \
    --env_name=lift-mh-image \
    --agent=agents/fbc_image.py \
    --agent.network_type=chitransformer \
    --agent.emb_dim=384 \
    --agent.n_heads=6 \
    --agent.num_layers=8
```

## Key Differences from much-ado-about-noising

1. **Config System**: Keeping `ml_collections.ConfigDict` instead of Hydra
2. **Agent Structure**: Maintaining existing agent class structure
3. **Dataset Interface**: Using existing `Dataset` class with extensions
4. **Training Loop**: Minimal changes to existing training scripts

This approach minimizes disruption to your existing codebase while adding full image support.
