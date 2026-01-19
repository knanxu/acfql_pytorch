# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of offline-to-online reinforcement learning algorithms, focusing on flow-based behavior cloning and Q-learning. The codebase supports multiple environments (D4RL, Robomimic, OGBench) and implements several flow-matching based agents.

**NEW**: The codebase now supports image-based observations with visual encoders (ResNet) and advanced network architectures (ChiUNet, ChiTransformer) ported from the much-ado-about-noising repository. See `IMAGE_IMPLEMENTATION_GUIDE.md` for details.

**Image-based Robomimic**: Full support for robomimic image environments with multi-camera observations. See `envs/robomimic_image_utils.py` for implementation.

## Key Commands

### Training

```bash
# Basic BC training (state-based)
python train_bc.py --env_name=halfcheetah-medium-v2 --agent=agents/fbc.py

# BC training with action chunking
python train_bc.py --env_name=lift-mh-low_dim --agent=agents/imfbc.py --agent.horizon_length=10

# Offline-to-online training with FQL
python main.py --env_name=halfcheetah-medium-v2 --agent=agents/fql.py --offline_steps=1000000 --online_steps=1000000

# Visual task training
python train_bc.py --env_name=visual-singletask-task0-v0 --agent=agents/fbc.py --agent.encoder=impala_small
```

### Testing

```bash
# Test environment loading and visualization
python test_env.py --env_name=halfcheetah-medium-v2

# Test OGBench environment
python test.py --env_name=cube-triple-play-singletask-v0
```

### Configuration

All agents use `ml_collections.ConfigDict` for configuration. Override config parameters using:
```bash
python train_bc.py --agent=agents/fbc.py --agent.lr=1e-4 --agent.batch_size=128
```

## Architecture

### Agent Implementations

The codebase implements four flow-based agents (all in `agents/`):

1. **FBC (Flow-based BC)** - `agents/fbc.py`
   - Standard flow matching for behavior cloning
   - Supports action chunking and visual observations
   - Uses `ActorVectorField` for flow-based action generation

2. **MFBC (Mean Flow BC)** - `agents/mfbc.py`
   - JVP-based flow matching with configurable time intervals (t_begin, t_end)
   - More efficient gradient computation

3. **IMFBC (Improved Mean Flow BC)** - `agents/imfbc.py`
   - Enhanced JVP formulation
   - Default agent for BC training

4. **FQL (Flow Q-Learning)** - `agents/fql.py`
   - Full RL agent with critic network
   - Supports offline pretraining + online fine-tuning
   - Includes one-step distillation policy

All agents:
- Use dict-based config (not dataclass)
- Support action chunking via `horizon_length` parameter
- Can handle visual observations with IMPALA encoder
- Implement flow matching for continuous action spaces

### Training Scripts

- **`train_bc.py`**: Pure behavior cloning training (offline only)
  - Simpler, focused on BC
  - Uses `sample_sequence()` for action chunking
  - Evaluates periodically during training

- **`main.py`**: Offline-to-online RL training
  - Supports both offline pretraining and online fine-tuning
  - Uses `ReplayBuffer` for online data collection
  - Implements UTD (update-to-data) ratio for online training

### Data Pipeline

The data pipeline is unified across all environments:

1. **Environment Loading** (`envs/env_utils.py`):
   - `make_env_and_datasets()` - Auto-detects environment type and loads data
   - Supports D4RL, Robomimic, OGBench

2. **Dataset Class** (`utils/datasets.py`):
   - Unified interface for all offline RL datasets
   - Key methods:
     - `sample(batch_size)` - Sample single transitions
     - `sample_sequence(batch_size, sequence_length)` - Sample trajectories for action chunking
   - Supports PyTorch format conversion via `pytorch_format=True`
   - Handles frame stacking and data augmentation

3. **ReplayBuffer** (`utils/replay_buffer.py`):
   - Used for online training in `main.py`
   - Can be initialized from offline dataset
   - Supports sequence sampling for action chunking

### Neural Network Components

**Core Networks** (in `utils/model.py`):
- **`ActorVectorField`**: MLP-based flow policy network
  - Predicts velocity field for continuous normalizing flow
  - Supports Fourier features for time encoding
  - Best for state-based observations

- **`Value`**: Critic network for Q-learning
  - Standard MLP architecture
  - Used in FQL agent

- **`ImpalaEncoder`**: Basic visual encoder
  - Two variants: `impala` and `impala_small`
  - Legacy encoder for simple image tasks

**Advanced Networks** (in `utils/networks/`):
- **`ChiUNet`**: U-Net architecture from Diffusion Policy
  - Multi-scale encoder-decoder with residual blocks
  - Global or local observation conditioning
  - ~20M parameters, best for image-based manipulation
  - Requires horizon_length to be power of 2

- **`ChiTransformer`**: Transformer architecture from Diffusion Policy
  - Causal self-attention for actions
  - Cross-attention for observations
  - ~30M parameters, best for sequential reasoning

**Visual Encoders** (in `utils/encoders.py`):
- **`MultiImageObsEncoder`**: Multi-modal observation encoder
  - Supports ResNet18/34/50 for visual features
  - Handles multiple RGB cameras simultaneously
  - Combines visual and proprioceptive data
  - Image preprocessing (resize, crop, normalization)
  - Sequence handling for temporal observations

- **`MLPEncoder`**: MLP encoder for low-dimensional observations
  - Projects state observations to embedding space
  - Supports temporal observation sequences

- **`IdentityEncoder`**: Pass-through encoder with optional dropout

**Timestep Embeddings** (in `utils/embeddings.py`):
- **`SinusoidalEmbedding`**: Sinusoidal positional embeddings
- **`FourierEmbedding`**: Random Fourier features
- **`PositionalEmbedding`**: Learnable positional embeddings

### Flow Matching

The core algorithm uses continuous normalizing flow:

**Training**: Learn velocity field v(x_t, t) where x_t = (1-t)*x_0 + t*x_1
- x_0: Gaussian noise
- x_1: Expert action
- Loss: MSE between predicted and true velocity

**Inference**: Integrate ODE from noise to action using Euler method
- Start from x_0 ~ N(0, I)
- Iterate: x_{t+dt} = x_t + v(x_t, t) * dt
- Final x_1 is the predicted action

**Action Chunking**: Predict multiple future actions as a sequence
- Actor outputs shape: (batch, horizon_length, action_dim)
- Enables temporal consistency and planning

## Important Implementation Details

### Config System

All agents use `ml_collections.ConfigDict` (not dataclass):
```python
# In agent config file (e.g., agents/fbc.py)
def get_config():
    config = ml_collections.ConfigDict()
    config.agent_name = 'fbc'
    config.lr = 3e-4
    config.batch_size = 256
    # ...
    return config
```

Override via command line:
```bash
--agent.lr=1e-4 --agent.batch_size=128
```

### Action Chunking

Controlled by `horizon_length` parameter:
- `horizon_length=1`: Single-step actions (no chunking)
- `horizon_length>1`: Multi-step action prediction

When action chunking is enabled:
- Training uses `sample_sequence()` instead of `sample()`
- Actor outputs shape changes from (B, A) to (B, H, A)
- Evaluation executes actions sequentially from the chunk

### Visual Observations

**For image-based robomimic environments:**
```bash
# Training with image observations
python train_bc.py --env_name=lift-mh-image --agent=agents/fbc.py \
  --agent.encoder=image --agent.network_type=chiunet \
  --agent.rgb_model_name=resnet18 --agent.horizon_length=16
```

The environment automatically returns dict observations with:
- Camera images (e.g., `agentview_image`, `robot0_eye_in_hand_image`)
- Proprioceptive data (e.g., `robot0_eef_pos`, `robot0_eef_quat`)
- Images are normalized to [0, 1] in (C, H, W) format

**For other visual environments:**
Enable visual encoder via config:
```python
config.encoder = 'impala_small'  # or 'impala'
```

Dataset must use PyTorch format:
```python
train_dataset.pytorch_format = True  # Converts (H,W,C) to (C,H,W)
```

### Robomimic Environments

Special handling required:
- Reward adjustment: shift from [0,1] to [-1,0]
- Implemented in `process_train_dataset()` in both training scripts
- Environment names: `lift-mh-low_dim`, `can-ph-low_dim`, `square-mh-low_dim`, etc.

### Device Handling

Agents automatically use CUDA if available:
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

For multi-GPU, set `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0 python train_bc.py ...
```

## File Organization

```
acfql_pytorch/
├── agents/              # Agent implementations (fbc, mfbc, imfbc, fql)
├── envs/                # Environment utilities (D4RL, Robomimic, OGBench)
├── utils/               # Core utilities (datasets, models, replay buffer)
├── train_bc.py          # BC training script (offline only)
├── main.py              # Full RL training script (offline + online)
├── evaluation.py        # Evaluation utilities
├── log_utils.py         # Logging and wandb integration
└── runs/                # Training outputs (checkpoints, logs, videos)
```

## Common Patterns

### Adding a New Agent

1. Create `agents/new_agent.py` with:
   - Agent class inheriting common structure
   - `get_config()` function returning `ml_collections.ConfigDict`
   - `create()` classmethod for initialization
   - `update()` method for training step
   - `sample_actions()` method for inference
   - `save()` and `load()` methods for checkpoints

2. Register in `agents/__init__.py`:
   ```python
   from agents.new_agent import NewAgent, get_config as new_agent_get_config
   agents['new_agent'] = (NewAgent, new_agent_get_config)
   ```

### Dataset Processing

Always process datasets through `process_train_dataset()`:
- Handles dataset proportion
- Adjusts Robomimic rewards
- Converts to sparse rewards if needed

### Evaluation

Use the unified `evaluate()` function from `evaluation.py`:
- Handles action chunking automatically
- Records videos if requested
- Returns statistics dict with 'return', 'length', 'success' (if available)

## Differences from JAX Version

This is a PyTorch port of a JAX/Flax implementation. Key differences:

1. **State Management**: PyTorch uses mutable state (in-place updates) vs JAX's immutable functional approach
2. **Random Numbers**: PyTorch uses global RNG state vs JAX's explicit PRNG keys
3. **Checkpointing**: PyTorch uses `.pt` files vs JAX's custom checkpoint format
4. **Agent Updates**: `info = agent.update(batch)` vs JAX's `agent, info = agent.update(batch)`
5. **Config System**: Both use `ml_collections.ConfigDict` but PyTorch agents accept dict directly

See `BC_VS_JAX.md` for detailed comparison.

## Debugging Tips

### Common Issues

1. **CUDA OOM**: Reduce `batch_size` in agent config
2. **Slow training**: Check if CUDA is being used (`agent.device`)
3. **Poor performance**: Verify `horizon_length` matches task requirements
4. **Visual task failures**: Ensure `pytorch_format=True` for dataset

### Logging

All training runs log to:
- **CSV files**: `runs/*/offline_agent.csv`, `eval.csv`
- **Wandb**: Automatic if wandb is configured
- **Checkpoints**: Saved at intervals specified by `--save_interval`

### Monitoring

Key metrics to watch:
- `actor/bc_flow_loss`: Flow matching loss (should decrease)
- `eval/return`: Episode return (should increase)
- `eval/success`: Success rate for manipulation tasks

## Environment-Specific Notes

### D4RL
- Automatically downloads to `~/.d4rl/`
- State-based observations (vectors)
- Standard reward scaling

### Robomimic
- Requires manual download to `~/.robomimic/`
- Supports both low_dim (state) and image observations
- Rewards shifted from [0,1] to [-1,0]
- Success-based tasks

### OGBench
- Automatically downloads to `~/.ogbench/data/`
- Visual observations (84x84 RGB)
- Goal-conditioned tasks