# Time Encoder Integration Guide

This guide explains the time encoder integration in the acfql_pytorch codebase.

## Overview

Time encoders are used in flow matching models to condition the velocity field prediction on the diffusion timestep `t ∈ [0, 1]`. The codebase now supports multiple configurable time encoding methods from `utils/embeddings.py`.

## Supported Time Encoders

### 1. Sinusoidal Embedding (Default)
- **Type**: `time_encoder='sinusoidal'`
- **Description**: Uses sinusoidal positional embeddings (similar to Transformer)
- **Formula**: `emb = [sin(t * freq), cos(t * freq)]` for different frequencies
- **Best for**: General purpose, stable training
- **Parameters**: `max_period=10000` (default)

### 2. Fourier Embedding
- **Type**: `time_encoder='fourier'`
- **Description**: Random Fourier features with learnable frequencies
- **Formula**: `emb = [cos(2π * t * freq), sin(2π * t * freq)]` with random frequencies
- **Best for**: High-frequency time variations, complex dynamics
- **Parameters**: `scale=16.0` (default)

### 3. Positional Embedding
- **Type**: `time_encoder='positional'`
- **Description**: Learnable positional embeddings (currently maps to sinusoidal)
- **Best for**: When you want the model to learn time representations
- **Note**: Currently implemented as sinusoidal, can be extended to learnable

### 4. No Time Encoding
- **Type**: `time_encoder=None`
- **Description**: Disables time conditioning (raw timestep value or no time input)
- **Best for**: One-step distillation policies, debugging
- **Note**: Model becomes time-independent

## Configuration

### Agent Config (agents/fbc.py)

```python
config = ml_collections.ConfigDict(
    dict(
        # New parameters (recommended)
        time_encoder='sinusoidal',  # Options: 'sinusoidal', 'fourier', 'positional', None
        time_encoder_dim=64,        # Embedding dimension (32, 64, 128, 256)

        # Legacy parameters (still supported for backward compatibility)
        use_fourier_features=False,  # If True, overrides time_encoder to 'fourier'
        fourier_feature_dim=64,      # If use_fourier_features=True, sets time_encoder_dim
        disable_time_embedding=False, # If True, sets time_encoder=None
    )
)
```

### Command Line Usage

```bash
# Use Fourier encoding with 128-dim embeddings
python train_bc.py --env_name=halfcheetah-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=fourier \
    --agent.time_encoder_dim=128

# Use sinusoidal encoding (default)
python train_bc.py --env_name=halfcheetah-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=sinusoidal \
    --agent.time_encoder_dim=64

# Disable time encoding
python train_bc.py --env_name=halfcheetah-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=None

# Legacy: Use Fourier features (backward compatible)
python train_bc.py --env_name=halfcheetah-medium-v2 \
    --agent=agents/fbc.py \
    --agent.use_fourier_features=True \
    --agent.fourier_feature_dim=128
```

## Implementation Details

### ActorVectorField (utils/model.py)

The `ActorVectorField` class now supports configurable time encoders:

```python
actor = ActorVectorField(
    observation_dim=17,
    action_dim=6,
    time_encoder='sinusoidal',  # Time encoder type
    time_encoder_dim=64,        # Embedding dimension
)

# Forward pass
output = actor(observations, noisy_actions, timestep)
```

**Key features:**
- Automatically creates time encoder from `utils/embeddings.py`
- Handles legacy parameters (`use_fourier_features`, `fourier_feature_dim`)
- Supports disabling time encoding with `time_encoder=None`
- Time embeddings are concatenated with observations and actions

### MeanActorVectorField (utils/model.py)

For JVP-based flow matching (MFBC/IMFBC agents):

```python
actor = MeanActorVectorField(
    observation_dim=17,
    action_dim=6,
    time_encoder='fourier',
    time_encoder_dim=64,
)

# Forward pass with t_begin and t_end
output = actor(observations, noisy_actions, t_begin, t_end)
```

**Key features:**
- Uses **two separate time encoders** for `t_begin` and `t_end`
- Both encoders use the same type and dimension
- Enables efficient JVP computation for mean flow matching

### Network Factory (utils/network_factory.py)

The factory automatically handles time encoder configuration:

```python
# Reads from config
time_encoder = config.get('time_encoder', 'sinusoidal')
time_encoder_dim = config.get('time_encoder_dim', 64)

# Handles legacy parameters
if config.get('disable_time_embedding', False):
    time_encoder = None
elif config.get('use_fourier_features', False):
    time_encoder = 'fourier'
    time_encoder_dim = config.get('fourier_feature_dim', 64)

# Creates actor with time encoder
actor = ActorVectorField(
    observation_dim=obs_dim,
    action_dim=act_dim,
    time_encoder=time_encoder,
    time_encoder_dim=time_encoder_dim,
)
```

## Architecture Changes

### Input Dimensions

The time encoder affects the MLP input dimension:

**Without time encoding:**
```
input_dim = obs_dim + action_dim
```

**With time encoding:**
```
input_dim = obs_dim + action_dim + time_encoder_dim
```

**For MeanActorVectorField:**
```
input_dim = obs_dim + action_dim + time_encoder_dim * 2  # Two time embeddings
```

### Time Embedding Module

Located in `utils/embeddings.py`:

```python
from utils.embeddings import get_timestep_embedding

# Create time encoder
time_encoder = get_timestep_embedding('sinusoidal', dim=64)

# Use in forward pass
t = torch.rand(batch_size)  # Timesteps in [0, 1]
time_emb = time_encoder(t)  # Shape: (batch_size, 64)
```

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Legacy parameters still work:**
   - `use_fourier_features=True` → automatically sets `time_encoder='fourier'`
   - `fourier_feature_dim=128` → automatically sets `time_encoder_dim=128`
   - `disable_time_embedding=True` → automatically sets `time_encoder=None`

2. **Default behavior unchanged:**
   - Default is `time_encoder='sinusoidal'` with `time_encoder_dim=64`
   - Matches previous behavior when `use_fourier_features=False`

3. **Old checkpoints compatible:**
   - Models saved with old code can be loaded
   - Time encoder parameters are inferred from model architecture

## Performance Considerations

### Time Encoder Dimension

- **32-dim**: Faster, less expressive, good for simple tasks
- **64-dim**: Default, good balance for most tasks
- **128-dim**: More expressive, better for complex dynamics
- **256-dim**: Maximum expressiveness, may overfit on small datasets

### Encoding Type

- **Sinusoidal**: Most stable, good default choice
- **Fourier**: Better for high-frequency dynamics, slightly more parameters
- **None**: Fastest, but loses time conditioning (only for special cases)

## Examples

### Example 1: Standard BC Training

```bash
python train_bc.py \
    --env_name=halfcheetah-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=sinusoidal \
    --agent.time_encoder_dim=64
```

### Example 2: High-Frequency Dynamics

```bash
python train_bc.py \
    --env_name=hopper-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=fourier \
    --agent.time_encoder_dim=128
```

### Example 3: One-Step Distillation

```bash
python train_bc.py \
    --env_name=walker2d-medium-v2 \
    --agent=agents/fbc.py \
    --agent.time_encoder=None \
    --agent.flow_steps=1
```

### Example 4: IMFBC with Fourier Features

```bash
python train_bc.py \
    --env_name=lift-mh-low_dim \
    --agent=agents/imfbc.py \
    --agent.time_encoder=fourier \
    --agent.time_encoder_dim=64 \
    --agent.horizon_length=10
```

## Testing

Run the test script to verify time encoder integration:

```bash
python test_time_encoder.py
```

This tests:
- All time encoder types (sinusoidal, fourier, positional, None)
- Different embedding dimensions (32, 64, 128, 256)
- Legacy parameter compatibility
- Both ActorVectorField and MeanActorVectorField

## Troubleshooting

### Issue: "Unknown embedding type"
**Solution**: Check that `time_encoder` is one of: `'sinusoidal'`, `'fourier'`, `'positional'`, or `None`

### Issue: Dimension mismatch error
**Solution**: Ensure `time_encoder_dim` is even (required for sin/cos pairs)

### Issue: Legacy parameters not working
**Solution**: The new parameters override legacy ones. Use either new or legacy, not both.

### Issue: Model performance degraded
**Solution**: Try different time encoders or adjust `time_encoder_dim`. Fourier encoding may help with complex dynamics.

## References

- **Embeddings module**: `utils/embeddings.py`
- **Actor models**: `utils/model.py` (ActorVectorField, MeanActorVectorField)
- **Network factory**: `utils/network_factory.py`
- **Agent config**: `agents/fbc.py` (get_config function)
- **Test script**: `test_time_encoder.py`

## Summary

The time encoder integration provides:
- ✅ Configurable time encoding types (sinusoidal, fourier, positional, none)
- ✅ Adjustable embedding dimensions (32-256)
- ✅ Full backward compatibility with legacy parameters
- ✅ Easy command-line configuration
- ✅ Support for both standard and JVP-based flow matching
- ✅ Comprehensive testing and documentation
