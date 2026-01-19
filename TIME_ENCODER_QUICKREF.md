# Time Encoder Quick Reference

## Quick Start

```bash
# Default (sinusoidal, 64-dim)
python train_bc.py --env_name=halfcheetah-medium-v2 --agent=agents/fbc.py

# Fourier encoding, 128-dim
python train_bc.py --env_name=halfcheetah-medium-v2 --agent=agents/fbc.py \
    --agent.time_encoder=fourier --agent.time_encoder_dim=128

# No time encoding
python train_bc.py --env_name=halfcheetah-medium-v2 --agent=agents/fbc.py \
    --agent.time_encoder=None
```

## Configuration Options

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `time_encoder` | `'sinusoidal'`, `'fourier'`, `'positional'`, `None` | `'sinusoidal'` | Time encoding method |
| `time_encoder_dim` | `32`, `64`, `128`, `256` | `64` | Embedding dimension |

## Time Encoder Types

| Type | Best For | Parameters | Speed |
|------|----------|------------|-------|
| **sinusoidal** | General purpose, stable training | 0 (deterministic) | Fast |
| **fourier** | High-frequency dynamics, complex tasks | ~64 (buffer) | Fast |
| **positional** | Learnable representations | 0 (maps to sinusoidal) | Fast |
| **None** | One-step policies, debugging | 0 | Fastest |

## Command Line Examples

```bash
# Sinusoidal (default)
--agent.time_encoder=sinusoidal --agent.time_encoder_dim=64

# Fourier features
--agent.time_encoder=fourier --agent.time_encoder_dim=128

# Disable time encoding
--agent.time_encoder=None

# Legacy (still works)
--agent.use_fourier_features=True --agent.fourier_feature_dim=64
```

## Python API

```python
from utils.model import ActorVectorField

# Create actor with time encoder
actor = ActorVectorField(
    observation_dim=17,
    action_dim=6,
    time_encoder='sinusoidal',  # or 'fourier', 'positional', None
    time_encoder_dim=64,
)

# Forward pass
output = actor(observations, noisy_actions, timestep)
```

## Config File

```python
import ml_collections

config = ml_collections.ConfigDict(dict(
    time_encoder='sinusoidal',  # Time encoding type
    time_encoder_dim=64,        # Embedding dimension
))
```

## Testing

```bash
# Run tests
python test_time_encoder.py

# Expected: All tests pass ✓
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unknown embedding type" | Use: `'sinusoidal'`, `'fourier'`, `'positional'`, or `None` |
| Dimension mismatch | Ensure `time_encoder_dim` is even |
| Poor performance | Try `time_encoder='fourier'` with larger `time_encoder_dim` |

## Files

- **Implementation**: `utils/model.py`, `utils/embeddings.py`
- **Factory**: `utils/network_factory.py`
- **Config**: `agents/fbc.py`
- **Tests**: `test_time_encoder.py`
- **Docs**: `TIME_ENCODER_GUIDE.md`, `TIME_ENCODER_CHANGES.md`

## Key Points

✅ **Configurable**: Choose encoding type and dimension
✅ **Backward compatible**: Legacy parameters still work
✅ **Easy to use**: Simple command-line configuration
✅ **Well tested**: All tests pass
✅ **Documented**: Complete guides available

## More Info

See `TIME_ENCODER_GUIDE.md` for detailed documentation.
