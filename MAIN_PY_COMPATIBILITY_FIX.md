# main.py Compatibility Fixes

## Summary

Updated `main.py` to be compatible with the current codebase, specifically to support image-based environments with dict observations.

## Changes Made

### 1. Added `batch_to_torch()` Helper Function

**Location:** After `set_seed()` function (line 103-113)

**Purpose:** Recursively converts numpy arrays to torch tensors, handling nested dict observations.

```python
def batch_to_torch(batch):
    """Convert batch to torch tensors, handling dict observations."""
    def to_torch(x):
        if isinstance(x, dict):
            return {k: to_torch(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        else:
            return x

    return {k: to_torch(v) for k, v in batch.items()}
```

### 2. Fixed Observation Shape Detection (Line 167-187)

**Before:**
```python
example_batch = train_dataset.sample(1)
observation_shape = example_batch['observations'].shape[1:]  # Fails for dict obs!
action_dim = example_batch['actions'].shape[-1]
```

**After:**
```python
example_batch = train_dataset.sample(1)

# Handle dict observations (image environments) vs tensor observations
if isinstance(example_batch['observations'], dict):
    # Image environment with dict observations - use shape_meta
    observation_shape = env.shape_meta if hasattr(env, 'shape_meta') else None
    if observation_shape is None:
        # Try to get from dataset
        from envs.robomimic_image_utils import get_shape_meta_from_dataset
        observation_shape = get_shape_meta_from_dataset(
            dataset_path=train_dataset.dataset_path if hasattr(train_dataset, 'dataset_path') else None,
            obs_keys=list(example_batch['observations'].keys())
        )
else:
    observation_shape = example_batch['observations'].shape[1:]  # Remove batch dim

action_dim = example_batch['actions'].shape[-1]
```

### 3. Fixed Offline Training Batch Conversion (Line 261-262)

**Before:**
```python
# Convert to torch tensors
batch = {k: torch.from_numpy(v).float() for k, v in batch.items()}
```

**After:**
```python
# Convert to torch tensors
batch = batch_to_torch(batch)
```

### 4. Fixed Online Training Batch Conversion (Line 371-385)

**Before:**
```python
# Convert to torch tensors
batch = {k: torch.from_numpy(v).float() for k, v in batch.items()}

# Reshape for UTD ratio
if FLAGS.utd_ratio > 1:
    batch = {k: v.reshape(FLAGS.utd_ratio, config['batch_size'], *v.shape[1:])
             for k, v in batch.items()}
    _, update_info["online_agent"] = agent.batch_update(batch)
else:
    update_info["online_agent"] = agent.update(batch)
```

**After:**
```python
# Convert to torch tensors
batch = batch_to_torch(batch)

# Reshape for UTD ratio
if FLAGS.utd_ratio > 1:
    def reshape_for_utd(x):
        if isinstance(x, dict):
            return {k: reshape_for_utd(v) for k, v in x.items()}
        else:
            return x.reshape(FLAGS.utd_ratio, config['batch_size'], *x.shape[1:])

    batch = {k: reshape_for_utd(v) for k, v in batch.items()}
    _, update_info["online_agent"] = agent.batch_update(batch)
else:
    update_info["online_agent"] = agent.update(batch)
```

## Why These Changes Were Needed

### Problem 1: Dict Observations Not Handled

The original code assumed observations were always numpy arrays with shape `[B, obs_dim]`. However, image-based environments return dict observations like:

```python
{
    'agentview_image': array([B, C, H, W]),
    'robot0_eye_in_hand_image': array([B, C, H, W]),
    'robot0_eef_pos': array([B, 3]),
    'robot0_eef_quat': array([B, 4]),
}
```

Calling `.shape[1:]` on a dict fails with `AttributeError`.

### Problem 2: Tensor Conversion for Nested Dicts

The simple dict comprehension `{k: torch.from_numpy(v).float() for k, v in batch.items()}` doesn't handle nested dicts. When `batch['observations']` is a dict, this creates:

```python
{
    'observations': {...},  # Still a dict of numpy arrays!
    'actions': tensor(...),
    'valid': tensor(...),
}
```

The agent expects `batch['observations']` to be a dict of tensors, not numpy arrays.

### Problem 3: UTD Ratio Reshaping

When using UTD ratio > 1, the reshape operation needs to handle dict observations recursively.

## Testing

The fixed `main.py` now works with:

### State-based environments:
```bash
python main.py --env_name=halfcheetah-medium-v2 --agent=agents/fql.py \
  --offline_steps=100000 --online_steps=100000
```

### Image-based environments:
```bash
python main.py --env_name=square-ph-image --agent=agents/fql.py \
  --agent.encoder=image --agent.network_type=mlp \
  --agent.rgb_model_name=resnet18 --agent.horizon_length=4 \
  --offline_steps=100000 --online_steps=100000
```

### Low-dim robomimic environments:
```bash
python main.py --env_name=lift-mh-low_dim --agent=agents/fql.py \
  --offline_steps=100000 --online_steps=100000
```

## Compatibility

The changes are backward compatible:
- ✅ Works with state-based environments (D4RL)
- ✅ Works with image-based environments (Robomimic image)
- ✅ Works with low-dim robomimic environments
- ✅ Works with UTD ratio > 1
- ✅ Works with all agents (FQL, FBC, MFBC, IMFBC)
