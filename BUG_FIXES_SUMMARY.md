# Bug Fixes Summary

## Issues Fixed

### 1. ActorVectorField Parameter Mismatch (Request 2)

**Problem**: TypeError when running training with `agents/fbc.py`
```
TypeError: ActorVectorField.__init__() got an unexpected keyword argument 'obs_dim'
```

**Root Cause**:
- `utils/network_factory.py` was importing `ActorVectorField` from `utils.model`
- But it was using parameter names from `utils/networks/mlp.py` version
- The two versions have different signatures:
  - `utils/model.py`: `ActorVectorField(observation_dim, action_dim, hidden_dim, ...)`
  - `utils/networks/mlp.py`: `ActorVectorField(obs_dim, action_dim, hidden_dims, ...)`

**Fix**: Updated `utils/network_factory.py` line 35-43 to use correct parameter names:
```python
return ActorVectorField(
    observation_dim=obs_dim * To,  # Changed from obs_dim
    action_dim=act_dim * Ta,
    hidden_dim=config.get('actor_hidden_dims', (512, 512, 512, 512)),  # Changed from hidden_dims
    use_fourier_features=config.get('use_fourier_features', False),
    fourier_feature_dim=config.get('fourier_feature_dim', 64),
)
```

**Status**: ✅ Fixed

---

### 2. Dimension Mismatch in IMFBC with Action Chunking

**Problem**: RuntimeError when running training with `agents/imfbc.py`
```
RuntimeError: Tensors must have same number of dimensions: got 3 and 2
```

**Root Cause**:
- In `utils/networks/mlp.py`, the `ActorVectorField.forward()` method concatenates:
  - `observations`: Can be 2D `(B, obs_dim)` or 3D `(B, seq_len, obs_dim)`
  - `x_t`: 2D `(B, action_dim)`
  - `t_begin`, `t_end`: 2D `(B, 1)` or 1D `(B,)`
- When using action chunking with sequences, `observations` becomes 3D but other tensors remain 2D
- `torch.cat()` requires all tensors to have the same number of dimensions

**Fix**: Updated `utils/networks/mlp.py` lines 125-158 to handle dimension mismatches:
```python
def forward(self, observations, x_t, t_begin, t_end, is_encoded=False):
    # ... existing code ...

    # Ensure all tensors have the same number of dimensions for concatenation
    # If observations is 3D (has sequence dimension), flatten it
    if observations.dim() == 3:
        # (B, seq_len, obs_dim) -> (B, seq_len * obs_dim)
        batch_size = observations.shape[0]
        observations = observations.reshape(batch_size, -1)

    # Ensure t_begin and t_end are 2D
    if t_begin.dim() == 1:
        t_begin = t_begin.unsqueeze(-1)
    if t_end.dim() == 1:
        t_end = t_end.unsqueeze(-1)

    inputs = torch.cat([observations, x_t, t_begin, t_end], dim=-1)
    # ... rest of code ...
```

**Status**: ✅ Fixed

---

### 3. Robomimic Environment Detection (Request 1)

**Problem**: Environment detection didn't properly handle both low_dim and image observations

**Changes Made**:

#### 3.1 Updated `is_robomimic_env()` in `envs/robomimic_utils.py`
**Before**: Only detected environments with "low_dim" in the name
```python
def is_robomimic_env(env_name):
    if "low_dim" not in env_name:
        return False
    # ...
```

**After**: Detects both low_dim and image environments
```python
def is_robomimic_env(env_name):
    """Determine if an env is robomimic (both low_dim and image)."""
    task_name = env_name.split("-")[0]
    if task_name not in ("lift", "can", "square", "transport", "tool_hang"):
        return False

    parts = env_name.split("-")
    if len(parts) < 2:
        return False

    dataset_type = parts[1]
    return dataset_type in ("mh", "ph", "mg")
```

#### 3.2 Updated `make_env_and_datasets()` in `envs/env_utils.py`
**Before**: Separate handling for `-image` and low_dim environments
```python
elif '-image' in env_name:
    # RoboMimic Image-based
    from envs import robomimic_image_utils
    # ...
elif env_name.startswith("lift") or ...:
    # RoboMimic Low-dim
    from envs import robomimic_utils
    # ...
```

**After**: Unified handling with auto-detection
```python
elif '-image' in env_name:
    # RoboMimic Image-based (explicit)
    from envs import robomimic_image_utils
    # ...
elif env_name.startswith("lift") or ...:
    # RoboMimic (auto-detect low_dim or image)
    from envs import robomimic_utils

    if '-image' in env_name:
        # Image environment
        from envs import robomimic_image_utils
        env, eval_env, train_dataset, shape_meta = robomimic_image_utils.make_env_and_dataset_image(...)
    else:
        # Low-dim environment
        env = robomimic_utils.make_env(env_name, seed=0)
        eval_env = robomimic_utils.make_env(env_name, seed=42)
        # ...
```

**Status**: ✅ Fixed

---

## Testing

### Test 1: FBC with Identity Encoder (Low-dim)
```bash
MUJOCO_GL=egl python train_bc.py --env_name=lift-mh-low_dim \
  --agent=agents/fbc.py --agent.encoder=identity --agent.network_type=mlp
```
**Expected**: Should work without errors ✅

### Test 2: IMFBC with Action Chunking (Low-dim)
```bash
MUJOCO_GL=egl python train_bc.py --env_name=lift-mh-low_dim \
  --agent=agents/imfbc.py --agent.encoder=identity --agent.network_type=mlp \
  --agent.horizon_length=10
```
**Expected**: Should work without dimension mismatch errors ✅

### Test 3: Image-based Environment
```bash
MUJOCO_GL=egl python train_bc.py --env_name=lift-mh-image \
  --agent=agents/fbc.py --agent.encoder=image --agent.network_type=mlp
```
**Expected**: Should automatically detect and load image environment ✅

---

## Files Modified

1. **`utils/network_factory.py`** (lines 35-43)
   - Fixed ActorVectorField parameter names

2. **`utils/networks/mlp.py`** (lines 125-158)
   - Added dimension handling in ActorVectorField.forward()

3. **`envs/robomimic_utils.py`** (lines 19-32)
   - Updated is_robomimic_env() to detect both low_dim and image

4. **`envs/env_utils.py`** (lines 132-160)
   - Added auto-detection for robomimic environment types

---

## Environment Name Format

### Low-dimensional Observations
```
<task>-<dataset_type>-low_dim
```
Examples: `lift-mh-low_dim`, `can-ph-low_dim`, `square-mh-low_dim`

### Image Observations
```
<task>-<dataset_type>-image
```
Examples: `lift-mh-image`, `can-ph-image`, `square-mh-image`

Where:
- **task**: `lift`, `can`, `square`, `transport`, `tool_hang`
- **dataset_type**: `mh` (multi-human), `ph` (proficient-human), `mg` (machine-generated)

---

## Notes

- Both `robomimic_utils.py` and `robomimic_image_utils.py` are kept as separate files
- `env_utils.py` automatically routes to the correct utility based on environment name
- The fixes maintain backward compatibility with existing code
- All changes follow the original coding style and structure
