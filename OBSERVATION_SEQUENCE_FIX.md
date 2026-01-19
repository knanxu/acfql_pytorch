# Observation Sequence Dimension Fix

## Problem Summary

When training with `obs_steps=1` and `horizon_length=4` (action chunking), the encoder was receiving observations with shape `[B, 4, C, H, W]` instead of `[B, C, H, W]`. This caused:

1. **4x unnecessary computation**: The ResNet encoder was processing all 4 timesteps through the network, even though only 1 observation was needed
2. **Dimension mismatch errors**: The ChiUNet expected input of size 256 but received 1024 (4 * 256)
3. **Slow training**: Processing 4 images per batch instead of 1

## Root Cause

The bug was in `utils/lazy_image_dataset.py` in the `sample_sequence()` method:

- **Expected behavior**: Return `'observations'` with shape `[B, C, H, W]` (initial observation only) and `'full_observations'` with shape `[B, T, C, H, W]` (full sequence)
- **Actual behavior**: Only returned `'observations'` with shape `[B, T, C, H, W]` (full sequence), missing the initial observation

This was inconsistent with the regular `Dataset` class which correctly returns both keys.

## Changes Made

### 1. Fixed LazyImageDataset.sample_sequence() (`utils/lazy_image_dataset.py`)

**Before:**
```python
batch = {
    'observations': observations,  # Shape: [B, T, C, H, W] - WRONG!
    'actions': np.stack(actions_seq, axis=0),
    'valid': np.stack(valid_mask, axis=0),
}
```

**After:**
```python
batch = {
    'observations': initial_observations,  # Shape: [B, C, H, W] - Initial obs only
    'full_observations': full_observations,  # Shape: [B, T, C, H, W] - Full sequence
    'actions': np.stack(actions_seq, axis=0),
    'valid': np.stack(valid_mask, axis=0),
}
```

### 2. Cleaned up encoder workarounds (`utils/encoders.py`)

Removed temporary workaround code that was handling the incorrect input shape by:
- Detecting 5D input when `use_seq=False`
- Flattening and then taking only the first timestep

This workaround is no longer needed since the dataset now provides the correct shape.

## Impact

### Performance Improvement
- **Before**: Processing 4 images through ResNet per batch element (1024 total for batch_size=256)
- **After**: Processing 1 image through ResNet per batch element (256 total for batch_size=256)
- **Speedup**: ~4x faster image encoding

### Memory Efficiency
- Encoder output is now `[B, emb_dim]` instead of `[B, T*emb_dim]`
- Reduced memory usage in forward pass

### Correctness
- ChiUNet now receives correct input dimensions
- No more dimension mismatch errors
- Proper separation between observation history (`obs_steps`) and action chunking (`horizon_length`)

## Verification

The fix ensures that:
1. When `obs_steps=1`, only 1 observation is encoded (not `horizon_length` observations)
2. The `'observations'` key contains initial observations: `[B, C, H, W]`
3. The `'full_observations'` key contains full sequences: `[B, T, C, H, W]`
4. This matches the behavior of the regular `Dataset` class

## Agent Support Status

All agents now support:
- ✅ **Encoder types**: `identity`, `mlp`, `image` (ResNet), `impala`
- ✅ **Multi-modal observations**: Multiple cameras + proprioceptive data
- ✅ **Action chunking**: Flexible horizon lengths

### Network Type Support by Agent

| Agent | MLP | ChiUNet | ChiTransformer | JannerUNet |
|-------|-----|---------|----------------|------------|
| FBC   | ✅  | ✅      | ✅             | ✅         |
| MFBC  | ✅  | ✅      | ✅             | ✅         |
| IMFBC | ✅  | ✅      | ✅             | ✅         |
| FQL   | ✅  | ❌      | ❌             | ❌         |

**Note**: FQL only supports MLP-based `ActorVectorField` because it requires specific flow-based architecture for its distillation mechanism.

## Files Modified

1. `utils/lazy_image_dataset.py` - Fixed `sample_sequence()` to return both `observations` and `full_observations`
2. `utils/encoders.py` - Removed workaround code for handling incorrect input shapes

## Testing

Test with the original command:
```bash
MUJOCO_GL=egl python train_bc.py --env_name=square-ph-image --agent=agents/fbc.py \
  --agent.encoder=image --agent.network_type=chiunet --agent.horizon_length=4 \
  --agent.rgb_model_name=resnet18 --agent.obs_steps=1
```

Expected behavior:
- No dimension mismatch errors
- Encoder processes only 1 image per batch element
- Training runs ~4x faster than before
