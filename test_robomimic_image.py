"""Test script for robomimic image environment utilities."""

import os
import sys

# Test imports
print("Testing imports...")
try:
    from envs.robomimic_image_utils import (
        get_shape_meta_from_dataset,
        make_env_and_dataset_image,
        RobomimicImageWrapper
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test shape_meta extraction
print("\nTesting shape_meta extraction...")
dataset_path = os.path.expanduser("~/.robomimic/lift/mh/low_dim_v141.hdf5")

if os.path.exists(dataset_path):
    try:
        shape_meta = get_shape_meta_from_dataset(
            dataset_path,
            obs_keys=None,
            use_eye_in_hand=True
        )
        print("✓ Shape meta extraction successful")
        print(f"  Action shape: {shape_meta['action']['shape']}")
        print(f"  Observation keys: {list(shape_meta['obs'].keys())}")

        for key, value in shape_meta['obs'].items():
            print(f"    {key}: shape={value['shape']}, type={value['type']}")
    except Exception as e:
        print(f"✗ Shape meta extraction failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠ Dataset not found at {dataset_path}")
    print("  Please download robomimic dataset to test fully")

# Test environment creation (if dataset exists)
print("\nTesting environment creation...")
if os.path.exists(dataset_path):
    try:
        env, eval_env, dataset, shape_meta = make_env_and_dataset_image(
            "lift-mh-image",
            seed=0,
            image_size=84,
            use_eye_in_hand=True
        )
        print("✓ Environment creation successful")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Dataset size: {len(dataset)}")

        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys())}")

        for key, value in obs.items():
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")

    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Skipping environment test (dataset not found)")

print("\n" + "="*50)
print("Test complete!")
print("="*50)
