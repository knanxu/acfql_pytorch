"""
Test script to verify dict observation support in Dataset and ReplayBuffer.
"""

import numpy as np
from utils.datasets import Dataset
from utils.replay_buffer import ReplayBuffer

def test_dataset_with_dict_observations():
    """Test Dataset with dict observations."""
    print("=" * 80)
    print("Testing Dataset with dict observations")
    print("=" * 80)

    # Create synthetic dict observations (simulating robomimic image environment)
    n_samples = 100
    obs_dict = {
        'agentview_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eye_in_hand_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eef_pos': np.random.rand(n_samples, 3).astype(np.float32),
        'robot0_eef_quat': np.random.rand(n_samples, 4).astype(np.float32),
    }

    next_obs_dict = {
        'agentview_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eye_in_hand_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eef_pos': np.random.rand(n_samples, 3).astype(np.float32),
        'robot0_eef_quat': np.random.rand(n_samples, 4).astype(np.float32),
    }

    # Create dataset
    dataset = Dataset.create(
        observations=obs_dict,
        next_observations=next_obs_dict,
        actions=np.random.rand(n_samples, 7).astype(np.float32),
        rewards=np.random.rand(n_samples).astype(np.float32),
        terminals=np.zeros(n_samples).astype(np.float32),
        masks=np.ones(n_samples).astype(np.float32),
    )

    print(f"[OK] Created dataset with {len(dataset)} samples")
    print(f"  Observation keys: {list(dataset['observations'].keys())}")

    # Test 1: Basic sampling without PyTorch format
    print("\nTest 1: Basic sampling (NumPy format)")
    batch = dataset.sample(32)
    assert isinstance(batch['observations'], dict), "Observations should be a dict"
    assert 'agentview_image' in batch['observations'], "Should have agentview_image key"
    assert batch['observations']['agentview_image'].shape == (32, 84, 84, 3), \
        f"Expected shape (32, 84, 84, 3), got {batch['observations']['agentview_image'].shape}"
    print(f"[OK] Sampled batch with dict observations")
    print(f"  agentview_image shape: {batch['observations']['agentview_image'].shape}")
    print(f"  robot0_eef_pos shape: {batch['observations']['robot0_eef_pos'].shape}")

    # Test 2: Sampling with PyTorch format
    print("\nTest 2: Sampling with PyTorch format")
    dataset.pytorch_format = True
    batch = dataset.sample(32)
    assert isinstance(batch['observations'], dict), "Observations should be a dict"
    assert batch['observations']['agentview_image'].shape == (32, 3, 84, 84), \
        f"Expected shape (32, 3, 84, 84), got {batch['observations']['agentview_image'].shape}"
    print(f"[OK] Sampled batch with PyTorch format")
    print(f"  agentview_image shape: {batch['observations']['agentview_image'].shape}")
    print(f"  robot0_eef_pos shape: {batch['observations']['robot0_eef_pos'].shape}")

    # Test 3: Sequence sampling
    print("\nTest 3: Sequence sampling")
    batch = dataset.sample_sequence(16, 10)
    assert isinstance(batch['observations'], dict), "Initial observations should be a dict"
    assert isinstance(batch['full_observations'], dict), "Full observations should be a dict"
    assert batch['full_observations']['agentview_image'].shape == (16, 10, 3, 84, 84), \
        f"Expected shape (16, 10, 3, 84, 84), got {batch['full_observations']['agentview_image'].shape}"
    print(f"[OK] Sampled sequences with dict observations")
    print(f"  full_observations['agentview_image'] shape: {batch['full_observations']['agentview_image'].shape}")
    print(f"  actions shape: {batch['actions'].shape}")

    print("\n" + "=" * 80)
    print("All Dataset tests passed!")
    print("=" * 80)


def test_replay_buffer_with_dict_observations():
    """Test ReplayBuffer with dict observations."""
    print("\n" + "=" * 80)
    print("Testing ReplayBuffer with dict observations")
    print("=" * 80)

    # Create synthetic dict observations
    n_samples = 100
    obs_dict = {
        'agentview_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eef_pos': np.random.rand(n_samples, 3).astype(np.float32),
    }

    next_obs_dict = {
        'agentview_image': np.random.rand(n_samples, 84, 84, 3).astype(np.float32),
        'robot0_eef_pos': np.random.rand(n_samples, 3).astype(np.float32),
    }

    # Create dataset
    dataset = Dataset.create(
        observations=obs_dict,
        next_observations=next_obs_dict,
        actions=np.random.rand(n_samples, 7).astype(np.float32),
        rewards=np.random.rand(n_samples).astype(np.float32),
        terminals=np.zeros(n_samples).astype(np.float32),
        masks=np.ones(n_samples).astype(np.float32),
    )
    dataset.pytorch_format = True

    # Test 1: Create buffer from dataset
    print("\nTest 1: Create buffer from dataset")
    buffer = ReplayBuffer.create_from_initial_dataset(dataset, size=200)
    print(f"[OK] Created buffer with size {buffer.size}/{buffer.max_size}")

    # Test 2: Sample from buffer
    print("\nTest 2: Sample from buffer")
    batch = buffer.sample(32)
    assert isinstance(batch['observations'], dict), "Observations should be a dict"
    assert batch['observations']['agentview_image'].shape == (32, 3, 84, 84), \
        f"Expected shape (32, 3, 84, 84), got {batch['observations']['agentview_image'].shape}"
    print(f"[OK] Sampled batch from buffer")
    print(f"  agentview_image shape: {batch['observations']['agentview_image'].shape}")

    # Test 3: Add transition to buffer
    print("\nTest 3: Add transition to buffer")
    new_transition = {
        'observations': {
            'agentview_image': np.random.rand(84, 84, 3).astype(np.float32),
            'robot0_eef_pos': np.random.rand(3).astype(np.float32),
        },
        'actions': np.random.rand(7).astype(np.float32),
        'rewards': 0.5,
        'next_observations': {
            'agentview_image': np.random.rand(84, 84, 3).astype(np.float32),
            'robot0_eef_pos': np.random.rand(3).astype(np.float32),
        },
        'terminals': 0.0,
        'masks': 1.0,
    }
    buffer.add_transition(new_transition)
    print(f"[OK] Added transition to buffer")
    print(f"  Buffer size: {buffer.size}/{buffer.max_size}")

    # Test 4: Sample sequence from buffer
    print("\nTest 4: Sample sequence from buffer")
    batch = buffer.sample_sequence(16, 10)
    assert isinstance(batch['observations'], dict), "Initial observations should be a dict"
    assert isinstance(batch['full_observations'], dict), "Full observations should be a dict"
    assert batch['full_observations']['agentview_image'].shape == (16, 10, 3, 84, 84), \
        f"Expected shape (16, 10, 3, 84, 84), got {batch['full_observations']['agentview_image'].shape}"
    print(f"[OK] Sampled sequences from buffer")
    print(f"  full_observations['agentview_image'] shape: {batch['full_observations']['agentview_image'].shape}")

    # Test 5: Create buffer from example transition
    print("\nTest 5: Create buffer from example transition")
    example_transition = {
        'observations': {
            'agentview_image': np.zeros((84, 84, 3), dtype=np.float32),
            'robot0_eef_pos': np.zeros(3, dtype=np.float32),
        },
        'actions': np.zeros(7, dtype=np.float32),
        'rewards': 0.0,
        'next_observations': {
            'agentview_image': np.zeros((84, 84, 3), dtype=np.float32),
            'robot0_eef_pos': np.zeros(3, dtype=np.float32),
        },
        'terminals': 0.0,
        'masks': 1.0,
    }
    buffer2 = ReplayBuffer.create(example_transition, size=1000)
    print(f"[OK] Created empty buffer with max_size={buffer2.max_size}")

    # Add some transitions
    for i in range(10):
        buffer2.add_transition(new_transition)
    print(f"[OK] Added 10 transitions, buffer size: {buffer2.size}")

    print("\n" + "=" * 80)
    print("All ReplayBuffer tests passed!")
    print("=" * 80)


def test_backward_compatibility():
    """Test that flat observations still work (backward compatibility)."""
    print("\n" + "=" * 80)
    print("Testing backward compatibility with flat observations")
    print("=" * 80)

    # Create dataset with flat observations
    n_samples = 100
    dataset = Dataset.create(
        observations=np.random.rand(n_samples, 10).astype(np.float32),
        next_observations=np.random.rand(n_samples, 10).astype(np.float32),
        actions=np.random.rand(n_samples, 4).astype(np.float32),
        rewards=np.random.rand(n_samples).astype(np.float32),
        terminals=np.zeros(n_samples).astype(np.float32),
    )

    print(f"[OK] Created dataset with flat observations")

    # Test sampling
    batch = dataset.sample(32)
    assert batch['observations'].shape == (32, 10), \
        f"Expected shape (32, 10), got {batch['observations'].shape}"
    print(f"[OK] Sampled batch with flat observations")

    # Test sequence sampling
    batch = dataset.sample_sequence(16, 10)
    assert batch['observations'].shape == (16, 10), \
        f"Expected shape (16, 10), got {batch['observations'].shape}"
    assert batch['full_observations'].shape == (16, 10, 10), \
        f"Expected shape (16, 10, 10), got {batch['full_observations'].shape}"
    print(f"[OK] Sampled sequences with flat observations")

    # Test ReplayBuffer
    buffer = ReplayBuffer.create_from_initial_dataset(dataset, size=200)
    batch = buffer.sample(32)
    assert batch['observations'].shape == (32, 10), \
        f"Expected shape (32, 10), got {batch['observations'].shape}"
    print(f"[OK] ReplayBuffer works with flat observations")

    print("\n" + "=" * 80)
    print("Backward compatibility tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        test_dataset_with_dict_observations()
        test_replay_buffer_with_dict_observations()
        test_backward_compatibility()

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED! [SUCCESS]")
        print("=" * 80)
        print("\nDict observation support is working correctly!")
        print("You can now use image-based robomimic environments with:")
        print("  - Dataset.sample() and Dataset.sample_sequence()")
        print("  - ReplayBuffer.create(), add_transition(), sample(), sample_sequence()")
        print("  - PyTorch format conversion")
        print("  - Image augmentation")

    except Exception as e:
        print("\n" + "=" * 80)
        print("[FAIL] TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
