"""Test script for pretrained frozen encoder functionality."""

import torch
import torch.nn as nn
from utils.encoders import MultiImageObsEncoder, get_resnet


def test_pretrained_weights():
    """Test loading pretrained ResNet weights."""
    print("=" * 70)
    print("Test 1: Loading Pretrained Weights")
    print("=" * 70)

    # Test with pretrained weights
    print("\n1.1 Loading ResNet18 with pretrained weights...")
    resnet_pretrained = get_resnet('resnet18', weights='IMAGENET1K_V1')
    print(f"✓ Loaded pretrained ResNet18")

    # Test without pretrained weights
    print("\n1.2 Loading ResNet18 without pretrained weights...")
    resnet_random = get_resnet('resnet18', weights=None)
    print(f"✓ Loaded random ResNet18")

    # Verify weights are different
    pretrained_weight = list(resnet_pretrained.parameters())[0]
    random_weight = list(resnet_random.parameters())[0]

    weight_diff = torch.abs(pretrained_weight - random_weight).mean().item()
    print(f"\n✓ Weight difference: {weight_diff:.6f} (should be > 0)")
    assert weight_diff > 0.01, "Weights should be different!"

    print("\n" + "=" * 70)
    print("✓ Test 1 PASSED: Pretrained weights loaded correctly")
    print("=" * 70)


def test_frozen_encoder():
    """Test freezing encoder parameters."""
    print("\n" + "=" * 70)
    print("Test 2: Freezing Encoder Parameters")
    print("=" * 70)

    # Create shape_meta for multi-image encoder
    shape_meta = {
        'obs': {
            'agentview_image': {'shape': (3, 84, 84), 'type': 'rgb'},
            'robot0_eef_pos': {'shape': (3,), 'type': 'low_dim'},
        }
    }

    # Test with frozen encoder
    print("\n2.1 Creating encoder with frozen RGB backbone...")
    encoder_frozen = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        emb_dim=256,
        pretrained=True,
        freeze_rgb_encoder=True,
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in encoder_frozen.parameters())
    trainable_params = sum(p.numel() for p in encoder_frozen.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # Verify RGB encoder is frozen
    rgb_encoder_params = sum(p.numel() for p in encoder_frozen.key_model_map['agentview_image'].parameters())
    rgb_trainable = sum(p.numel() for p in encoder_frozen.key_model_map['agentview_image'].parameters() if p.requires_grad)

    print(f"\n  RGB encoder parameters: {rgb_encoder_params:,}")
    print(f"  RGB trainable: {rgb_trainable:,}")
    assert rgb_trainable == 0, "RGB encoder should be completely frozen!"
    print(f"  ✓ RGB encoder is frozen")

    # Verify MLP is trainable
    mlp_params = sum(p.numel() for p in encoder_frozen.mlp.parameters())
    mlp_trainable = sum(p.numel() for p in encoder_frozen.mlp.parameters() if p.requires_grad)

    print(f"\n  MLP parameters: {mlp_params:,}")
    print(f"  MLP trainable: {mlp_trainable:,}")
    assert mlp_trainable == mlp_params, "MLP should be completely trainable!"
    print(f"  ✓ MLP projection is trainable")

    print("\n" + "=" * 70)
    print("✓ Test 2 PASSED: Encoder freezing works correctly")
    print("=" * 70)


def test_unfreezing():
    """Test unfreezing encoder for fine-tuning."""
    print("\n" + "=" * 70)
    print("Test 3: Unfreezing Encoder for Fine-tuning")
    print("=" * 70)

    shape_meta = {
        'obs': {
            'agentview_image': {'shape': (3, 84, 84), 'type': 'rgb'},
        }
    }

    # Create frozen encoder
    print("\n3.1 Creating frozen encoder...")
    encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        emb_dim=256,
        pretrained=True,
        freeze_rgb_encoder=True,
    )

    frozen_params = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"  Frozen parameters: {frozen_params:,}")

    # Unfreeze encoder
    print("\n3.2 Unfreezing encoder...")
    encoder.unfreeze_rgb_encoder()

    unfrozen_params = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"  Frozen parameters after unfreezing: {unfrozen_params:,}")

    # Verify all parameters are trainable
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    assert trainable_params == total_params, "All parameters should be trainable after unfreezing!"
    print(f"  ✓ All parameters are trainable")

    print("\n" + "=" * 70)
    print("✓ Test 3 PASSED: Unfreezing works correctly")
    print("=" * 70)


def test_forward_pass():
    """Test forward pass with frozen encoder."""
    print("\n" + "=" * 70)
    print("Test 4: Forward Pass with Frozen Encoder")
    print("=" * 70)

    shape_meta = {
        'obs': {
            'agentview_image': {'shape': (3, 84, 84), 'type': 'rgb'},
            'robot0_eef_pos': {'shape': (3,), 'type': 'low_dim'},
        }
    }

    print("\n4.1 Creating encoder...")
    encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        emb_dim=256,
        pretrained=True,
        freeze_rgb_encoder=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    encoder.eval()

    # Create dummy input
    batch_size = 4
    obs_dict = {
        'agentview_image': torch.randn(batch_size, 3, 84, 84, device=device),
        'robot0_eef_pos': torch.randn(batch_size, 3, device=device),
    }

    print(f"\n4.2 Running forward pass (batch_size={batch_size})...")
    with torch.no_grad():
        output = encoder(obs_dict)

    print(f"  Input shapes:")
    for key, val in obs_dict.items():
        print(f"    {key}: {val.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"
    print(f"  ✓ Output shape is correct")

    # Test gradient flow
    print(f"\n4.3 Testing gradient flow...")
    encoder.train()
    obs_dict_train = {
        'agentview_image': torch.randn(batch_size, 3, 84, 84, device=device, requires_grad=True),
        'robot0_eef_pos': torch.randn(batch_size, 3, device=device, requires_grad=True),
    }

    output = encoder(obs_dict_train)
    loss = output.sum()
    loss.backward()

    # Check that MLP has gradients
    mlp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.mlp.parameters())
    assert mlp_has_grad, "MLP should have gradients!"
    print(f"  ✓ MLP receives gradients")

    # Check that RGB encoder does NOT have gradients
    rgb_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in encoder.key_model_map['agentview_image'].parameters())
    assert not rgb_has_grad, "RGB encoder should NOT have gradients!"
    print(f"  ✓ RGB encoder does not receive gradients (frozen)")

    print("\n" + "=" * 70)
    print("✓ Test 4 PASSED: Forward pass and gradient flow work correctly")
    print("=" * 70)


def test_parameter_efficiency():
    """Compare parameter counts between frozen and unfrozen."""
    print("\n" + "=" * 70)
    print("Test 5: Parameter Efficiency Comparison")
    print("=" * 70)

    shape_meta = {
        'obs': {
            'agentview_image': {'shape': (3, 84, 84), 'type': 'rgb'},
            'robot0_eef_pos': {'shape': (3,), 'type': 'low_dim'},
        }
    }

    # Frozen encoder
    print("\n5.1 Frozen encoder:")
    encoder_frozen = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        emb_dim=256,
        pretrained=True,
        freeze_rgb_encoder=True,
    )

    total_frozen = sum(p.numel() for p in encoder_frozen.parameters())
    trainable_frozen = sum(p.numel() for p in encoder_frozen.parameters() if p.requires_grad)

    print(f"  Total: {total_frozen:,}")
    print(f"  Trainable: {trainable_frozen:,}")
    print(f"  Ratio: {100 * trainable_frozen / total_frozen:.2f}%")

    # Unfrozen encoder
    print("\n5.2 Unfrozen encoder:")
    encoder_unfrozen = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        emb_dim=256,
        pretrained=True,
        freeze_rgb_encoder=False,
    )

    total_unfrozen = sum(p.numel() for p in encoder_unfrozen.parameters())
    trainable_unfrozen = sum(p.numel() for p in encoder_unfrozen.parameters() if p.requires_grad)

    print(f"  Total: {total_unfrozen:,}")
    print(f"  Trainable: {trainable_unfrozen:,}")
    print(f"  Ratio: {100 * trainable_unfrozen / total_unfrozen:.2f}%")

    # Comparison
    print(f"\n5.3 Efficiency gain:")
    reduction = 100 * (1 - trainable_frozen / trainable_unfrozen)
    print(f"  Parameter reduction: {reduction:.1f}%")
    print(f"  Speedup factor: {trainable_unfrozen / trainable_frozen:.1f}x fewer parameters to train")

    print("\n" + "=" * 70)
    print("✓ Test 5 PASSED: Parameter efficiency demonstrated")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TESTING PRETRAINED FROZEN ENCODER FUNCTIONALITY")
    print("=" * 70)

    try:
        test_pretrained_weights()
        test_frozen_encoder()
        test_unfreezing()
        test_forward_pass()
        test_parameter_efficiency()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Pretrained weights load correctly")
        print("  ✓ Encoder freezing works as expected")
        print("  ✓ Unfreezing enables fine-tuning")
        print("  ✓ Forward pass and gradients work correctly")
        print("  ✓ Significant parameter reduction achieved")
        print("\nYou can now use pretrained frozen encoders for efficient training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
