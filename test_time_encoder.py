"""Test script to verify time encoder integration."""

import torch
from utils.model import ActorVectorField, MeanActorVectorField
from utils.embeddings import get_timestep_embedding

def test_time_encoders():
    """Test different time encoder configurations."""
    print("Testing Time Encoder Integration\n")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    obs_dim = 17
    action_dim = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 1: Sinusoidal encoding (default)
    print("\n1. Testing ActorVectorField with Sinusoidal encoding")
    actor_sin = ActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        time_encoder='sinusoidal',
        time_encoder_dim=64
    ).to(device)

    obs = torch.randn(batch_size, obs_dim, device=device)
    x_t = torch.randn(batch_size, action_dim, device=device)
    t = torch.rand(batch_size, device=device)

    output = actor_sin(obs, x_t, t)
    print(f"   Input shapes: obs={obs.shape}, x_t={x_t.shape}, t={t.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {actor_sin.time_encoder_type}")
    print(f"   ✓ Sinusoidal encoding works!")

    # Test 2: Fourier encoding
    print("\n2. Testing ActorVectorField with Fourier encoding")
    actor_fourier = ActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        time_encoder='fourier',
        time_encoder_dim=64
    ).to(device)

    output = actor_fourier(obs, x_t, t)
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {actor_fourier.time_encoder_type}")
    print(f"   ✓ Fourier encoding works!")

    # Test 3: Positional encoding
    print("\n3. Testing ActorVectorField with Positional encoding")
    actor_pos = ActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        time_encoder='positional',
        time_encoder_dim=64
    ).to(device)

    output = actor_pos(obs, x_t, t)
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {actor_pos.time_encoder_type}")
    print(f"   ✓ Positional encoding works!")

    # Test 4: No time encoding
    print("\n4. Testing ActorVectorField without time encoding")
    actor_no_time = ActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        time_encoder=None,
        time_encoder_dim=64
    ).to(device)

    output = actor_no_time(obs, x_t, t)
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {actor_no_time.time_encoder_type}")
    print(f"   Use time: {actor_no_time.use_time}")
    print(f"   ✓ No time encoding works!")

    # Test 5: Legacy parameter compatibility
    print("\n5. Testing legacy parameter compatibility (use_fourier_features=True)")
    actor_legacy = ActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        use_fourier_features=True,
        fourier_feature_dim=128
    ).to(device)

    output = actor_legacy(obs, x_t, t)
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {actor_legacy.time_encoder_type}")
    print(f"   ✓ Legacy parameters work!")

    # Test 6: MeanActorVectorField with time encoders
    print("\n6. Testing MeanActorVectorField with Sinusoidal encoding")
    mean_actor = MeanActorVectorField(
        observation_dim=obs_dim,
        action_dim=action_dim,
        time_encoder='sinusoidal',
        time_encoder_dim=64
    ).to(device)

    t_begin = torch.rand(batch_size, device=device)
    t_end = torch.rand(batch_size, device=device)

    output = mean_actor(obs, x_t, t_begin, t_end)
    print(f"   Input shapes: obs={obs.shape}, x_t={x_t.shape}")
    print(f"   Time shapes: t_begin={t_begin.shape}, t_end={t_end.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Time encoder type: {mean_actor.time_encoder_type}")
    print(f"   ✓ MeanActorVectorField works!")

    # Test 7: Verify time embedding dimensions
    print("\n7. Testing different time embedding dimensions")
    for dim in [32, 64, 128, 256]:
        actor_test = ActorVectorField(
            observation_dim=obs_dim,
            action_dim=action_dim,
            time_encoder='sinusoidal',
            time_encoder_dim=dim
        ).to(device)
        output = actor_test(obs, x_t, t)
        print(f"   time_encoder_dim={dim}: output shape {output.shape} ✓")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nUsage examples:")
    print("  - Default (sinusoidal): time_encoder='sinusoidal', time_encoder_dim=64")
    print("  - Fourier features:     time_encoder='fourier', time_encoder_dim=64")
    print("  - Positional:           time_encoder='positional', time_encoder_dim=64")
    print("  - No time encoding:     time_encoder=None")
    print("\nCommand line usage:")
    print("  python train_bc.py --agent.time_encoder=fourier --agent.time_encoder_dim=128")
    print("  python train_bc.py --agent.time_encoder=None  # Disable time encoding")

if __name__ == '__main__':
    test_time_encoders()
