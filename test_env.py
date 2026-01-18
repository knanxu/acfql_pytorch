import argparse
import numpy as np
import gymnasium as gym
import sys
import os
import h5py

# 尝试导入 MoviePy
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        print("Warning: moviepy not installed. Video generation might fail.")
        ImageSequenceClip = None

# 动态添加路径
sys.path.append(os.getcwd())

# 导入环境构建工具
try:
    from envs.env_utils import make_env_and_datasets
except ImportError:
    print("Error: Could not import envs.env_utils. Make sure you are in the correct directory.")
    sys.exit(1)

def load_robomimic_states(env_name, dataset_size):
    """专门为 Robomimic 环境手动加载物理状态(states)"""
    try:
        print("🔍 Attempting to load expert physics states from HDF5...")
        task, dataset_type, _ = env_name.split("-")
        if dataset_type == "mg":
            file_name = "low_dim_sparse_v15.hdf5"
        else:
            file_name = "low_dim_v15.hdf5"
            
        dataset_path = os.path.join(
            os.path.expanduser("~/.robomimic"), 
            task, dataset_type, file_name
        )
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset file not found: {dataset_path}")
            return None

        f = h5py.File(dataset_path, "r")
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        
        all_states = []
        for ep in demos:
            if "states" in f[f"data/{ep}"]:
                all_states.append(np.array(f[f"data/{ep}/states"]))
            else:
                return None 
        
        states_flat = np.concatenate(all_states, axis=0)
        
        if len(states_flat) != dataset_size:
            print(f"⚠️ State count mismatch: Loaded {len(states_flat)}, Dataset {dataset_size}")
        else:
            print(f"✅ Successfully loaded {len(states_flat)} state records.")
            
        return states_flat

    except Exception as e:
        print(f"⚠️ Failed to load external states: {e}")
        return None

def format_arr(arr):
    """
    格式化打印完整数组
    不再截断，显示所有维度数据
    """
    if arr is None: return "None"
    arr = np.array(arr)
    # precision=4 保证精度，suppress_small=True 避免显示极小值(如 1e-10)
    # separator=', ' 让输出更紧凑易读
    return np.array2string(arr.flatten(), precision=4, suppress_small=True, separator=', ')

def reset_env_to_state(env, dataset, idx, external_states=None):
    """统一重置逻辑"""
    env.reset()
    
    state_to_use = None
    if external_states is not None:
        state_to_use = external_states[idx]
    elif 'states' in dataset:
        state_to_use = dataset['states'][idx]
        
    if state_to_use is not None:
        try:
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'reset_to'):
                unwrapped.reset_to({'states': state_to_use})
            elif hasattr(unwrapped, 'env') and hasattr(unwrapped.env, 'reset_to'):
                unwrapped.env.reset_to({'states': state_to_use})
            elif hasattr(env, 'env') and hasattr(env.env, 'reset_to'):
                 env.env.reset_to({'states': state_to_use})
            else:
                sim = None
                curr = env
                while hasattr(curr, 'env') or hasattr(curr, 'unwrapped'):
                    if hasattr(curr, 'sim'):
                        sim = curr.sim
                        break
                    curr = curr.env if hasattr(curr, 'env') else curr.unwrapped
                
                if sim:
                    sim.set_state_from_flattened(state_to_use)
                    sim.forward()
            return True
        except Exception as e:
            print(f"Warning: Robomimic state reset failed: {e}")

    if 'qpos' in dataset and 'qvel' in dataset:
        try:
            qpos = dataset['qpos'][idx]
            qvel = dataset['qvel'][idx]
            if hasattr(env.unwrapped, 'set_state'):
                env.unwrapped.set_state(qpos, qvel)
                return True
        except Exception as e:
            print(f"Warning: set_state failed: {e}")
            
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='square-mh-low_dim', help='Environment name')
    parser.add_argument('--output_file', type=str, default='trajectory_vis.mp4', help='Output video file')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of trajectories to visualize')
    args = parser.parse_args()

    print(f"🚀 Initializing Environment: {args.env_name}")
    
    try:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            args.env_name,
            frame_stack=None
        )
    except Exception as e:
        print(f"❌ Error loading environment: {e}")
        return

    dataset = train_dataset
    print(f"✅ Loaded. Dataset size: {len(dataset.initial_locs)} trajectories.")

    external_states = None
    if "low_dim" in args.env_name:
        external_states = load_robomimic_states(args.env_name, dataset.size)

    frames = []

    for ep_i in range(args.num_episodes):
        while True:
            traj_idx = np.random.randint(len(dataset.initial_locs))
            start_idx = dataset.initial_locs[traj_idx]
            end_idx = dataset.terminal_locs[traj_idx]
            length = end_idx - start_idx + 1
            if length > 20: 
                break
        
        print(f"\n🎬 [Episode {ep_i+1}/{args.num_episodes}] Replaying Trajectory #{traj_idx} (Steps: {length})")
        print("-" * 150)
        # 修改了表头，去掉 "First 4" 的说明
        print(f"{'Step':<6} | {'Reward':<8} | {'Action (Full)':<50} | {'State (Full)':<80} | {'Next State (Full)':<80}")
        print("-" * 150)

        actions = dataset['actions'][start_idx : end_idx + 1]
        observations = dataset['observations'][start_idx : end_idx + 1]
        rewards = dataset['rewards'][start_idx : end_idx + 1]
        
        if 'next_observations' in dataset:
            next_observations = dataset['next_observations'][start_idx : end_idx + 1]
        else:
            next_observations = np.roll(observations, -1, axis=0)
            next_observations[-1] = observations[-1]

        reset_success = reset_env_to_state(env, dataset, start_idx, external_states)
        if not reset_success:
            print("⚠️  Warning: Initial state could not be set precisely. Replay might diverge!")

        for t, action in enumerate(actions):
            s_curr = observations[t]
            s_next = next_observations[t]
            r_curr = rewards[t]
            
            r_val = r_curr.item() if hasattr(r_curr, 'item') else r_curr
            
            # 使用修改后的 format_arr，打印完整数据
            # 移除了固定宽度限制，以免截断
            print(f"{t:<6} | {r_val:<8.3f} | {format_arr(action)} | {format_arr(s_curr)} | {format_arr(s_next)}")

            env.step(action)
            
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                pass

        if len(frames) > 0:
            black_frame = np.zeros_like(frames[-1])
            for _ in range(15): 
                frames.append(black_frame)

    if len(frames) > 0 and ImageSequenceClip is not None:
        print(f"\n💾 Saving video ({len(frames)} frames) to {args.output_file}...")
        
        fps = 30
        if hasattr(env, 'metadata'):
            fps = env.metadata.get('render_fps', 30)
        elif hasattr(env, 'control_freq'):
            fps = env.control_freq
            
        try:
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(args.output_file, logger='bar')
            print("✅ Video generation successful!")
        except Exception as e:
            print(f"❌ Video save failed: {e}")
    elif len(frames) == 0:
        print("❌ No frames captured.")

    env.close()

if __name__ == "__main__":
    main()