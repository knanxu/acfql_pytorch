import argparse
import numpy as np
import gymnasium as gym
import mujoco

# 兼容不同版本的 MoviePy

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cube-triple-play-singletask-v0')
    parser.add_argument('--output_file', type=str, default='vis_multi_trajectory.mp4')
    parser.add_argument('--num_episodes', type=int, default=5, help='要回放的轨迹数量')
    args = parser.parse_args()

    print(f"Initializing environment: {args.env_name}...")

    # 1. 创建环境 (必须开启 add_info=True 以获取 qpos/qvel)
    env, _, train_dataset_dict, _ = make_ogbench_env_and_datasets(
        args.env_name,
        add_info=True
    )
    
    dataset = Dataset(train_dataset_dict)
    num_trajectories = len(dataset.initial_locs)
    print(f"Dataset loaded. Total trajectories available: {num_trajectories}")

    # 2. 获取底层 MuJoCo 对象
    # 注意：env.unwrapped 用于直接访问底层 MuJoCo 环境
    model = env.unwrapped.model
    
    # --- 修改离屏渲染缓冲区大小 (防止 ValueError) ---
    height = 480
    width = 640
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    # ---------------------------------------------

    frames = []

    print(f"Starting rendering loop for {args.num_episodes} episodes...")
    
    # 使用 Renderer 上下文
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        
        for ep_num in range(args.num_episodes):
            # 3. 随机采样一条轨迹
            traj_idx = np.random.randint(num_trajectories)
            start_idx = dataset.initial_locs[traj_idx]
            end_idx = dataset.terminal_locs[traj_idx]
            length = end_idx - start_idx + 1
            
            print(f"[{ep_num+1}/{args.num_episodes}] Replaying Trajectory #{traj_idx} (Steps: {length})...")
            
            # 获取动作序列
            actions = dataset['actions'][start_idx : end_idx + 1]

            # 4. 重置环境并设置初始状态
            env.reset()
            # 重置后，重新获取 data 引用是个好习惯，确保它是最新的
            data = env.unwrapped.data
            
            if 'qpos' in dataset and hasattr(env.unwrapped, 'set_state'):
                init_qpos = dataset['qpos'][start_idx]
                init_qvel = dataset['qvel'][start_idx]
                env.unwrapped.set_state(init_qpos, init_qvel)
            else:
                print("Warning: Could not set initial state (missing qpos or set_state).")

            # 5. 执行动作并渲染
            for i, action in enumerate(actions):
                env.step(action)
                
                # 同步物理状态到渲染器
                renderer.update_scene(data)
                
                # 获取图像帧
                pixels = renderer.render()
                frames.append(pixels)
            
            # 6. 在每条轨迹之间添加一段黑屏过渡 (约0.5秒)
            # 这样你在看视频时能明显知道换下一条轨迹了
            if ep_num < args.num_episodes - 1:
                black_frame = np.zeros_like(frames[-1])
                for _ in range(15): # 假设 30fps，15帧就是0.5秒
                    frames.append(black_frame)

    # 7. 生成最终视频
    print(f"Generating video with MoviePy ({len(frames)} frames)...")
    fps = env.metadata.get('render_fps', 30)
    
    try:
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(args.output_file, logger='bar')
        print(f"\nSuccess! Video saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving video: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()