# Dataset 使用指南

本指南介绍如何使用 `utils/datasets.py` 中的 `Dataset` 类进行离线强化学习数据处理。

## 目录

1. [快速开始](#快速开始)
2. [从本地加载数据](#从本地加载数据)
3. [图像格式说明](#图像格式说明)
4. [基本采样方法](#基本采样方法)
5. [序列采样详解](#序列采样详解)
6. [高级功能](#高级功能)
7. [完整示例](#完整示例)

---

## 快速开始

### 什么是 Dataset？

`Dataset` 是一个统一的离线强化学习数据接口，支持：
- **D4RL** (MuJoCo, AntMaze, Adroit 等)
- **Robomimic** (机器人操作任务)
- **OGBench** (目标条件任务)

### 基本概念

Dataset 包含以下核心字段：
- `observations`: 状态观测
- `actions`: 动作
- `rewards`: 奖励
- `next_observations`: 下一个状态
- `terminals`: 是否终止（episode 结束）
- `masks`: 是否有效（用于 bootstrapping）

---

## 从本地加载数据

### 方法 1: 使用 `env_utils.make_env_and_datasets()`（推荐）

这是最简单的方法，自动识别数据集类型并加载：

```python
from envs.env_utils import make_env_and_datasets

# 自动识别并加载环境和数据集
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

print(f"Dataset size: {len(train_dataset)}")
print(f"Available fields: {list(train_dataset.keys())}")
```

**支持的环境名称：**
- D4RL: `'halfcheetah-medium-v2'`, `'antmaze-umaze-v2'`, `'pen-human-v1'` 等
- Robomimic: `'lift-mh-low_dim'`, `'can-ph-low_dim'`, `'square-mh-low_dim'` 等
- OGBench: `'visual-singletask-task0-v0'` 等

### 方法 2: 分别加载 D4RL 数据

```python
from envs import d4rl_utils

env = d4rl_utils.make_env('halfcheetah-medium-v2')
dataset = d4rl_utils.get_dataset(env, 'halfcheetah-medium-v2')

# dataset 已经是 Dataset 对象
print(f"Dataset size: {len(dataset)}")
```

**数据位置：** D4RL 数据会自动下载到 `~/.d4rl/` 目录

### 方法 3: 加载 Robomimic 数据

```python
from envs import robomimic_utils

# 注意：需要先确保数据已下载到 ~/.robomimic/
env = robomimic_utils.make_env('lift-mh-low_dim', seed=0)
dataset = robomimic_utils.get_dataset(env, 'lift-mh-low_dim')

print(f"Dataset size: {len(dataset)}")
```

**数据位置：** Robomimic 数据应该在 `~/.robomimic/<task>/<dataset_type>/low_dim_v15.hdf5`

### 方法 4: 加载 OGBench 数据

```python
from envs import ogbench_utils

env, train_dataset, val_dataset = ogbench_utils.make_ogbench_env_and_datasets(
    'visual-singletask-task0-v0',
    dataset_dir='~/.ogbench/data'
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
```

**数据位置：** OGBench 数据会自动下载到 `~/.ogbench/data/` 目录

### 方法 5: 手动创建 Dataset

如果你有自己的数据：

```python
from utils.datasets import Dataset
import numpy as np

# 假设你已经有了这些数组
observations = np.random.randn(1000, 17)  # 1000 个样本，17 维状态
actions = np.random.randn(1000, 6)         # 6 维动作
rewards = np.random.randn(1000, 1)
next_observations = np.random.randn(1000, 17)
terminals = np.zeros((1000, 1))
terminals[99::100] = 1  # 每 100 步终止一次

# 创建 Dataset
dataset = Dataset.create(
    observations=observations,
    actions=actions,
    rewards=rewards,
    next_observations=next_observations,
    terminals=terminals
)

print(f"Dataset size: {len(dataset)}")
```

---

## 图像格式说明

### NumPy 格式 vs PyTorch 格式

在处理视觉观测（图像）时，`Dataset` 类支持两种格式：

#### 1. **NumPy 格式（默认）** - Channels Last

这是 NumPy、OpenCV、PIL 等库的标准格式：

```python
# 数据集内部存储格式
单张图像: (H, W, C)     # 高度 × 宽度 × 通道数
批量图像: (B, H, W, C)  # batch_size × 高度 × 宽度 × 通道数
序列图像: (B, T, H, W, C)  # batch_size × 时间步 × 高度 × 宽度 × 通道数

# 示例：84x84 的 RGB 图像
单张: (84, 84, 3)
批量: (256, 84, 84, 3)
序列: (32, 10, 84, 84, 3)
```

**优点：**
- 与 NumPy/OpenCV/PIL 直接兼容
- 便于可视化和调试
- 不需要转换即可使用 matplotlib 显示

#### 2. **PyTorch 格式** - Channels First

这是 PyTorch 和大多数深度学习框架的标准格式：

```python
# PyTorch 期望的格式
单张图像: (C, H, W)     # 通道数 × 高度 × 宽度
批量图像: (B, C, H, W)  # batch_size × 通道数 × 高度 × 宽度
序列图像: (B, T, C, H, W)  # batch_size × 时间步 × 通道数 × 高度 × 宽度

# 示例：84x84 的 RGB 图像
单张: (3, 84, 84)
批量: (256, 3, 84, 84)
序列: (32, 10, 3, 84, 84)
```

**优点：**
- PyTorch 卷积层的标准输入格式
- GPU 加速计算效率更高
- 大多数预训练模型期望的格式

### 如何使用

#### 方法 1: 自动转换（推荐）

设置 `dataset.pytorch_format = True`，自动转换为 PyTorch 格式：

```python
from envs.env_utils import make_env_and_datasets

# 加载数据集
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('visual-singletask-task0-v0')

# 启用 PyTorch 格式
train_dataset.pytorch_format = True

# 采样 - 自动返回 PyTorch 格式
batch = train_dataset.sample(batch_size=256)
print(batch['observations'].shape)
# 输出: (256, 3, 84, 84)  # (B, C, H, W)

# 序列采样 - 也会自动转换
seq_batch = train_dataset.sample_sequence(batch_size=32, sequence_length=10)
print(seq_batch['full_observations'].shape)
# 输出: (32, 10, 3, 84, 84)  # (B, T, C, H, W)
```

#### 方法 2: 手动转换

保持默认的 NumPy 格式，在转换为 tensor 时手动调整：

```python
import torch

# 不设置 pytorch_format（默认 False）
batch = train_dataset.sample(batch_size=256)
print(batch['observations'].shape)
# 输出: (256, 84, 84, 3)  # (B, H, W, C) - NumPy 格式

# 转换为 PyTorch tensor 并调整维度
observations = torch.from_numpy(batch['observations']).float()
observations = observations.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
print(observations.shape)
# 输出: torch.Size([256, 3, 84, 84])
```

### 格式转换示例

`Dataset` 类提供了自动识别和转换的功能：

```python
import numpy as np

# 创建测试数据
images = np.random.randint(0, 255, (100, 84, 84, 3), dtype=np.uint8)  # NumPy 格式
dataset = Dataset.create(
    observations=images,
    actions=np.random.randn(100, 6),
    rewards=np.random.randn(100, 1),
    next_observations=images,
    terminals=np.zeros((100, 1))
)

# 默认返回 NumPy 格式
batch1 = dataset.sample(32)
print(f"默认格式: {batch1['observations'].shape}")
# 输出: 默认格式: (32, 84, 84, 3)

# 启用 PyTorch 格式
dataset.pytorch_format = True
batch2 = dataset.sample(32)
print(f"PyTorch 格式: {batch2['observations'].shape}")
# 输出: PyTorch 格式: (32, 3, 84, 84)
```

### 重要注意事项

⚠️ **数据增强（augmentation）总是在 NumPy 格式下进行**

```python
# 数据增强流程
dataset.p_aug = 0.5
dataset.pytorch_format = True

batch = dataset.sample(256)
# 内部流程：
# 1. 采样数据 (NumPy 格式)
# 2. 应用随机裁剪增强 (在 NumPy 格式下)
# 3. 转换为 PyTorch 格式 (如果 pytorch_format=True)
# 4. 返回结果
```

⚠️ **帧堆叠（frame stacking）会增加通道数**

```python
# 设置帧堆叠
dataset.frame_stack = 4
dataset.pytorch_format = True

batch = dataset.sample(256)
print(batch['observations'].shape)
# 输出: (256, 12, 84, 84)  # 3 channels × 4 frames = 12 channels
```

### 性能考虑

**什么时候使用 `pytorch_format=True`？**

✅ **推荐使用：**
- 训练视觉任务的神经网络
- 使用 PyTorch 的 Conv2d/Conv3d 层
- 与预训练的视觉模型集成
- 需要 GPU 加速的图像处理

❌ **不推荐使用：**
- 只是查看或可视化数据
- 使用非 PyTorch 框架
- 状态向量（非图像）数据

**转换开销：**
- 格式转换是零拷贝操作（使用 `transpose`），非常快
- 对训练性能影响可忽略不计

---

## 基本采样方法

### 随机采样单个转换 (Transitions)

```python
# 采样一个 batch
batch_size = 256
batch = dataset.sample(batch_size)

print("Batch keys:", batch.keys())
# 输出: dict_keys(['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'masks'])

print(f"Observations shape: {batch['observations'].shape}")
# 输出: Observations shape: (256, 17)  # (batch_size, obs_dim)

print(f"Actions shape: {batch['actions'].shape}")
# 输出: Actions shape: (256, 6)  # (batch_size, action_dim)

print(f"Rewards shape: {batch['rewards'].shape}")
# 输出: Rewards shape: (256, 1)  # (batch_size, 1)
```

### 转换为 PyTorch 张量

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 方法 1: 使用 to_tensors 方法
states, actions, next_states, rewards, not_done = dataset.to_tensors(
    batch_size=256,
    device=device
)

print(f"States shape: {states.shape}, device: {states.device}")
# 输出: States shape: torch.Size([256, 17]), device: cuda:0

# 方法 2: 手动转换
batch = dataset.sample(256)
states = torch.from_numpy(batch['observations']).float().to(device)
actions = torch.from_numpy(batch['actions']).float().to(device)
```

---

## 序列采样详解

### 什么是序列采样？

`sample_sequence()` 用于采样**连续的时间序列**，这对于以下方法非常有用：
- **Transformer-based 方法** (Decision Transformer, Trajectory Transformer)
- **RNN-based 方法** (LSTM, GRU)
- **时序模型** (需要上下文的任何方法)

### 基本用法

```python
batch_size = 32
sequence_length = 10  # 每个序列包含 10 个连续时间步

batch = dataset.sample_sequence(
    batch_size=batch_size,
    sequence_length=sequence_length,
    discount=0.99
)
```

### 返回数据的形状详解

`sample_sequence()` 返回一个字典，包含以下字段：

#### 1. `observations` - 初始观测
```python
print(batch['observations'].shape)
# 状态向量: (32, 17) - (batch_size, obs_dim)
# 含义: 每个序列的起始观测 (t=0 时刻)

# 图像（NumPy 格式，pytorch_format=False）:
# (32, 84, 84, 3) - (batch_size, height, width, channels)

# 图像（PyTorch 格式，pytorch_format=True）:
# (32, 3, 84, 84) - (batch_size, channels, height, width)
```

#### 2. `full_observations` - 完整观测序列
```python
print(batch['full_observations'].shape)
# 状态向量: (32, 10, 17) - (batch_size, sequence_length, obs_dim)
# 含义: 每个时间步的观测 [s_0, s_1, ..., s_9]

# 图像（NumPy 格式，pytorch_format=False）:
# (32, 10, 84, 84, 3) - (batch_size, seq_len, height, width, channels)

# 图像（PyTorch 格式，pytorch_format=True）:
# (32, 10, 3, 84, 84) - (batch_size, seq_len, channels, height, width)
```

#### 3. `next_observations` - 下一个观测序列
```python
print(batch['next_observations'].shape)
# 状态向量: (32, 10, 17) - (batch_size, sequence_length, obs_dim)
# 含义: 每个时间步的下一个观测 [s_1, s_2, ..., s_10]

# 图像格式同 full_observations
```

#### 4. `actions` - 动作序列
```python
print(batch['actions'].shape)
# 输出: (32, 10, 6)
# 形状: (batch_size, sequence_length, action_dim)
# 含义: 每个时间步采取的动作 [a_0, a_1, ..., a_9]
```

#### 5. `next_actions` - 下一个动作序列
```python
print(batch['next_actions'].shape)
# 输出: (32, 10, 6)
# 形状: (batch_size, sequence_length, action_dim)
# 含义: 每个时间步的下一个动作 [a_1, a_2, ..., a_10]
```

#### 6. `rewards` - 累积折扣奖励
```python
print(batch['rewards'].shape)
# 输出: (32, 10)
# 形状: (batch_size, sequence_length)
# 含义: 从序列开始累积的折扣奖励
#       rewards[i, t] = r_0 + γ*r_1 + γ²*r_2 + ... + γᵗ*r_t
```

**示例：**
```python
discount = 0.99
# 如果原始奖励是 [r0=1, r1=2, r2=3]
# 累积奖励会是:
# t=0: 1
# t=1: 1 + 0.99*2 = 2.98
# t=2: 2.98 + 0.99²*3 = 5.9203
```

#### 7. `masks` - 有效性掩码
```python
print(batch['masks'].shape)
# 输出: (32, 10)
# 形状: (batch_size, sequence_length)
# 含义: 该时间步是否有效 (1=有效, 0=无效)
#       沿序列传播，取最小值
```

#### 8. `terminals` - 终止标志
```python
print(batch['terminals'].shape)
# 输出: (32, 10)
# 形状: (batch_size, sequence_length)
# 含义: 是否到达终止状态 (1=终止, 0=未终止)
#       沿序列传播，取最大值（一旦终止就一直是终止）
```

#### 9. `valid` - 序列有效性
```python
print(batch['valid'].shape)
# 输出: (32, 10)
# 形状: (batch_size, sequence_length)
# 含义: 该时间步是否在 episode 内部 (1=有效, 0=跨越了 episode)
#       valid[i, t] = 0 如果 terminals[i, t-1] = 1
```

### 视觉观测的格式说明

对于视觉观测（图像），可以选择 NumPy 格式或 PyTorch 格式：

```python
# 加载视觉数据集
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('visual-singletask-task0-v0')

# 方式 1: NumPy 格式（默认）
batch = train_dataset.sample_sequence(batch_size=32, sequence_length=10)
print(batch['full_observations'].shape)
# 输出: (32, 10, 84, 84, 3)
# 形状: (batch, seq, height, width, channels)

# 方式 2: PyTorch 格式（推荐用于训练）
train_dataset.pytorch_format = True
batch = train_dataset.sample_sequence(batch_size=32, sequence_length=10)
print(batch['full_observations'].shape)
# 输出: (32, 10, 3, 84, 84)
# 形状: (batch, seq, channels, height, width)
# 可以直接输入到 PyTorch 的 Conv2d/Conv3d 层
```

### 完整示例：训练 Transformer

```python
import torch
import torch.nn as nn

# 加载数据
from envs.env_utils import make_env_and_datasets
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 采样序列
batch_size = 64
sequence_length = 20
batch = train_dataset.sample_sequence(
    batch_size=batch_size,
    sequence_length=sequence_length,
    discount=0.99
)

# 转换为 PyTorch 张量
device = torch.device('cuda')
observations = torch.from_numpy(batch['full_observations']).float().to(device)
actions = torch.from_numpy(batch['actions']).float().to(device)
rewards = torch.from_numpy(batch['rewards']).float().to(device)
valid = torch.from_numpy(batch['valid']).float().to(device)

print(f"Observations: {observations.shape}")  # (64, 20, 17)
print(f"Actions: {actions.shape}")            # (64, 20, 6)
print(f"Rewards: {rewards.shape}")            # (64, 20)
print(f"Valid mask: {valid.shape}")           # (64, 20)

# 在训练循环中使用
# model_output = transformer(observations, actions, rewards, mask=valid)
```

---

## 高级功能

### 1. 帧堆叠 (Frame Stacking)

用于视觉任务，将多个连续帧堆叠起来提供运动信息：

```python
# 设置帧堆叠
dataset.frame_stack = 4  # 堆叠 4 帧

# 采样时会自动堆叠
batch = dataset.sample(256)
print(batch['observations'].shape)
# 如果原始图像是 (84, 84, 3)
# 堆叠后会变成 (84, 84, 12)  # 3 * 4 = 12 通道
```

### 2. 图像增强 (Data Augmentation)

```python
# 设置增强概率
dataset.p_aug = 0.5  # 50% 概率应用增强
dataset.aug_padding = 4  # 随机裁剪的 padding

# 采样时会自动应用增强
batch = dataset.sample(256)
# 图像会被随机裁剪
```

### 3. 添加历史信息

```python
from utils.datasets import add_history

# 添加历史观测和动作
history_length = 5
dataset_with_history = add_history(dataset, history_length)

# 现在数据集包含额外的字段
batch = dataset_with_history.sample(256)
print(batch['observation_history'].shape)
# 输出: (256, 4, 17)  # (batch_size, history_length-1, obs_dim)
print(batch['action_history'].shape)
# 输出: (256, 4, 6)   # (batch_size, history_length-1, action_dim)
```

### 4. 复制和修改数据集

```python
# 复制数据集并修改某些字段
import numpy as np

new_dataset = dataset.copy(
    rewards=dataset['rewards'] * 10,  # 缩放奖励
    actions=np.clip(dataset['actions'], -0.99, 0.99)  # 裁剪动作
)

print(f"Original rewards mean: {dataset['rewards'].mean():.2f}")
print(f"New rewards mean: {new_dataset['rewards'].mean():.2f}")
```

### 5. 加载到 ReplayMemory

如果你需要使用 `ReplayMemory` 类：

```python
from utils.replay_memory import ReplayMemory

# 创建 ReplayMemory
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = torch.device('cuda')

memory = ReplayMemory(
    state_dim=state_dim,
    action_dim=action_dim,
    capacity=1_000_000,
    device=device
)

# 从 Dataset 加载数据
dataset.to_replay_memory(memory, max_samples=100_000)

# 现在可以使用 ReplayMemory 采样
states, actions, next_states, rewards, not_done = memory.sample(256)
```

---

## 完整示例

### 示例 1: 训练 SAC/TD3 等 Actor-Critic 方法

```python
import torch
from envs.env_utils import make_env_and_datasets

# 1. 加载数据
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 2. 训练循环
device = torch.device('cuda')
batch_size = 256
num_updates = 100000

for update in range(num_updates):
    # 采样一个 batch
    states, actions, next_states, rewards, not_done = train_dataset.to_tensors(
        batch_size=batch_size,
        device=device
    )
    
    # 训练你的模型
    # loss = your_algorithm.update(states, actions, next_states, rewards, not_done)
    
    if update % 1000 == 0:
        print(f"Update {update}, batch shapes: s={states.shape}, a={actions.shape}")
```

### 示例 2: 训练 Decision Transformer

```python
import torch
from envs.env_utils import make_env_and_datasets

# 1. 加载数据
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 2. 训练循环
device = torch.device('cuda')
batch_size = 64
sequence_length = 20
num_updates = 50000

for update in range(num_updates):
    # 采样序列
    batch = train_dataset.sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
        discount=0.99
    )
    
    # 转换为张量
    states = torch.from_numpy(batch['full_observations']).float().to(device)
    actions = torch.from_numpy(batch['actions']).float().to(device)
    rewards = torch.from_numpy(batch['rewards']).float().to(device)
    timesteps = torch.arange(sequence_length).repeat(batch_size, 1).to(device)
    
    # 训练 Decision Transformer
    # loss = decision_transformer.train_step(states, actions, rewards, timesteps)
    
    if update % 1000 == 0:
        print(f"Update {update}")
        print(f"  States: {states.shape}")      # (64, 20, 17)
        print(f"  Actions: {actions.shape}")    # (64, 20, 6)
        print(f"  Rewards: {rewards.shape}")    # (64, 20)
```

### 示例 3: 训练视觉任务（CNN 或 Vision Transformer）

```python
import torch
import torch.nn as nn
from envs.env_utils import make_env_and_datasets

# 1. 加载视觉数据集
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('visual-singletask-task0-v0')

# 2. 启用 PyTorch 格式
train_dataset.pytorch_format = True

# 3. 可选：启用数据增强
train_dataset.p_aug = 0.5  # 50% 概率应用随机裁剪
train_dataset.aug_padding = 4

# 4. 训练循环（单步方法，如 DQN/SAC）
device = torch.device('cuda')
batch_size = 128

for update in range(10000):
    # 采样 batch
    batch = train_dataset.sample(batch_size)
    
    # 转换为 tensor（已经是 PyTorch 格式）
    states = torch.from_numpy(batch['observations']).float().to(device)
    actions = torch.from_numpy(batch['actions']).float().to(device)
    next_states = torch.from_numpy(batch['next_observations']).float().to(device)
    
    print(f"States shape: {states.shape}")  # (128, 3, 84, 84)
    
    # 可以直接输入到 CNN
    # features = cnn(states)  # CNN expects (B, C, H, W)
    # q_values = q_network(features, actions)
    
    if update % 1000 == 0:
        print(f"Update {update}, states: {states.shape}")

# 5. 序列方法（如 Transformer）
batch_size = 32
sequence_length = 16

for update in range(5000):
    # 采样序列
    batch = train_dataset.sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # 转换为 tensor
    obs_seq = torch.from_numpy(batch['full_observations']).float().to(device)
    action_seq = torch.from_numpy(batch['actions']).float().to(device)
    
    print(f"Obs sequence shape: {obs_seq.shape}")  # (32, 16, 3, 84, 84)
    
    # 方式 A: 使用 3D 卷积处理时空信息
    # features = conv3d(obs_seq)  # Input: (B, T, C, H, W) or (B, C, T, H, W)
    
    # 方式 B: 先用 2D 卷积提取特征，再用 Transformer
    # B, T, C, H, W = obs_seq.shape
    # obs_flat = obs_seq.view(B * T, C, H, W)
    # features = cnn(obs_flat)  # (B*T, feature_dim)
    # features = features.view(B, T, -1)
    # output = transformer(features)
    
    if update % 1000 == 0:
        print(f"Update {update}")
```

### 示例 4: 检查数据集统计信息

```python
from envs.env_utils import make_env_and_datasets
import numpy as np

# 加载数据
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 打印基本信息
print(f"Dataset size: {len(train_dataset)}")
print(f"Available fields: {list(train_dataset.keys())}")

# 统计信息
print(f"\nObservation stats:")
print(f"  Shape: {train_dataset['observations'].shape}")
print(f"  Mean: {train_dataset['observations'].mean(axis=0)[:5]}")  # 前 5 维
print(f"  Std: {train_dataset['observations'].std(axis=0)[:5]}")

print(f"\nAction stats:")
print(f"  Shape: {train_dataset['actions'].shape}")
print(f"  Mean: {train_dataset['actions'].mean(axis=0)}")
print(f"  Std: {train_dataset['actions'].std(axis=0)}")
print(f"  Min: {train_dataset['actions'].min(axis=0)}")
print(f"  Max: {train_dataset['actions'].max(axis=0)}")

print(f"\nReward stats:")
print(f"  Mean: {train_dataset['rewards'].mean():.4f}")
print(f"  Std: {train_dataset['rewards'].std():.4f}")
print(f"  Min: {train_dataset['rewards'].min():.4f}")
print(f"  Max: {train_dataset['rewards'].max():.4f}")

# Episode 信息
n_episodes = len(train_dataset.terminal_indices)
episode_starts = train_dataset.episode_starts
episode_ends = train_dataset.terminal_indices
episode_lengths = episode_ends - episode_starts + 1

print(f"\nEpisode info:")
print(f"  Number of episodes: {n_episodes}")
print(f"  Average episode length: {episode_lengths.mean():.1f}")
print(f"  Min episode length: {episode_lengths.min()}")
print(f"  Max episode length: {episode_lengths.max()}")
```

---

## 常见问题

### Q1: 图像应该使用什么格式？NumPy 还是 PyTorch？

**简短回答：** 训练时使用 PyTorch 格式（`dataset.pytorch_format = True`），可视化时使用 NumPy 格式。

**详细说明：**

PyTorch 格式的优势：
- ✅ 直接兼容 `nn.Conv2d`, `nn.Conv3d` 等层
- ✅ GPU 内存布局更优，性能更好
- ✅ 与预训练模型（ResNet, ViT 等）直接兼容
- ✅ 零拷贝转换，开销可忽略

```python
# 推荐：训练时使用 PyTorch 格式
train_dataset.pytorch_format = True
batch = train_dataset.sample(256)
images = torch.from_numpy(batch['observations']).float().to(device)
# images 的 shape: (256, 3, 84, 84) - 可以直接输入 CNN

# 可视化时使用 NumPy 格式
train_dataset.pytorch_format = False
batch = train_dataset.sample(1)
img = batch['observations'][0]  # Shape: (84, 84, 3)
plt.imshow(img)  # matplotlib 期望 (H, W, C) 格式
```

### Q2: `sample()` 和 `sample_sequence()` 有什么区别？

- **`sample()`**: 采样**独立的单个转换** (s, a, s', r)，用于传统的 off-policy 算法 (DQN, SAC, TD3)
- **`sample_sequence()`**: 采样**连续的时间序列** [(s₀, a₀, r₀), (s₁, a₁, r₁), ...], 用于序列模型 (Transformers, RNNs)

### Q3: 为什么 `sample_sequence()` 返回累积奖励而不是即时奖励？

累积奖励对于某些算法（如 Decision Transformer）更有用，因为它们需要知道从当前状态开始的"return-to-go"。如果你需要即时奖励，可以通过差分计算：

```python
batch = dataset.sample_sequence(batch_size=32, sequence_length=10, discount=0.99)
cumulative_rewards = batch['rewards']

# 计算即时奖励
immediate_rewards = np.zeros_like(cumulative_rewards)
immediate_rewards[:, 0] = cumulative_rewards[:, 0]
for t in range(1, cumulative_rewards.shape[1]):
    immediate_rewards[:, t] = (cumulative_rewards[:, t] - cumulative_rewards[:, t-1]) / (0.99 ** t)
```

### Q4: 如何处理跨越 episode 边界的序列？

`sample_sequence()` 会采样可能跨越 episode 的序列，但提供了 `valid` 字段来标识有效的时间步：

```python
batch = dataset.sample_sequence(batch_size=32, sequence_length=10)
valid = batch['valid']  # (32, 10)

# 在训练时使用 valid 作为掩码
loss = compute_loss(predictions, targets) * valid
loss = loss.sum() / valid.sum()
```

### Q5: 如何确保数据已下载？

```python
# D4RL: 第一次运行时会自动下载
from envs import d4rl_utils
env = d4rl_utils.make_env('halfcheetah-medium-v2')  # 自动下载到 ~/.d4rl/

# Robomimic: 需要手动下载
# 参考: https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

# OGBench: 会自动下载
from envs import ogbench_utils
env, train_ds, val_ds = ogbench_utils.make_ogbench_env_and_datasets(
    'visual-singletask-task0-v0',
    dataset_dir='~/.ogbench/data'  # 自动下载到此目录
)
```

### Q6: 帧堆叠和 PyTorch 格式可以一起使用吗？

可以！它们可以完美配合使用：

```python
# 设置帧堆叠和 PyTorch 格式
dataset.frame_stack = 4  # 堆叠 4 帧
dataset.pytorch_format = True

batch = dataset.sample(256)
print(batch['observations'].shape)
# 输出: (256, 12, 84, 84)
# 说明: 3 (RGB) × 4 (frames) = 12 channels

# 这个格式可以直接输入到 CNN
# conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3)
# features = conv1(torch.from_numpy(batch['observations']).float())
```

### Q7: 我可以混合多个数据集吗？

可以，但需要手动合并：

```python
import numpy as np
from utils.datasets import Dataset

# 加载多个数据集
dataset1 = ...  # 第一个数据集
dataset2 = ...  # 第二个数据集

# 合并
combined_dataset = Dataset.create(
    observations=np.concatenate([dataset1['observations'], dataset2['observations']]),
    actions=np.concatenate([dataset1['actions'], dataset2['actions']]),
    rewards=np.concatenate([dataset1['rewards'], dataset2['rewards']]),
    next_observations=np.concatenate([dataset1['next_observations'], dataset2['next_observations']]),
    terminals=np.concatenate([dataset1['terminals'], dataset2['terminals']]),
    masks=np.concatenate([dataset1['masks'], dataset2['masks']]),
)
```

---

## 总结

`Dataset` 类提供了统一、高效的离线 RL 数据处理接口：

✅ **简单加载**: 一行代码加载 D4RL/Robomimic/OGBench 数据
✅ **灵活采样**: 支持单步采样和序列采样
✅ **自动处理**: Episode 边界、帧堆叠、数据增强
✅ **PyTorch 集成**: 直接转换为张量，支持 PyTorch 图像格式 (C, H, W)
✅ **图像格式**: 自动转换 NumPy (H, W, C) ↔ PyTorch (C, H, W)
✅ **类型安全**: 完整的类型提示

### 快速参考

```python
# 基本用法
from envs.env_utils import make_env_and_datasets
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 状态向量任务
batch = train_dataset.sample(256)

# 视觉任务（推荐设置）
train_dataset.pytorch_format = True  # 自动转换为 (B, C, H, W)
train_dataset.p_aug = 0.5           # 数据增强
batch = train_dataset.sample(256)

# 序列任务
batch = train_dataset.sample_sequence(batch_size=32, sequence_length=10)
```

如有问题，请查看源代码 `utils/datasets.py` 或参考上述示例。
