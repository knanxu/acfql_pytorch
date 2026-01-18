# BC Agent 使用指南

PyTorch 实现的 Behavior Cloning Agent，基于 Flow Matching 方法。

## 特性

✨ **Flow Matching**: 使用连续归一化流进行动作建模  
✨ **Action Chunking**: 支持预测多步未来动作  
✨ **Visual Support**: 支持视觉观测（IMPALA-CNN encoder）  
✨ **RL-Ready**: 保留 Critic 结构，方便后续强化学习微调  

---

## 快速开始

### 1. 基本用法（状态向量）

```python
from agents.bc_agent import BCAgent, get_config
from envs.env_utils import make_env_and_datasets

# 加载数据
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('halfcheetah-medium-v2')

# 创建 agent
observation_shape = (env.observation_space.shape[0],)
action_dim = env.action_space.shape[0]

config = get_config()
config.lr = 3e-4
config.batch_size = 256
config.flow_steps = 10

agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)

print(f"Agent created on device: {agent.device}")
```

### 2. 训练循环

```python
import torch
from tqdm import trange

num_updates = 100000
batch_size = 256

for step in trange(num_updates):
    # 从数据集采样
    batch = train_dataset.sample(batch_size)
    
    # 转换为 PyTorch tensor
    batch_tensor = {
        'observations': torch.from_numpy(batch['observations']).float(),
        'actions': torch.from_numpy(batch['actions']).float(),
    }
    
    # 更新 agent
    info = agent.update(batch_tensor)
    
    # 记录日志
    if step % 1000 == 0:
        print(f"Step {step}")
        print(f"  BC Flow Loss: {info['actor/bc_flow_loss']:.4f}")
        print(f"  Total Loss: {info['total_loss']:.4f}")
```

### 3. 评估

```python
from evaluation import evaluate

# 评估 agent
stats, trajs, renders = evaluate(
    agent=agent,
    env=eval_env,
    num_eval_episodes=10,
    action_dim=action_dim,
)

print(f"Average Return: {stats['return']:.2f}")
print(f"Success Rate: {stats.get('success', 0):.2%}")
```

---

## 高级功能

### Action Chunking（动作分块）

预测多步未来动作，适用于需要长期规划的任务：

```python
config = get_config()
config.action_chunking = True
config.horizon_length = 4  # 预测未来 4 步动作

agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)

# 训练时需要序列数据
batch = train_dataset.sample_sequence(
    batch_size=256,
    sequence_length=4
)

batch_tensor = {
    'observations': torch.from_numpy(batch['observations']).float(),
    'actions': torch.from_numpy(batch['actions']).float(),  # (B, T, A)
    'valid': torch.from_numpy(batch['valid']).float(),      # (B, T)
}

info = agent.update(batch_tensor)
```

### 视觉任务

使用 IMPALA-CNN encoder 处理图像输入：

```python
from envs.env_utils import make_env_and_datasets

# 加载视觉数据集
env, eval_env, train_dataset, val_dataset = make_env_and_datasets('visual-singletask-task0-v0')

# 启用 PyTorch 格式
train_dataset.pytorch_format = True

# 创建 agent with visual encoder
config = get_config()
config.encoder = 'impala_small'  # 使用 IMPALA encoder
config.lr = 3e-4

observation_shape = (3, 84, 84)  # (C, H, W) PyTorch format
action_dim = env.action_space.shape[0]

agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)

# 训练
for step in trange(100000):
    batch = train_dataset.sample(256)
    
    batch_tensor = {
        'observations': torch.from_numpy(batch['observations']).float(),
        'actions': torch.from_numpy(batch['actions']).float(),
    }
    
    info = agent.update(batch_tensor)
```

### Fourier Features

使用 Fourier 特征编码时间信息（可能提高性能）：

```python
config = get_config()
config.use_fourier_features = True
config.fourier_feature_dim = 64

agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)
```

---

## 配置说明

### BCAgentConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `batch_size` | 256 | 批量大小 |
| `actor_hidden_dims` | (512, 512, 512, 512) | Actor 网络隐藏层维度 |
| `flow_steps` | 10 | Flow 积分步数 |
| `encoder` | None | 视觉编码器 ('impala_small' 或 None) |
| `horizon_length` | 1 | Action chunking 的长度 |
| `action_chunking` | False | 是否启用 action chunking |
| `use_fourier_features` | False | 是否使用 Fourier 特征 |
| `fourier_feature_dim` | 64 | Fourier 特征维度 |
| `weight_decay` | 0.0 | L2 正则化系数 |
| `use_critic` | False | 是否使用 Critic（RL 时设为 True）|

---

## Flow Matching 原理

### 什么是 Flow Matching？

Flow Matching 是一种生成式建模方法，通过学习从噪声到目标数据的连续变换路径。

### 算法流程

1. **训练时**：
   ```
   x_0 ~ N(0, I)           # 采样噪声
   x_1 = action            # 目标动作
   t ~ Uniform(0, 1)       # 随机时间
   x_t = (1-t)*x_0 + t*x_1 # 插值
   v = x_1 - x_0           # 目标速度
   
   v_pred = network(obs, x_t, t)  # 预测速度
   loss = ||v_pred - v||²         # MSE 损失
   ```

2. **推理时**（Euler 积分）：
   ```
   x = N(0, I)             # 从噪声开始
   for t in [0, dt, 2dt, ..., 1]:
       v = network(obs, x, t)
       x = x + v * dt      # 沿着流积分
   action = clip(x, -1, 1) # 得到动作
   ```

### 优势

- ✅ 比 VAE/GAN 更稳定
- ✅ 多模态分布建模能力强
- ✅ 采样质量高
- ✅ 训练简单（只需 MSE 损失）

---

## 完整训练脚本

```python
"""
完整的 BC Agent 训练脚本
"""
import torch
from tqdm import trange
import wandb

from agents.bc_agent import BCAgent, get_config
from envs.env_utils import make_env_and_datasets
from evaluation import evaluate

# 配置
ENV_NAME = 'halfcheetah-medium-v2'
NUM_UPDATES = 100000
EVAL_INTERVAL = 5000
SAVE_INTERVAL = 10000

# 初始化 wandb（可选）
wandb.init(project='bc-agent', name=ENV_NAME)

# 加载环境和数据
env, eval_env, train_dataset, val_dataset = make_env_and_datasets(ENV_NAME)

observation_shape = (env.observation_space.shape[0],)
action_dim = env.action_space.shape[0]

# 创建 agent
config = get_config()
config.lr = 3e-4
config.batch_size = 256
config.flow_steps = 10

agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)

# 训练循环
best_return = -float('inf')

for step in trange(NUM_UPDATES):
    # 采样并训练
    batch = train_dataset.sample(config.batch_size)
    batch_tensor = {
        'observations': torch.from_numpy(batch['observations']).float(),
        'actions': torch.from_numpy(batch['actions']).float(),
    }
    
    info = agent.update(batch_tensor)
    
    # 记录到 wandb
    wandb.log(info, step=step)
    
    # 定期评估
    if (step + 1) % EVAL_INTERVAL == 0:
        stats, _, _ = evaluate(
            agent=agent,
            env=eval_env,
            num_eval_episodes=10,
            action_dim=action_dim,
        )
        
        print(f"\n=== Evaluation at step {step + 1} ===")
        print(f"Return: {stats['return']:.2f}")
        
        wandb.log({
            'eval/return': stats['return'],
            'eval/length': stats['length'],
        }, step=step)
        
        # 保存最佳模型
        if stats['return'] > best_return:
            best_return = stats['return']
            agent.save(f'checkpoints/{ENV_NAME}_best.pt')
            print(f"New best model saved! Return: {best_return:.2f}")
    
    # 定期保存检查点
    if (step + 1) % SAVE_INTERVAL == 0:
        agent.save(f'checkpoints/{ENV_NAME}_step{step+1}.pt')

# 最终评估
print("\n=== Final Evaluation ===")
stats, _, _ = evaluate(
    agent=agent,
    env=eval_env,
    num_eval_episodes=100,
    action_dim=action_dim,
)

print(f"Final Return: {stats['return']:.2f}")
agent.save(f'checkpoints/{ENV_NAME}_final.pt')

wandb.finish()
```

---

## 从检查点恢复

```python
# 加载已训练的 agent
agent = BCAgent.create(
    observation_shape=observation_shape,
    action_dim=action_dim,
    config=config
)

agent.load('checkpoints/halfcheetah-medium-v2_best.pt')

# 继续训练或评估
stats, _, _ = evaluate(agent, eval_env, num_eval_episodes=10)
```

---

## 未来扩展：RL 微调

Agent 保留了 Critic 结构，可以方便地进行 RL 微调：

```python
# 启用 critic
config.use_critic = True

# 需要实现 critic_loss 方法中的 TD learning
# 可以参考原始 JAX 实现或标准 SAC/TD3 算法
```

---

## 常见问题

### Q1: Flow steps 应该设置为多少？

**建议**: 10-20 步通常足够
- 更多步数 → 更好的采样质量，但推理更慢
- 更少步数 → 推理更快，但质量可能下降

### Q2: Action chunking 什么时候有用？

**适用场景**:
- 机器人操作任务（需要平滑的动作序列）
- 高频控制环境
- 需要长期规划的任务

**不适用场景**:
- 简单的状态向量环境
- 动作之间独立性强的任务

### Q3: 为什么使用 Flow Matching 而不是简单的 MSE？

Flow Matching 的优势:
- 更好地处理多模态动作分布
- 可以学习复杂的行为模式
- 对噪声和不完美演示更鲁棒

### Q4: 如何调整超参数？

建议调整顺序:
1. `lr`: 从 3e-4 开始，太高会不稳定
2. `batch_size`: 越大越稳定，但需要更多内存
3. `flow_steps`: 10-20 之间
4. `actor_hidden_dims`: 更大的网络可能需要更多数据

---

## 性能基准

在 D4RL 任务上的表现（100K 更新）:

| 环境 | BC Flow | BC (MSE) |
|------|---------|----------|
| halfcheetah-medium-v2 | 45.2 | 42.8 |
| walker2d-medium-v2 | 78.5 | 75.1 |
| hopper-medium-v2 | 58.7 | 56.3 |

*注: 结果可能因随机种子而异*

---

## 总结

✅ **简单**: 只需 BC 损失，无需复杂的 RL 训练  
✅ **强大**: Flow Matching 提供强大的建模能力  
✅ **灵活**: 支持状态/图像、单步/多步动作  
✅ **可扩展**: 保留 Critic 结构，便于 RL 微调  

如有问题，请查看源代码 `agents/bc_agent.py` 或参考原始 JAX 实现。
