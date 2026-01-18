# PyTorch BC vs JAX 版本对比

## 代码结构对应关系

### 文件组织

| JAX 版本 | PyTorch 版本 | 说明 |
|---------|-------------|------|
| `main.py` | `train_bc.py` | 主训练脚本 |
| `agents/acfql_agent.py` | `agents/bc_agent.py` | Agent 实现 |
| `utils/flax_utils.py` | - | PyTorch 不需要（直接用 torch） |
| `utils/datasets.py` | `utils/datasets.py` | 数据集（已适配 PyTorch） |
| `evaluation.py` | `evaluation.py` | 评估函数（已适配） |

### 主要函数对应

| JAX 版本 | PyTorch 版本 |
|---------|-------------|
| `agent.create()` | `BCAgent.create()` |
| `agent.update(batch)` | `agent.update(batch)` |
| `agent.batch_update(batch)` | `agent.batch_update(batch)` |
| `agent.sample_actions(obs, rng=key)` | `agent.sample_actions(obs)` |
| `jax.random.split()` | PyTorch 内置随机数管理 |

---

## 训练脚本对比

### main() 函数结构

两个版本的 `main()` 函数现在具有**相同的结构**：

```python
# ===== 1. Setup =====
exp_name = get_exp_name(FLAGS.seed)
run = setup_wandb(...)
FLAGS.save_dir = os.path.join(...)
# 保存 flags.json

# ===== 2. House keeping =====
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
# PyTorch: torch.manual_seed(FLAGS.seed)
# JAX: jax.random.PRNGKey(FLAGS.seed)

log_step = 0
discount = FLAGS.discount

# ===== 3. Data loading =====
env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

# ===== 4. Handle dataset =====
def process_train_dataset(ds):
    # 处理 dataset proportion
    # 处理 sparse reward
    # 处理 robomimic reward
    return ds

train_dataset = process_train_dataset(train_dataset)
example_batch = train_dataset.sample(FLAGS.batch_size)

# ===== 5. Create agent =====
# JAX: agent = agent_class.create(FLAGS.seed, obs, actions, config)
# PyTorch: agent = BCAgent.create(observation_shape, action_dim, config)

# ===== 6. Setup logging =====
prefixes = ["eval", "offline_agent"]
logger = LoggingHelper(csv_loggers={...}, wandb_logger=wandb)

# ===== 7. Offline training loop =====
for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
    log_step += 1
    
    # Sample batch
    batch = train_dataset.sample_sequence(...) or train_dataset.sample(...)
    
    # Update agent
    # JAX: agent, offline_info = agent.update(batch)
    # PyTorch: offline_info = agent.update(batch)
    
    if i % FLAGS.log_interval == 0:
        logger.log(offline_info, "offline_agent", step=log_step)
    
    # Saving
    if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
        # JAX: save_agent(agent, FLAGS.save_dir, log_step)
        # PyTorch: agent.save(checkpoint_path)
    
    # Eval
    if i == FLAGS.offline_steps - 1 or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
        eval_info, _, renders = evaluate(...)
        logger.log(eval_info, "eval", step=log_step)

# ===== 8. Cleanup =====
for key, csv_logger in logger.csv_loggers.items():
    csv_logger.close()

# 保存最终模型
with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
    f.write(run.url)
```

---

## 命令行参数对比

### 完全相同的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run_group` | 'BC' / 'Debug' | 运行组 |
| `--seed` | 0 | 随机种子 |
| `--env_name` | ... | 环境名称 |
| `--save_dir` | 'runs/fbc/' / 'exp/' | 保存目录 |
| `--offline_steps` | 1000000 | 离线训练步数 |
| `--log_interval` | 5000 | 日志间隔 |
| `--eval_interval` | 100000 | 评估间隔 |
| `--save_interval` | -1 | 保存间隔 |
| `--discount` | 0.99 | 折扣因子 |
| `--eval_episodes` | 50 | 评估 episode 数 |
| `--video_episodes` | 0 | 视频 episode 数 |
| `--video_frame_skip` | 3 | 视频帧跳过 |
| `--dataset_proportion` | 1.0 | 数据集比例 |
| `--horizon_length` | 5 | Action chunking 长度 |
| `--sparse` | False | 稀疏奖励 |

### PyTorch 版本移除的参数

| JAX 参数 | 移除原因 |
|---------|---------|
| `--ogbench_dataset_dir` | 不需要本地 OGBench 数据 |
| `--dataset_replace_interval` | 简化训练流程 |
| `--online_steps` | 只做 BC，不做 online RL |
| `--buffer_size` | BC 不需要 replay buffer |
| `--start_training` | BC 不需要预填充 buffer |
| `--utd_ratio` | BC 不需要 update-to-data ratio |

### PyTorch 版本新增的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 3e-4 | 学习率（JAX 在 config 文件中） |
| `--batch_size` | 256 | Batch size（JAX 在 config 文件中） |
| `--flow_steps` | 10 | Flow 步数（JAX 在 config 文件中） |
| `--action_chunking` | False | 是否启用 action chunking |
| `--use_fourier_features` | False | 是否使用 Fourier features |
| `--encoder` | None | 视觉编码器 |
| `--weight_decay` | 0.0 | L2 正则化 |

---

## Agent API 对比

### 创建 Agent

```python
# JAX 版本
agent = ACFQLAgent.create(
    seed=FLAGS.seed,
    ex_observations=example_batch['observations'],
    ex_actions=example_batch['actions'],
    config=config,
)

# PyTorch 版本
agent = BCAgent.create(
    observation_shape=obs_shape,
    action_dim=action_dim,
    config=config,
)
```

### 更新 Agent

```python
# JAX 版本（函数式，返回新 agent）
agent, info = agent.update(batch)

# PyTorch 版本（原地更新，返回 info）
info = agent.update(batch)
```

### 采样动作

```python
# JAX 版本（需要显式传入 RNG）
rng, key = jax.random.split(rng)
actions = agent.sample_actions(observations=obs, rng=key)

# PyTorch 版本（使用内置随机数）
actions = agent.sample_actions(observations=obs)
```

### 保存/加载

```python
# JAX 版本
from utils.flax_utils import save_agent, load_agent
save_agent(agent, save_dir, step)
agent = load_agent(path, agent)

# PyTorch 版本
agent.save('checkpoint.pt')
agent.load('checkpoint.pt')
```

---

## 数据处理对比

### 采样 Batch

```python
# 两个版本完全相同
if FLAGS.action_chunking:
    batch = train_dataset.sample_sequence(
        batch_size=FLAGS.batch_size,
        sequence_length=FLAGS.horizon_length,
        discount=discount
    )
else:
    batch = train_dataset.sample(FLAGS.batch_size)
```

### 转换为 Tensor

```python
# JAX 版本
# batch 已经是 JAX arrays，不需要转换

# PyTorch 版本
batch_tensor = {
    'observations': torch.from_numpy(batch['observations']).float(),
    'actions': torch.from_numpy(batch['actions']).float(),
    # ...
}
```

---

## 评估对比

### 评估函数调用

```python
# 两个版本完全相同
eval_info, trajs, renders = evaluate(
    agent=agent,
    env=eval_env,
    action_dim=action_dim,
    num_eval_episodes=FLAGS.eval_episodes,
    num_video_episodes=FLAGS.video_episodes,
    video_frame_skip=FLAGS.video_frame_skip,
)

if len(renders) > 0:
    eval_info['video'] = get_wandb_video(
        renders, 
        fps=int(30 / max(FLAGS.video_frame_skip, 1))
    )

logger.log(eval_info, "eval", step=log_step)
```

### 动作执行（evaluation.py 中）

```python
# 两个版本相同
action = actor_fn(observations=observation)

# Action chunking 处理（evaluation.py 已适配）
if len(action_queue) == 0:
    action = np.array(action).reshape(-1, action_dim)
    action_chunk_len = action.shape[0]
    for a in action:
        action_queue.append(a)
else:
    have_new_action = False

action = action_queue.pop(0)
```

---

## 运行示例对比

### JAX 版本

```bash
python main.py \
    --env_name=halfcheetah-medium-v2 \
    --offline_steps=1000000 \
    --online_steps=0 \
    --agent=agents/acfql_bc_distill.py \
    --horizon_length=5
```

### PyTorch 版本

```bash
python train_bc.py \
    --env_name=halfcheetah-medium-v2 \
    --offline_steps=1000000 \
    --horizon_length=5 \
    --action_chunking=False
```

**主要区别：**
- PyTorch 版本不需要 `--agent` 参数（agent 固定为 BC）
- PyTorch 版本不需要 `--online_steps`（纯 BC）
- PyTorch 版本用 `--action_chunking` 控制是否使用序列采样

---

## 输出文件对比

### 保存的文件

| JAX 版本 | PyTorch 版本 |
|---------|-------------|
| `flags.json` | `flags.json` ✅ |
| `offline_agent.csv` | `offline_agent.csv` ✅ |
| `eval.csv` | `eval.csv` ✅ |
| `checkpoint_{step}/` (目录) | `checkpoint_{step}.pt` (文件) |
| `final/` (目录) | `final_model.pt` (文件) |
| `token.tk` | `token.tk` ✅ |

### 检查点格式

```python
# JAX 版本
checkpoint/
  ├── agent/
  │   ├── network/
  │   └── ...
  └── metadata.json

# PyTorch 版本
checkpoint.pt  # 单个文件，包含所有内容
{
    'actor': actor.state_dict(),
    'critic': critic.state_dict(),
    'target_critic': target_critic.state_dict(),
    'actor_optimizer': actor_optimizer.state_dict(),
    'critic_optimizer': critic_optimizer.state_dict(),
    'config': config,
    'step': step,
}
```

---

## 关键差异总结

### 1. 随机数管理

| JAX | PyTorch |
|-----|---------|
| 显式 RNG 传递 | 全局随机数状态 |
| `jax.random.split()` | `torch.manual_seed()` |
| 函数式编程 | 命令式编程 |

### 2. Agent 更新

| JAX | PyTorch |
|-----|---------|
| 函数式（返回新 agent） | 原地更新 |
| `agent, info = agent.update()` | `info = agent.update()` |
| 不可变数据结构 | 可变数据结构 |

### 3. 模型保存

| JAX | PyTorch |
|-----|---------|
| 自定义 checkpoint 格式 | 标准 `.pt` 格式 |
| 需要辅助函数 | 内置 `save()`/`load()` |
| 目录结构 | 单个文件 |

### 4. 配置管理

| JAX | PyTorch |
|-----|---------|
| 配置文件 (`.py`) | 命令行参数 + dataclass |
| `ml_collections.ConfigDict` | `dataclass` |
| `--agent=agents/xxx.py` | 直接在代码中指定 |

---

## 迁移清单

如果你想从 JAX 版本迁移到 PyTorch 版本：

### ✅ 完全兼容的部分
- [x] 环境加载（`make_env_and_datasets`）
- [x] 数据集处理（`process_train_dataset`）
- [x] 数据采样（`sample` / `sample_sequence`）
- [x] 评估流程（`evaluate`）
- [x] 日志记录（`LoggingHelper`）
- [x] Action chunking 执行

### ⚠️ 需要适配的部分
- [ ] Agent 创建接口不同
- [ ] 更新方式不同（函数式 vs 原地）
- [ ] 随机数管理不同
- [ ] 模型保存格式不同

### 🔄 可选的改进
- [ ] 添加 `torch.compile()` 加速
- [ ] 添加混合精度训练
- [ ] 添加分布式训练支持
- [ ] 添加更多超参数到命令行

---

## 性能对比

| 指标 | JAX | PyTorch |
|------|-----|---------|
| 训练速度 | ⚡⚡⚡⚡ (JIT 编译) | ⚡⚡⚡ (稍慢) |
| 内存使用 | 💾💾💾 | 💾💾💾💾 (稍多) |
| 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 调试友好 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 生态系统 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 快速参考

### 启动训练

```bash
# 基本训练
python train_bc.py --env_name=halfcheetah-medium-v2

# Action chunking
python train_bc.py --env_name=halfcheetah-medium-v2 --action_chunking=True

# Robomimic
python train_bc.py --env_name=lift-mh-low_dim --action_chunking=True --horizon_length=10
```

### 加载模型评估

```python
from agents.bc_agent import BCAgent, BCAgentConfig

config = BCAgentConfig(...)
agent = BCAgent.create(observation_shape, action_dim, config)
agent.load('runs/fbc/.../final_model.pt')

# 评估
from evaluation import evaluate
stats, _, _ = evaluate(agent, env, action_dim, num_eval_episodes=100)
```

---

如有问题，请对比：
- JAX 版本: `main.py`
- PyTorch 版本: `train_bc.py`
