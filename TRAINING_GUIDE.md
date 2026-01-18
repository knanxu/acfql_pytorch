# BC Training Guide

完整的 Behavior Cloning 训练指南，使用 PyTorch 实现。

## 快速开始

### 1. 基本训练（状态向量环境）

```bash
python train_bc.py \
    --env_name=halfcheetah-medium-v2 \
    --training_steps=100000 \
    --eval_interval=10000 \
    --seed=0
```

### 2. 使用 Action Chunking

```bash
python train_bc.py \
    --env_name=halfcheetah-medium-v2 \
    --training_steps=100000 \
    --action_chunking=True \
    --horizon_length=5 \
    --seed=0
```

### 3. 视觉任务训练

```bash
python train_bc.py \
    --env_name=visual-singletask-task0-v0 \
    --encoder=impala_small \
    --training_steps=500000 \
    --batch_size=128 \
    --lr=1e-4 \
    --seed=0
```

### 4. Robomimic 任务

```bash
python train_bc.py \
    --env_name=lift-mh-low_dim \
    --training_steps=200000 \
    --action_chunking=True \
    --horizon_length=10 \
    --batch_size=256 \
    --seed=0
```

---

## 命令行参数详解

### 基本设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run_group` | 'BC' | Wandb 运行组名 |
| `--seed` | 0 | 随机种子 |
| `--env_name` | 'halfcheetah-medium-v2' | 环境名称 |
| `--save_dir` | 'runs/fbc/' | 保存目录 |

### 训练设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--training_steps` | 1000000 | 训练步数 |
| `--batch_size` | 256 | 批量大小 |
| `--log_interval` | 1000 | 日志记录间隔 |
| `--eval_interval` | 50000 | 评估间隔 |
| `--save_interval` | 100000 | 保存检查点间隔 |

### Agent 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 3e-4 | 学习率 |
| `--flow_steps` | 10 | Flow matching 积分步数 |
| `--horizon_length` | 5 | Action chunking 长度 |
| `--action_chunking` | False | 是否使用 action chunking |
| `--use_fourier_features` | False | 是否使用 Fourier 特征 |

### 评估设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eval_episodes` | 10 | 评估 episode 数量 |
| `--video_episodes` | 0 | 录制视频的 episode 数量 |
| `--video_frame_skip` | 3 | 视频帧跳过数量 |

### 数据集设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_proportion` | 1.0 | 使用数据集的比例 |
| `--dataset_replace_interval` | 0 | 数据集替换间隔（大数据集） |
| `--ogbench_dataset_dir` | None | OGBench 数据集目录 |
| `--sparse` | False | 使用稀疏奖励 |

### 编码器设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder` | None | 视觉编码器 (None 或 'impala_small') |

---

## 训练流程

### 1. 数据加载

脚本会自动：
- 检测环境类型（状态/视觉）
- 加载对应的数据集（D4RL/Robomimic/OGBench）
- 处理数据集（比例、稀疏奖励等）

### 2. Agent 创建

根据环境和配置自动创建：
- 状态向量：MLP-based flow network
- 视觉：IMPALA-CNN + MLP flow network
- Action chunking：自动调整动作维度

### 3. 训练循环

```
for step in range(training_steps):
    1. 采样 batch（单步或序列）
    2. 更新 agent（flow matching loss）
    3. 记录日志（每 log_interval 步）
    4. 评估（每 eval_interval 步）
    5. 保存检查点（每 save_interval 步）
```

### 4. 评估

评估时：
- 使用确定性采样（temperature=0）
- Action chunking 自动展开执行
- 记录 return、length、success 等指标
- 可选录制视频上传到 wandb

---

## 输出文件

训练完成后，`runs/fbc/` 目录下会包含：

```
runs/fbc/bc_training/BC/halfcheetah-medium-v2/sd000_20240116_123456/
├── flags.json              # 训练配置
├── train.csv               # 训练日志
├── eval.csv                # 评估日志
├── checkpoint_100000.pt    # 检查点
├── checkpoint_200000.pt
├── final_model.pt          # 最终模型
└── wandb_url.txt           # Wandb 链接
```

---

## 常见环境

### D4RL MuJoCo

```bash
# Medium datasets
--env_name=halfcheetah-medium-v2
--env_name=walker2d-medium-v2
--env_name=hopper-medium-v2

# Medium-Expert datasets
--env_name=halfcheetah-medium-expert-v2
--env_name=walker2d-medium-expert-v2

# AntMaze
--env_name=antmaze-umaze-v2
--env_name=antmaze-medium-play-v2
```

### D4RL Adroit

```bash
--env_name=pen-human-v1
--env_name=hammer-human-v1
--env_name=door-human-v1
--env_name=relocate-human-v1
```

### Robomimic

```bash
--env_name=lift-mh-low_dim
--env_name=can-mh-low_dim
--env_name=square-mh-low_dim
--env_name=transport-mh-low_dim
```

### OGBench

```bash
--env_name=visual-singletask-task0-v0
--env_name=visual-singletask-task1-v0
```

---

## 高级用法

### 1. 多卡训练

```bash
# 使用第 0 块 GPU
CUDA_VISIBLE_DEVICES=0 python train_bc.py --env_name=...

# 使用第 1 块 GPU
CUDA_VISIBLE_DEVICES=1 python train_bc.py --env_name=...
```

### 2. 批量实验（不同种子）

```bash
for seed in 0 1 2 3 4; do
    python train_bc.py \
        --env_name=halfcheetah-medium-v2 \
        --seed=$seed \
        --run_group=BC_seeds &
done
```

### 3. 从检查点恢复

修改 `train_bc.py`：

```python
# 在创建 agent 后添加
if os.path.exists('path/to/checkpoint.pt'):
    agent.load('path/to/checkpoint.pt')
    print(f"Resumed from checkpoint")
```

### 4. 超参数搜索

```bash
# Learning rate search
for lr in 1e-4 3e-4 1e-3; do
    python train_bc.py \
        --env_name=halfcheetah-medium-v2 \
        --lr=$lr \
        --run_group=lr_search
done

# Flow steps search
for steps in 5 10 20; do
    python train_bc.py \
        --env_name=halfcheetah-medium-v2 \
        --flow_steps=$steps \
        --run_group=flow_search
done
```

---

## 性能调优建议

### 1. 学习率

- **状态向量**: `3e-4` (default)
- **视觉任务**: `1e-4` (更小)
- **大型网络**: `1e-4` to `3e-5`

### 2. Batch Size

- **MuJoCo**: 256 (default)
- **视觉任务**: 128 (GPU 内存限制)
- **大数据集**: 512 (如果内存足够)

### 3. Flow Steps

- **快速采样**: 5 steps
- **平衡**: 10 steps (default)
- **高质量**: 20 steps

### 4. Action Chunking

- **机器人操作**: horizon_length=10-20
- **MuJoCo**: horizon_length=5
- **不需要**: 保持 False

---

## 监控训练

### Wandb Dashboard

训练会自动记录到 wandb：

1. **训练曲线**:
   - `train/actor_loss`: BC flow loss
   - `train/bc_flow_loss`: 同上
   - `train/steps_per_sec`: 训练速度

2. **评估指标**:
   - `eval/return`: 平均回报
   - `eval/length`: 平均 episode 长度
   - `eval/success`: 成功率（如果环境支持）
   - `eval/video`: 视频（如果启用）

### 命令行输出

```
[Step 10000] Running evaluation...
Evaluation results:
  Return: 4523.45
  Length: 987.3
  Success: 0.80

[Step 10000] Saved checkpoint to runs/fbc/.../checkpoint_10000.pt
```

---

## 故障排除

### 1. CUDA Out of Memory

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小 batch size
--batch_size=128

# 或使用 CPU
--device=cpu  # 需要在代码中添加此参数
```

### 2. 数据集太大

**问题**: 内存不足以加载整个数据集

**解决方案**:
```bash
# 使用部分数据集
--dataset_proportion=0.5

# 或使用数据集替换
--dataset_replace_interval=1000
```

### 3. 训练不稳定

**问题**: Loss 震荡或发散

**解决方案**:
```bash
# 降低学习率
--lr=1e-4

# 增加 batch size
--batch_size=512

# 使用 weight decay
# 在 BCAgentConfig 中设置 weight_decay=1e-4
```

### 4. 评估性能差

**问题**: 训练 loss 低但评估效果差

**可能原因**:
- Flow steps 太少
- Action chunking 配置不当
- 数据集质量问题

**解决方案**:
```bash
# 增加 flow steps
--flow_steps=20

# 尝试不同的 horizon_length
--horizon_length=10

# 检查数据集统计
```

---

## 与 JAX 版本的对比

| 特性 | JAX 版本 | PyTorch 版本 |
|------|---------|-------------|
| 框架 | JAX/Flax | PyTorch |
| JIT 编译 | ✅ | ❌ (可用 torch.compile) |
| 训练速度 | 稍快 | 稍慢但差距小 |
| 内存使用 | 稍少 | 稍多 |
| 易用性 | 较难 | 更容易 |
| 调试 | 较难 | 更容易 |
| 模型保存 | 自定义格式 | .pt 标准格式 |

---

## 最佳实践

### 1. 实验管理

```bash
# 使用有意义的 run_group
--run_group=BC_MuJoCo_v1
--run_group=BC_Visual_experiments

# 记录多个种子
for seed in {0..4}; do
    python train_bc.py --seed=$seed --run_group=final_results
done
```

### 2. 评估策略

```bash
# 初期：快速评估
--eval_interval=10000 --eval_episodes=5

# 后期：详细评估
--eval_interval=50000 --eval_episodes=50

# 最终：完整评估 + 视频
--eval_episodes=100 --video_episodes=5
```

### 3. 保存策略

```bash
# 开发阶段：频繁保存
--save_interval=10000

# 正式训练：少量保存
--save_interval=100000

# 只保存最终模型
--save_interval=-1
```

---

## 下一步

训练完成后，你可以：

1. **评估模型**:
   ```python
   from agents.bc_agent import BCAgent
   agent = BCAgent.create(...)
   agent.load('runs/fbc/.../final_model.pt')
   ```

2. **部署到实际环境**:
   ```python
   obs, _ = env.reset()
   action = agent.sample_actions(obs)
   ```

3. **RL 微调**:
   - 设置 `config.use_critic = True`
   - 实现 `critic_loss` 方法
   - 添加 online 数据收集

---

如有问题，请查看：
- `train_bc.py` - 训练脚本
- `agents/bc_agent.py` - Agent 实现
- `BC_AGENT_GUIDE.md` - Agent 使用指南
- `DATASET_USAGE_GUIDE.md` - 数据集使用指南
