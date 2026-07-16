# Checkpoint 机制与断点恢复

> 创建日期：2026-07-10
> 状态：概念学习阶段，待实现

---

## 一、什么是 Checkpoint

Checkpoint 就是训练过程中的"存档"。

```
游戏                            训练
────                            ────
存档：角色等级、装备、位置        模型参数、优化器状态、epoch
读档：从存档点继续玩              从中断的 epoch 继续训
没存档 → 只能从头重来             没 checkpoint → 从随机初始化重训
```

核心价值：**训练中途崩溃（如 NCCL 超时）后，不需要从零开始，而是从最近的 checkpoint 恢复。**

---

## 二、Checkpoint 必须保存的内容

一个完整的 checkpoint 不仅是模型参数，还必须包含优化器状态：

```
Checkpoint 完整内容
├── 1. 模型参数 (model.state_dict())       ← 权重矩阵
├── 2. 优化器状态 (optimizer.state_dict())  ← AdamW 的一阶动量 + 二阶动量
├── 3. 当前 epoch / step                   ← 恢复训练进度
├── 4. (可选) 学习率调度器状态 (scheduler.state_dict())
└── 5. (可选) 随机数种子 (torch.get_rng_state())
```

### 为什么必须保存优化器状态？

AdamW 优化器内部维护每个参数的两个状态：

| 状态 | 名称 | 作用 |
|------|------|------|
| 一阶动量 | exp_avg | 梯度的指数移动平均（方向趋势） |
| 二阶动量 | exp_avg_sq | 梯度平方的指数移动平均（自适应学习率） |

如果只恢复模型参数、不恢复优化器状态：

```
正常训练（连续 5 个 epoch）：
  优化器积累了 5 轮的梯度趋势 → 更新方向稳定

只恢复参数，优化器重置：
  momentum = 0, variance = 0 → 优化器"失忆"
  → 第一步更新方向可能错误 → Loss 突然跳变 → 训练短暂震荡
```

### 三种保存策略对比

| 保存内容 | 能否恢复训练 | 问题 |
|---------|------------|------|
| 只存模型参数 | 能，但效果差 | 优化器状态丢失，Loss 跳变 |
| 模型参数 + 优化器状态 | 能，效果好 | **标准做法** |
| 模型参数 + 优化器 + epoch + 种子 | 完美恢复 | 最佳实践 |

---

## 三、当前代码的问题

`mini_gpt_train.py` 的 `model_save()` 只保存了模型参数：

```python
torch.save({
    'model_state_dict': model.state_dict(),    # ✅ 模型参数
    'vocab': tokenizer.vocab,                   # ✅ 词表
    'vocab_size': tokenizer.vocab_size,         # ✅ 词表大小
    'model_config': model_config,               # ✅ 模型配置
    # ❌ 缺少 optimizer.state_dict() — 恢复训练会震荡
    # ❌ 缺少 epoch 信息 — 不知道从哪继续
}, 'model.pth')
```

这个格式适合**训练完成后保存用于推理**，但**不适合中途崩溃恢复**。

---

## 四、Checkpoint 的完整工作流程

```
第1次训练：
  Epoch 0 完成 → 保存 checkpoint_epoch0.pth
  Epoch 1 完成 → 保存 checkpoint_epoch1.pth（保留 epoch0 作为备份）
  Epoch 2      → ❌ NCCL 超时崩溃！

重新启动训练：
  → 检测到 checkpoint_epoch1.pth 存在
  → 加载：模型参数 + 优化器状态 + epoch=1
  → 从 Epoch 2 继续训练
  → 不需要从随机初始化重新开始

训练全部完成后：
  → 最终保存 model.pth（只含模型参数，供推理用）
  → checkpoint 文件可以删除
```

### 为什么保留多个 checkpoint？

保存 checkpoint 的瞬间如果崩溃，文件可能损坏。保留最近 2-3 个 epoch 的 checkpoint，损坏时可以回退到上一个。

```
checkpoint_epoch8.pth   ← 最新（如果损坏，回退到 epoch7）
checkpoint_epoch7.pth   ← 备份
checkpoint_epoch6.pth   ← 备份（更老的可以删）
```

---

## 五、实现要点（TODO）

### DDP 脚本 (`mini_gpt_train_ddp.py`)

```python
# --- 保存 checkpoint（每个 epoch 结束后，只 rank 0 执行） ---
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.module.state_dict(),  # DDP 要 .module
        'optimizer_state_dict': optimizer.state_dict(),  # ← 关键
        'epoch': epoch,
    }, path)

# --- 恢复 checkpoint（训练开始时检查） ---
def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location='cpu')
    model.module.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])  # ← 关键
    start_epoch = ckpt['epoch'] + 1  # 从下一个 epoch 继续
    return start_epoch

# --- 训练循环中 ---
for epoch in range(start_epoch, total_epochs):  # ← 从 start_epoch 开始
    ...
    # 每个 epoch 结束后保存
    if is_main:
        save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch{epoch}.pth')
        # 删除过旧的 checkpoint（只保留最近 2-3 个）
```

### Accelerate 脚本 (`mini_gpt_train_accelerate.py`)

Accelerate 有内置的 checkpoint 支持：

```python
# Accelerate 方式（推荐，更简洁）
accelerator.save_state('checkpoint_dir')  # 保存
# accelerator.load_state('checkpoint_dir')  # 恢复
```

---

## 六、TODO 清单

- [ ] 给 `mini_gpt_train_ddp.py` 添加 checkpoint 保存/恢复
- [ ] 给 `mini_gpt_train_accelerate.py` 添加 checkpoint 保存/恢复
- [ ] 决定保存频率：每 epoch / 每 N steps
- [ ] 实现旧 checkpoint 自动清理（只保留最近 2-3 个）
- [ ] 测试：中途 kill 进程后能否正确恢复
- [ ] 更新 `多卡并行训练.md` 文档，补充 checkpoint 相关章节

---

## 七、参考

- PyTorch 官方文档：[Saving and Loading a General Checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- Accelerate 文档：[Checkpoint](https://huggingface.co/docs/accelerate/en/package_reference/torch_wrappers#accelerate.Accelerator.save_state)
