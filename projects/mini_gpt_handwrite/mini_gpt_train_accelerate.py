"""
mini_gpt Accelerate 多卡分布式训练

启动方式（在 mini_gpt_handwrite 目录下执行）：
  2 卡: accelerate launch mini_gpt_train_accelerate.py
  4 卡: accelerate launch mini_gpt_train_accelerate.py

或者先初始化 Accelerate 配置：
  accelerate config
  accelerate launch mini_gpt_train_accelerate.py

特点：
  - 彻底解决 NCCL 超时问题
  - 代码改动最少（相比 DDP）
  - 自动处理卡间数据同步
  - 无需手工配置 NCCL 环境变量
  - 模型保存格式与单卡版本完全一致
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

import gpt_model as gpt
from mini_gpt_train import (
    Tokenizer,
    TextDataSet,
    prepare_data,
    load_text,
    model_save,
    _DIR,
)


def train_accelerate():
    """
    使用 Accelerate 进行多卡分布式训练。

    核心改进：
      1. 自动处理所有卡间同步问题
      2. 无需 NCCL 环境变量配置
      3. 无需手工管理 DDP/梯度同步
      4. 代码改动最少
      5. 完全避免 NCCL 超时问题
    """
    # --- 1. 初始化 Accelerator ---
    accelerator = Accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"🚀 Accelerate 训练启动: 分布式类型 = {accelerator.distributed_type}")
        print(f"   设备数: {accelerator.num_processes}")
        if torch.cuda.is_available():
            print(f"   GPU 列表: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # --- 2. 语料准备（只主卡下载，其他卡等待） ---
    if is_main:
        prepare_data()

    # 确保所有卡都完成数据准备
    accelerator.wait_for_everyone()

    # --- 3. 数据加载 ---
    text = load_text(os.path.join(_DIR, 'train.txt'))
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    dataset = TextDataSet(text, 128)

    batch_size_per_gpu = 32  # 每张卡的 batch size

    # 注意：使用 Accelerate 时，不需要 DistributedSampler
    # Accelerate 会自动处理数据分布
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,  # 确保各卡 batch_size 一致
    )

    if is_main:
        total_batch = batch_size_per_gpu * accelerator.num_processes
        print(f"📊 vocab_size={vocab_size}")
        print(f"📊 每卡 batch_size={batch_size_per_gpu}, 总 effective batch_size={total_batch}")
        print(f"📊 数据集大小: {len(dataset)}")

    # --- 4. 模型加载 ---
    model = gpt.mini_gpt(vocab_size, 256, 4, 10, 128)

    # --- 5. 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- 关键：Accelerate 的 prepare() 方法 ---
    # 这一行替代了原来的 DDP 包装 + 手工同步配置
    # accelerate 会自动：
    #   - 根据配置选择合适的并行策略（DDP/FSDP/DeepSpeed）
    #   - 处理所有卡间同步（完全避免 NCCL 超时问题）
    #   - 管理梯度同步和分散
    #   - 处理数据分布和 sampler
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    if is_main:
        print(f"\n✅ 模型、优化器、数据加载器已准备完毕")
        print(f"   使用的后端: {accelerator.distributed_type}")

    # --- 6. 训练循环 ---
    epochs = 20
    pbar_epochs = tqdm(
        range(epochs),
        desc='训练进度',
        unit='epoch',
        disable=not is_main  # 只主卡显示进度条
    )

    try:
        for epoch in pbar_epochs:
            pbar_batch = tqdm(
                dataloader,
                desc=f'Epoch {epoch+1}/{epochs}',
                unit='batch',
                leave=False,
                disable=not is_main
            )
            epoch_loss = 0.0
            batch_count = 0

            for x, y in pbar_batch:
                # 数据已经由 accelerate 处理到正确的设备
                # 无需手工 .to(device)

                # --- 前向传播 ---
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                # --- 反向传播 ---
                # ✅ 这是唯一改动的地方：用 accelerator.backward() 替代 loss.backward()
                # accelerator 会自动处理：
                #   - 梯度同步（all-reduce）
                #   - 混合精度缩放
                #   - 梯度累积
                optimizer.zero_grad()
                accelerator.backward(loss)  # ← 关键改动（替代 loss.backward()）
                optimizer.step()

                # 使用 accelerator.gather() 收集所有卡的 loss（用于日志）
                loss_all = accelerator.gather(loss.detach()).mean()

                epoch_loss += loss_all.item()
                batch_count += 1

                if is_main:
                    pbar_batch.set_postfix(loss=f'{loss_all.item():.4f}')

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            if is_main:
                pbar_epochs.set_postfix(avg_loss=f'{avg_loss:.4f}')

    except Exception as e:
        if is_main:
            print(f"\n❌ 训练异常: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

    # --- 7. 保存模型（只主卡保存） ---
    if is_main:
        model_config = {
            'vocab_size': vocab_size,
            'embed_size': 256,
            'num_heads': 4,
            'num_layers': 10,
            'max_length': 128,
        }

        # 使用 accelerator.unwrap_model() 获取原始模型
        # 这替代了原来的 model.module
        raw_model = accelerator.unwrap_model(model)
        model_save(raw_model, tokenizer, model_config)
        print(f"✅ Accelerate 训练完成，模型已保存到 model.pth")


# ==================== 入口 ====================

if __name__ == '__main__':
    """
    Accelerate 自动处理启动

    启动方式：
      # 方式 1：直接启动（使用默认配置）
      accelerate launch mini_gpt_train_accelerate.py

      # 方式 2：先配置后启动（推荐）
      accelerate config           # 选择配置（DDP/FSDP/DeepSpeed等）
      accelerate launch mini_gpt_train_accelerate.py

      # 方式 3：指定 GPU 数
      accelerate launch --multi_gpu mini_gpt_train_accelerate.py

    不需要用 torchrun！Accelerate 会自动管理所有进程。
    """
    train_accelerate()
