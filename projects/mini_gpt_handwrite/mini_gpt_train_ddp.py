"""
mini_gpt DDP 多卡分布式训练

启动方式（在 mini_gpt_handwrite 目录下执行）：
  2 卡: torchrun --nproc_per_node=2 mini_gpt_train_ddp.py
  4 卡: torchrun --nproc_per_node=4 mini_gpt_train_ddp.py

注意：
  - 需要 NCCL 后端，仅支持 Linux（Windows 下需用 gloo 后端，但 GPU 训练推荐 Linux）
  - 每张卡的 batch_size=32，总 effective batch_size = 32 × 卡数
  - 训练完成后模型保存到 model.pth，与单卡版本格式完全一致，可直接用单卡代码加载
"""

import os

# ⚠️ 必须在 import torch 之前设置这些环境变量！
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟，而不是默认 600秒
os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 强制同步
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

import sys
import jieba
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from collections import Counter
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


# ==================== DDP 工具函数 ====================

def setup_ddp():
    """
    初始化 DDP 进程组。
    torchrun 会自动设置 LOCAL_RANK / RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT 等环境变量。
    """
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """销毁 DDP 进程组，释放通信资源"""
    dist.destroy_process_group()


# ==================== DDP 训练主函数 ====================

def train_ddp():
    """
    DDP 多卡分布式训练主函数。

    核心流程：
      1. 初始化进程组，确定当前卡的 local_rank
      2. 主卡（rank=0）负责下载语料，其他卡等待
      3. 用 DistributedSampler 将数据切分到各卡
      4. 模型用 DDP 包装，反向传播时自动同步梯度
      5. 只在主卡保存模型（去掉 module. 前缀，保证单卡兼容）
    """
    # --- 1. DDP 初始化 ---
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)  # 只有主卡打印日志 / 保存模型

    if is_main:
        print(f"🚀 DDP 训练启动: {world_size} 卡并行")
        print(f"   卡列表: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")

    # --- 2. 语料准备（只主卡下载，其他卡等待） ---
    if is_main:
        prepare_data()
    dist.barrier()  # 同步点：等主卡下载完，其他卡再继续

    # --- 3. 数据加载 ---
    text = load_text(os.path.join(_DIR, 'train.txt'))
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    dataset = TextDataSet(text, 128)

    batch_size_per_gpu = 32  # 每张卡的 batch size
    # DistributedSampler: 将数据集按 rank 切分，每卡拿到不同的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # ← 关键！确保各卡 batch_size 完全相同
    )
    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,  # 丢弃末尾不足 batch 的数据，避免各卡 batch size 不一致
    )

    if is_main:
        total_batch = batch_size_per_gpu * world_size
        print(f"📊 vocab_size={vocab_size}")
        print(f"📊 每卡 batch_size={batch_size_per_gpu}, 总 effective batch_size={total_batch}")
        print(f"📊 设备: {world_size} x {torch.cuda.get_device_name(local_rank)}")

    # --- 4. 模型加载 + DDP 包装 ---
    model = gpt.mini_gpt(vocab_size, 256, 4, 10, 128)
    model.to(device)

    # DDP 包装：关键参数优化
    # find_unused_parameters=False：避免每次 backward 扫描未使用参数（节省时间）
    # static_graph=True：固定计算图，加快 all-reduce
    # broadcast_buffers=False：禁用 buffer 同步，避免 epoch 开始的 BROADCAST 超时
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        static_graph=True,
        broadcast_buffers=False  # ← 关键！禁用 buffer 同步避免超时
    )

    # --- 5. 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- 6. 训练循环 ---
    epochs = 20
    pbar_epochs = tqdm(
        range(epochs), desc='训练进度', unit='epoch',
        disable=not is_main  # 只主卡显示进度条
    )
    try:
        for epoch in pbar_epochs:
            # 每个 epoch 必须调用 set_epoch，让 sampler 重新打乱数据
            # 否则各卡在每个 epoch 拿到的数据顺序都一样，失去随机性
            sampler.set_epoch(epoch)

            pbar_batch = tqdm(
                dataLoader, desc=f'Epoch {epoch+1}/{epochs}',
                unit='batch', leave=False, disable=not is_main
            )
            epoch_loss = 0
            batch_count = 0

            for x, y in pbar_batch:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # 前向传播
                logits = model(x)
                # 计算损失
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播（DDP 在这里自动执行跨卡梯度同步 all-reduce）
                loss.backward()
                # 更新参数（每张卡用相同的梯度更新，保持模型一致）
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                pbar_batch.set_postfix(loss=f'{loss.item():.4f}')

            avg_loss = epoch_loss / batch_count
            pbar_epochs.set_postfix(avg_loss=f'{avg_loss:.4f}')

    except Exception as e:
        if is_main:
            print(f"\n❌ 训练异常: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        cleanup_ddp()

    # --- 7. 保存模型（只主卡保存） ---
    if is_main:
        model_config = {
            'vocab_size': vocab_size,
            'embed_size': 256,
            'num_heads': 4,
            'num_layers': 10,
            'max_length': 128,
        }
        # 关键：DDP 包装后模型的 state_dict key 会多 'module.' 前缀
        # 用 model.module 取出原始模型再保存，保证单卡代码可以直接加载
        raw_model = model.module
        model_save(raw_model, tokenizer, model_config)
        print(f"✅ DDP 训练完成，模型已保存到 model.pth")


# ==================== 入口 ====================

if __name__ == '__main__':
    """
    必须用 torchrun 启动，不能直接 python mini_gpt_train_ddp.py
    
    示例：
      torchrun --nproc_per_node=2 mini_gpt_train_ddp.py   # 2 卡
      torchrun --nproc_per_node=4 mini_gpt_train_ddp.py   # 4 卡
    """
    # 安全检查：如果没有用 torchrun 启动，给出提示
    if 'LOCAL_RANK' not in os.environ:
        print("❌ 错误：必须用 torchrun 启动 DDP 训练！")
        print()
        print("正确用法：")
        print("  torchrun --nproc_per_node=2 mini_gpt_train_ddp.py   # 2 卡")
        print("  torchrun --nproc_per_node=4 mini_gpt_train_ddp.py   # 4 卡")
        print()
        print("错误用法：")
        print("  python mini_gpt_train_ddp.py  ← 这样不行")
        sys.exit(1)

    train_ddp()
