"""
mini_gpt DDP 多卡分布式训练

启动方式（在 mini_gpt_handwrite 目录下执行）：
  2 卡: torchrun --nproc_per_node=2 mini_gpt_train_ddp.py
  4 卡: torchrun --nproc_per_node=4 mini_gpt_train_ddp.py

注意：
  - 需要 NCCL 后端，仅支持 Linux（Windows 下需用 gloo 后端，但 GPU 训练推荐 Linux）
  - 每张卡的 batch_size=128，总 effective batch_size = 128 × 卡数
  - 训练完成后模型保存到 model.pth，与单卡版本格式完全一致，可直接用单卡代码加载
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

import gpt_model_sdpa as gpt
from mini_gpt_train import (
    Tokenizer,
    prepare_data,
    load_text,
    model_save,
    _DIR,
)


# ==================== 高效 DataSet ====================

class FastTextDataSet(torch.utils.data.Dataset):
    """
    复用外部已构建好的 tokenizer，避免每个进程重复 jieba 分词。
    原 TextDataSet.__init__ 会再创建一个 Tokenizer，在大语料下非常慢。
    """
    def __init__(self, id_tensor, block_size):
        self.id_tensor = id_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.id_tensor) - self.block_size

    def __getitem__(self, index):
        x = self.id_tensor[index : index + self.block_size]
        y = self.id_tensor[index + 1 : index + 1 + self.block_size]
        return x, y


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
        # 关键诊断：确认 CUDA 真的可用，否则全部在 CPU 跑会极慢
        print(f"   CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")

    # --- 2. 语料准备（只主卡下载，其他卡等待） ---
    if is_main:
        prepare_data()
    dist.barrier()  # 同步点：等主卡下载完，其他卡再继续

    # --- 3. 数据加载 ---
    text = load_text(os.path.join(_DIR, 'train.txt'))
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size
    # 先编码一次，传给 FastTextDataSet，避免内部再分词
    id_tensor = torch.tensor(tokenizer.encoder(), dtype=torch.long)
    dataset = FastTextDataSet(id_tensor, 256)

    batch_size_per_gpu = 128  # 每张卡的 batch size
    # K8s Pod 下 num_workers 会导致每个 worker 复制完整 Dataset 内存，引发内存翻倍
    # 使用 FastTextDataSet（只含 tensor，不分词）+ num_workers=0 避免此问题
    # DistributedSampler: 将数据集按 rank 切分，每卡拿到不同的数据子集
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
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
    model = gpt.mini_gpt_sdpa(vocab_size, 1024, 8, 16, 256)  # num_heads=8, head_dim=128（Flash Attention 最优）
    model.to(device)

    # torch.compile 与 DDP 在某些 GPU/驱动组合下会导致 NCCL 死锁
    # 暂不启用，如需尝试可取消注释
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    #     if is_main:
    #         print("🔥 torch.compile 已启用（compile → DDP 正确顺序）")

    # DDP 包装：gradient_as_bucket_view 省一份梯度拷贝
    # static_graph=True: 告诉 DDP 计算图固定，避免每次 step 重新探测未使用参数，减少 NCCL 异常
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, static_graph=True)

    # --- 5. 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # zero_grad(set_to_none=True) 比默认 zero_grad() 更快更省显存

    # --- 5.5 AMP 混合精度 ---
    scaler = torch.amp.GradScaler('cuda')
    if is_main:
        print("⚡ AMP 混合精度已启用（FP16 计算 + FP32 主权重）")

    # --- 6. 训练循环 ---
    epochs = 50
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
            epoch_loss = 0.0
            batch_count = 0

            for x, y in pbar_batch:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # 前向传播（AMP：Linear/matmul 自动用 FP16，LayerNorm/softmax 保持 FP32）
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                # 梯度清零（set_to_none 比 fill_(0) 更快更省显存）
                optimizer.zero_grad(set_to_none=True)
                # 反向传播（AMP：先放大 loss 防梯度下溢，再反向传播；DDP 自动 all-reduce）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 用 detach 避免每步 GPU→CPU 同步打断流水线，epoch 结束再统一算
                epoch_loss += loss.detach().float()
                batch_count += 1

            # epoch 结束一次性同步，计算平均 loss
            avg_loss = (epoch_loss / batch_count).item()
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
            'embed_size': 1024,
            'num_heads': 8,
            'num_layers': 16,
            'max_length': 256,
        }
        # 关键：DDP + compile 双层包装，需要逐层解包拿到原始模型
        # DDP → OptimizedModule(torch.compile) → 原始模型
        raw_model = model.module  # 解 DDP
        if hasattr(raw_model, '_orig_mod'):
            raw_model = raw_model._orig_mod  # 解 torch.compile
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
