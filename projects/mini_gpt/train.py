"""
Mini-GPT 训练脚本
=================

用字符级别（char-level）的小数据集训练 Mini-GPT。
不需要 GPU，CPU 几分钟就能训练出一个能生成"像样文本"的模型。

训练流程（和你笔记中的传统 NN 完全一样）：
    1. 准备数据：把文本切成 token 序列
    2. 前向传播：model(input) → 预测下一个 token
    3. 算 Loss：Cross-Entropy（预测分布 vs 真实分布）
    4. 反向传播：算梯度
    5. 更新参数：梯度下降

用法：
    python train.py                    # 使用默认莎士比亚文本
    python train.py --text "你自己的文本"  # 使用自定义文本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(__file__))
from model import MiniGPT


# ============================================================
# 1. 字符级别的 Tokenizer（最简单的版本）
# ============================================================

class CharTokenizer:
    """
    字符级别的 Tokenizer：每个字符就是一个 token。
    
    和真实 LLM 的区别：
    - 真实 LLM 用 BPE/SentencePiece（把常见词组合成一个 token）
    - 我们用字符级别，简单直观，vocab_size 通常只有 100~200
    
    流程：
    "hello" → [h, e, l, l, o] → [7, 4, 11, 11, 14]（数字ID）
    """
    
    def __init__(self, text: str):
        # 从文本中提取所有不重复的字符，排序
        chars = sorted(list(set(text)))
        
        # 添加特殊 token
        self.pad_token = '<PAD>'
        chars = [self.pad_token] + chars
        
        # 建立 char ↔ id 的映射
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> list:
        """文本 → token ID 列表"""
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, ids: list) -> str:
        """token ID 列表 → 文本"""
        return ''.join(self.id_to_char.get(i, '?') for i in ids if i != 0)


# ============================================================
# 2. 数据集
# ============================================================

class CharDataset(Dataset):
    """
    把文本切成固定长度的片段，用于训练。
    
    训练数据构造：
        输入: [t0, t1, t2, ..., t_{n-1}]    → 模型预测
        标签: [t1, t2, t3, ..., t_n]        → 真实的下一个 token
    
    这就是 GPT 的训练方式：给定前面的 token，预测下一个 token。
    Loss = CrossEntropy(模型预测, 真实下一个token)
    """
    
    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 把整个文本编码成 token ID
        self.data = tokenizer.encode(text)
        
        # 过滤掉太短的文本
        self.n_samples = max(0, len(self.data) - block_size - 1)
        
        print(f"数据集信息:")
        print(f"  文本长度: {len(self.data)} tokens")
        print(f"  词汇表大小: {tokenizer.vocab_size}")
        print(f"  块大小 (block_size): {block_size}")
        print(f"  训练样本数: {self.n_samples}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 取一段文本
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # 输入 = 前 block_size 个 token
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # 标签 = 后 block_size 个 token（向右移一位）
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


# ============================================================
# 3. 训练循环
# ============================================================

def train(model, dataloader, epochs, learning_rate, device, eval_interval=50):
    """
    训练模型（和传统 NN 训练流程完全一样）
    
    1. 前向传播 → 算 logits
    2. CrossEntropy Loss
    3. 反向传播 → 更新 W_Q/W_K/W_V + FFN + Embedding 参数
    """
    model.to(device)
    model.train()
    
    # AdamW 优化器（LLM 训练的标准选择）
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # 学习率调度器（余弦衰减）
    total_steps = epochs * len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    losses = []
    step = 0
    
    print(f"\n开始训练...")
    print(f"  设备: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  每 epoch 步数: {len(dataloader)}")
    print(f"  总步数: {total_steps}")
    print(f"  学习率: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # ---- 前向传播 ----
            logits = model(x)  # (batch, seq_len, vocab_size)
            
            # ---- 算 Loss ----
            # CrossEntropy 期望 (N, C) 和 (N,)，需要 reshape
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
                y.view(-1),                         # (batch*seq_len,)
            )
            
            # ---- 反向传播 + 更新参数 ----
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            # 定期打印
            if step % eval_interval == 0:
                avg_loss = sum(epoch_losses[-eval_interval:]) / min(eval_interval, len(epoch_losses))
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step:>5d} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # Epoch 结束
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        elapsed = time.time() - epoch_start
        losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"平均 Loss: {avg_epoch_loss:.4f} | "
              f"耗时: {elapsed:.1f}s")
    
    print("-" * 50)
    print(f"训练完成！最终 Loss: {losses[-1]:.4f}")
    
    return losses


# ============================================================
# 4. 文本生成
# ============================================================

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 40, device: str = 'cpu') -> str:
    """
    自回归生成文本（GPT 的核心工作方式）
    
    生成流程：
    1. 把 prompt 编码成 token IDs
    2. 送入模型，得到最后一个位置的 logits
    3. 从 logits 中采样下一个 token（temperature + top_k）
    4. 把新 token 追加到序列末尾
    5. 重复 2-4，直到生成 max_new_tokens 个 token
    
    参数说明：
    - temperature: 控制生成的"随机性"
        0.1 = 非常确定（接近贪心）
        0.8 = 平衡创意和连贯
        1.5 = 很随机
    - top_k: 只从概率最高的 k 个 token 中采样
    """
    model.eval()
    
    # 编码 prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    generated = list(input_ids[0].cpu().numpy())
    
    for _ in range(max_new_tokens):
        # 截断到 max_len
        context = input_ids[:, -model.max_len:]
        
        # 前向传播
        logits = model(context)  # (1, seq_len, vocab_size)
        
        # 取最后一个位置的 logits
        next_logits = logits[:, -1, :]  # (1, vocab_size)
        
        # Temperature 缩放
        next_logits = next_logits / temperature
        
        # Top-K 过滤：只保留概率最高的 k 个
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_k_values[:, [-1]]] = float('-inf')
        
        # Softmax → 概率分布
        probs = torch.softmax(next_logits, dim=-1)
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        
        # 追加到序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated.append(next_token.item())
    
    return tokenizer.decode(generated)


# ============================================================
# 5. 主程序
# ============================================================

def get_default_text():
    """获取默认训练文本（小型莎士比亚文本片段）"""
    return """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.
"""


def main():
    parser = argparse.ArgumentParser(description="Mini-GPT 训练脚本")
    parser.add_argument('--text', type=str, default=None, help='训练文本（不提供则用默认莎士比亚文本）')
    parser.add_argument('--text-file', type=str, default=None, help='训练文本文件路径')
    parser.add_argument('--d-model', type=int, default=128, help='模型维度')
    parser.add_argument('--n-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--n-layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--block-size', type=int, default=64, help='训练序列长度')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--save-path', type=str, default=None, help='模型保存路径')
    args = parser.parse_args()
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 获取训练文本
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = get_default_text()
    
    print("=" * 60)
    print("Mini-GPT 训练")
    print("=" * 60)
    print(f"文本预览: {text[:100]}...")
    
    # 创建 Tokenizer
    tokenizer = CharTokenizer(text)
    print(f"词汇表: {tokenizer.vocab_size} 个字符")
    
    # 创建数据集
    dataset = CharDataset(text, tokenizer, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # 创建模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.block_size + 16,
    )
    
    print(f"\n模型配置:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_heads: {args.n_heads} (d_k = {args.d_model // args.n_heads})")
    print(f"  n_layers: {args.n_layers}")
    print(f"  参数量: {model.get_num_params():,}")
    print(f"  参数大小: {model.get_param_size_mb():.2f} MB")
    
    # 训练
    losses = train(
        model, dataloader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )
    
    # 保存模型
    save_path = args.save_path or os.path.join(
        os.path.dirname(__file__), 'mini_gpt_checkpoint.pt'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_char_to_id': tokenizer.char_to_id,
        'tokenizer_id_to_char': tokenizer.id_to_char,
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'max_len': args.block_size + 16,
        },
        'final_loss': losses[-1],
    }, save_path)
    print(f"\n模型已保存到: {save_path}")
    
    # 生成测试
    print("\n" + "=" * 60)
    print("生成测试")
    print("=" * 60)
    
    prompts = ["First Citizen:\n", "Second Citizen:\n", "All:\n"]
    
    for prompt in prompts:
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=150,
            temperature=0.8,
            top_k=40,
            device=device,
        )
        print(f"\n--- Prompt: '{prompt.strip()}' ---")
        print(generated)
        print()


if __name__ == "__main__":
    main()
