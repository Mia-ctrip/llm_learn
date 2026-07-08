"""
mini_gpt SDPA 版本 —— 使用 F.scaled_dot_product_attention 替代手写 attention

与原始 gpt_model.py 的区别：
  1. MultiHeadAttentionSDPA 使用 F.scaled_dot_product_attention 替代手写 attention
  2. 不需要手动构造 mask，is_causal=True 自动处理因果掩码
  3. decoder_layer_sdpa 不需要传 mask 参数
  4. mini_gpt_sdpa 的 forward 不需要动态生成 mask

其他部分（Embedding、FFN、output_layer、positional_encoding）从 gpt_model.py 复用

为什么用 SDPA：
  - 手写 attention 是 3 个独立 kernel launch（QK^T、softmax、×V），S/P 矩阵必须落 HBM
  - SDPA 是 1 个融合 kernel，PyTorch 自动选 Flash Attention backend
  - Flash Attention 用 Tiling 让 S/P 矩阵留在 SRAM，不落 HBM
  - 更容易触发 Tensor Core，训练速度和显存都有显著提升

切换方式：
  训练脚本中把 gpt.mini_gpt 换成 gpt_model_sdpa.mini_gpt_sdpa 即可
  参数结构完全一致，训练出的模型可以用原始 mini_gpt 加载
"""

import torch
from torch import nn
import torch.nn.functional as F

from gpt_model import (
    EmbeddingLayer,
    FFN,
    output_layer,
    positional_encoding,
)


# ==================== SDPA Attention ====================

def sdpa_attention(q, k, v):
    """
    SDPA attention（生产训练用）
    - 1 个融合 kernel，PyTorch 自动选 Flash Attention / Memory Efficient backend
    - S/P 矩阵不落 HBM（Tiling）
    - 更容易触发 Tensor Core
    - 输入必须 4D: (batch, num_heads, seq_len, head_dim)
    - is_causal=True 自动应用下三角 causal mask，不需要手动构造
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ==================== SDPA 模型组件 ====================

class MultiHeadAttentionSDPA(nn.Module):
    """
    SDPA 版多头注意力
    与原始 MultiHeadAttention 的区别：
      - forward 中使用 sdpa_attention() 替代手写 attention()
      - 不需要 mask 参数，is_causal=True 自动处理因果掩码
      - 触发 Flash Attention，S/P 不落 HBM，Tensor Core 利用率更高
    """
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # reshape 为 4D: (batch, num_heads, seq_len, head_dim)
        # 这是触发 Flash Attention 的必要条件
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA：融合 kernel，自动处理 causal mask + 缩放（÷√d_k）
        z = sdpa_attention(Q, K, V)
        # 合并多头: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, embed_size)
        z = z.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        return self.w_o(z)


class decoder_layer_sdpa(nn.Module):
    """SDPA 版 Decoder 层，attention 不需要 mask 参数"""
    def __init__(self, embed_size, num_heads, hidden_size):
        super().__init__()
        self.attention = MultiHeadAttentionSDPA(embed_size, num_heads)
        self.ffn = FFN(embed_size, hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Pre-Norm，不需要 mask（is_causal=True 已在 SDPA 内部处理）
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class mini_gpt_sdpa(nn.Module):
    """
    SDPA 版 mini_gpt（生产训练用）
    与原始 mini_gpt 的区别：
      - 使用 decoder_layer_sdpa（内部用 SDPA attention）
      - forward 不需要动态生成 mask
    其他部分（Embedding、FFN、output_layer、positional_encoding）从 gpt_model.py 复用
    """
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_size)
        self.register_buffer('pos_enc', positional_encoding(embed_size, max_length))
        self.decoders = nn.ModuleList([
            decoder_layer_sdpa(embed_size, num_heads, embed_size * 4)
            for _ in range(num_layers)
        ])
        self.output_layer = output_layer(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.pos_enc[:, :x.size(1), :] + x
        # 不需要 mask，SDPA 的 is_causal=True 自动处理
        for layer in self.decoders:
            x = layer(x)
        x = self.output_layer(x)
        return x
