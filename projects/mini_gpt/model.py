"""
Mini-GPT: 从零实现一个 Decoder-Only Transformer
=================================================

这个项目把 Attention 笔记中的理论变成可运行的代码。
每个模块对应你笔记中的一个章节，代码中有详细注释对应关系。

模块结构：
    1. MultiHeadAttention  → 对应笔记 §8 (Multi-Head Attention)
    2. TransformerBlock    → 对应笔记 §5 (FFN) + 残差连接 + LayerNorm
    3. MiniGPT             → 完整模型：Embedding + 位置编码 + Transformer Block堆叠 + 输出层

用法：
    model = MiniGPT(vocab_size=100, d_model=128, n_heads=4, n_layers=4)
    logits = model(input_ids)  # (batch, seq_len, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Multi-Head Attention（对应笔记 §8）
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    对应你笔记中的计算流程：
    - 每个 Head 有独立的 W_Q, W_K, W_V（随机初始化，训练后学到不同模式）
    - 8个Head各算一次 Attention，然后 Concat + W_O 合并
    
    这里用一个"等效实现"：不是循环8次，而是一次矩阵运算搞定所有Head。
    原理：把 d_model 维度拆成 (n_heads, d_k)，然后转置让每个Head独立计算。
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个 Head 的维度（笔记中的 64）
        
        # 三个投影矩阵（笔记中的 W_Q, W_K, W_V）
        # 注意：这里用一个大的 Linear 代替 8 个小的，效果一样，计算更快
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影矩阵（笔记中的 W_O）
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)  输入序列
        mask: (seq_len, seq_len) 因果遮罩（后面会讲）
        
        返回: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # ---- Step 1: 投影到 Q, K, V（对应笔记 Step 1）----
        # 这里一次性算出所有 Head 的 Q/K/V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # ---- Step 2: 拆分成多个 Head ----
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # ---- Step 3: 计算注意力分数（对应笔记 Step 2-3）----
        # Q × K^T / √d_k
        # (batch, n_heads, seq_len, d_k) × (batch, n_heads, d_k, seq_len)
        # = (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # ---- Step 4: 因果遮罩（GPT 特有！）----
        # 让每个 token 只能看到自己和前面的 token，不能看到后面的
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ---- Step 5: Softmax 归一化（对应笔记 Step 4）----
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        
        # ---- Step 6: 加权求和（对应笔记 Step 5-6）----
        # attn_weights × V
        output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        
        # ---- Step 7: 合并所有 Head（对应笔记 Concat + W_O）----
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 通过 W_O 融合
        output = self.W_o(output)  # (batch, seq_len, d_model)
        
        return output


# ============================================================
# 2. Transformer Block（对应笔记 §5 FFN + 残差连接 + LayerNorm）
# ============================================================

class TransformerBlock(nn.Module):
    """
    一个 Transformer Decoder Block
    
    结构（Pre-Norm 版本，现代 GPT 都用这个）：
    
        x → LayerNorm → Multi-Head Attention → + x（残差连接）
          → LayerNorm → FFN → + x（残差连接）
          → 输出
    
    你笔记中还没学到的两个组件：
    
    1. 残差连接（Residual Connection）：
       - 把输入直接加到输出上：output = x + Sublayer(x)
       - 作用：让梯度能直接回传，解决深层网络训练困难的问题
       - 类比：高速公路的"直达通道"，不用走每一层的"收费站"
    
    2. Layer Normalization：
       - 对每个 token 的向量做归一化（均值=0，方差=1）
       - 作用：稳定训练，防止数值爆炸
       - Pre-Norm：放在 Sublayer 前面（GPT-2/LLaMA 的做法）
       - Post-Norm：放在 Sublayer 后面（原版 Transformer 的做法）
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # FFN 中间层维度，通常是 d_model 的 4 倍（笔记中的 512→2048）
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        # FFN：两层全连接（对应笔记 §5）
        # 第一层：d_model → d_ff（升维）+ GELU 激活
        # 第二层：d_ff → d_model（降维回来）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU 是 ReLU 的平滑版本，GPT 系列都用这个
            nn.Linear(d_ff, d_model),
        )
        
        # Layer Normalization（Pre-Norm）
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # Self-Attention + 残差连接
        x = x + self.attention(self.ln1(x), mask)
        
        # FFN + 残差连接
        x = x + self.ffn(self.ln2(x))
        
        return x


# ============================================================
# 3. 位置编码（对应笔记 §9 Positional Encoding）
# ============================================================

def create_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成正弦位置编码（对应笔记 §9.4 的公式）
    
    公式：
        PE(pos, 2k)   = sin(pos / 10000^(2k/d))
        PE(pos, 2k+1) = cos(pos / 10000^(2k/d))
    
    pos = 词的位置（第几个词）
    k = 维度的索引
    d = 向量维度（d_model）
    
    返回: (max_len, d_model) 的位置编码矩阵
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    
    # 计算分母：10000^(2k/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    
    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos / 10000^(2k/d))
    pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos / 10000^(2k/d))
    
    return pe  # (max_len, d_model)


# ============================================================
# 4. 因果遮罩（Causal Mask）—— GPT 特有的！
# ============================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    创建因果遮罩矩阵（下三角矩阵）
    
    为什么需要？
    GPT 是自回归模型：生成第 i 个 token 时，只能看到第 0~i 个 token。
    如果不加遮罩，Attention 会让每个 token 看到"未来"的信息，
    训练时就会"作弊"（数据泄露），学不到东西。
    
    遮罩矩阵（以 seq_len=4 为例）：
    
        t0  t1  t2  t3
    t0 [ 1   0   0   0 ]   ← t0 只能看自己
    t1 [ 1   1   0   0 ]   ← t1 能看 t0 和自己
    t2 [ 1   1   1   0 ]   ← t2 能看 t0, t1 和自己
    t3 [ 1   1   1   1 ]   ← t3 能看所有
    
    0 的位置会被填 -inf，softmax 后变成 0，等于"看不到"。
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
    return mask  # (seq_len, seq_len)


# ============================================================
# 5. Mini-GPT 完整模型
# ============================================================

class MiniGPT(nn.Module):
    """
    完整的 Mini-GPT 模型（Decoder-Only Transformer）
    
    对应你笔记中的架构：
    输入 → Token Embedding + Positional Encoding → [Transformer Block × N] → LayerNorm → 输出
    
    参数说明（以你的笔记中的原版 Transformer 为参考）：
    - vocab_size: 词汇表大小（原版是 37000+，我们用小的）
    - d_model:    模型维度（原版 512，我们用 128~256）
    - n_heads:    注意力头数（原版 8，我们用 4）
    - n_layers:   Transformer Block 层数（原版 6，我们用 4）
    - max_len:    最大序列长度
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token Embedding（对应笔记 §9.3 的语义 Embedding）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（对应笔记 §9.4，固定的 sin/cos 值，不参与训练）
        self.register_buffer(
            'pos_encoding',
            create_positional_encoding(max_len, d_model)
        )
        
        # 堆叠 N 个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # 最终 LayerNorm（GPT 系列的标准做法）
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出层：投影到 vocab_size（预测下一个 token 的概率分布）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（Weight Tying）：让 lm_head 和 token_embedding 共享权重
        # 这是 GPT-2 的优化技巧，能大幅减少参数量
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重（标准正态分布，缩放因子 0.02）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)  token ID 序列
        返回: (batch, seq_len, vocab_size)  每个位置的下一个 token 预测 logits
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len, f"序列长度 {seq_len} 超过最大值 {self.max_len}"
        
        # ---- Step 1: Embedding + 位置编码（对应笔记 §9.2）----
        # token_emb: 查表得到语义向量
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # pos_emb: 获取位置向量
        pos_emb = self.pos_encoding[:seq_len]  # (seq_len, d_model)
        
        # 叠加（笔记中的 "语义向量 + 位置向量"）
        # 乘以 √d_model 是为了让 embedding 的数值范围大于位置编码
        x = token_emb * math.sqrt(self.d_model) + pos_emb  # (batch, seq_len, d_model)
        
        # ---- Step 2: 创建因果遮罩 ----
        mask = create_causal_mask(seq_len).to(input_ids.device)
        
        # ---- Step 3: 通过所有 Transformer Block ----
        for block in self.blocks:
            x = block(x, mask)
        
        # ---- Step 4: 最终归一化 + 输出 ----
        x = self.ln_f(x)  # (batch, seq_len, d_model)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_param_size_mb(self) -> float:
        """获取模型参数占用的内存（MB）"""
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 / 1024


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mini-GPT 模型测试")
    print("=" * 60)
    
    # 创建一个小模型
    model = MiniGPT(
        vocab_size=256,   # 256 个 token（字符级别）
        d_model=128,      # 模型维度 128
        n_heads=4,        # 4 个注意力头
        n_layers=4,       # 4 层 Transformer Block
        max_len=128,      # 最大序列长度 128
    )
    
    print(f"\n模型参数: {model.get_num_params():,}")
    print(f"参数大小: {model.get_param_size_mb():.2f} MB")
    
    # 测试前向传播
    input_ids = torch.randint(0, 256, (2, 32))  # batch=2, seq_len=32
    logits = model(input_ids)
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"输出含义: batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]}")
    
    # 验证因果遮罩
    mask = create_causal_mask(5)
    print(f"\n因果遮罩矩阵 (5×5):")
    print(mask)
    
    print("\n✅ Mini-GPT 模型测试通过！")
