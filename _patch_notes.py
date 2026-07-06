import re

path = r'd:\agent\model\Attention\attention_learning_notes.md'
content = open(path, encoding='utf-8').read()

# ── Change 1: Positional Encoding intuition ──────────────────────────────────
old1 = (
    "**位置向量公式**（固定值，非训练参数）：\n"
    "```\n"
    "t_i 的第 2k 维   = sin(i / 10000^(2k/d))\n"
    "t_i 的第 2k+1 维 = cos(i / 10000^(2k/d))\n"
    "i = 词位置，k = 维度索引，d = d_model\n"
    "```"
)
new1 = (
    "**位置向量公式**（固定值，非训练参数）：\n"
    "```\n"
    "t_i 的第 2k 维   = sin(i / 10000^(2k/d))\n"
    "t_i 的第 2k+1 维 = cos(i / 10000^(2k/d))\n"
    "i = 词位置，k = 维度索引，d = d_model\n"
    "```\n"
    "\n"
    "**直觉：每个位置产生独特的"指纹向量"**\n"
    "```\n"
    "低维度（k 小）→ 分母 10000^(2k/d) 小 → 频率高 → 相邻位置之间差异大（区分近距离）\n"
    "高维度（k 大）→ 分母 10000^(2k/d) 大 → 频率低 → 只有远距离位置才有明显差异（区分远距离）\n"
    "所有维度组合 → 每个位置的向量唯一，不会混淆\n"
    "```\n"
    "\n"
    "**⚠️ 实际代码中加位置编码前会对 token embedding 缩放：**\n"
    "```\n"
    "x = token_embedding × √d_model + positional_encoding\n"
    "```\n"
    "目的：token embedding 随机初始化后数值偏小，乘以 √d_model 让语义信息和位置信息\n"
    "处于相近的数值量级，避免位置编码的信号被语义信息\"淹没\"。"
)
assert old1 in content, 'old1 not found'
content = content.replace(old1, new1, 1)
print('Change 1 done')

# ── Change 2: FFN ReLU → GELU ─────────────────────────────────────────────────
old2 = (
    "### 5.2 FFN（Feed-Forward Network）\n"
    "\n"
    "两层全连接层：`FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂`\n"
    "- 512 → 2048（升维）+ ReLU → 2048 → 512（降维，无激活）\n"
    "- 逐位置独立（每个词单独做，词间不影响）\n"
    "- 分工：Self-Attention = 词间信息交流，FFN = 每个词自身信息加工"
)
new2 = (
    "### 5.2 FFN（Feed-Forward Network）\n"
    "\n"
    "两层全连接层：\n"
    "```\n"
    "原版 Transformer：FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂\n"
    "GPT 系列：       FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂  ← 用 GELU 替代 ReLU\n"
    "```\n"
    "- **GELU vs ReLU**：ReLU 在 x < 0 时硬截断为 0；GELU 是其平滑版本，x < 0 时\n"
    "  输出接近 0 但不完全为 0，梯度更平滑，训练更稳定\n"
    "- 512 → 2048（升维）+ GELU → 2048 → 512（降维，无激活）\n"
    "- **逐位置独立**（每个词单独做，词间不影响）\n"
    "- 分工：Self-Attention = 词间信息交流，FFN = 每个词自身信息深度加工（引入非线性表达能力）"
)
assert old2 in content, 'old2 not found'
content = content.replace(old2, new2, 1)
print('Change 2 done')

# ── Change 3: Add Pre-Norm vs Post-Norm, rename 5.5 → 5.6 ────────────────────
old3 = "### 5.5 Encoder 多层数据流"
new3 = (
    "### 5.5 Pre-Norm vs Post-Norm（现代 GPT 用哪种？）\n"
    "\n"
    "```\n"
    "原版 Transformer（Post-Norm）：          现代 GPT / LLaMA（Pre-Norm）：\n"
    "\n"
    "  x = LayerNorm(x + Attention(x))          x = x + Attention(LayerNorm(x))\n"
    "  x = LayerNorm(x + FFN(x))                x = x + FFN(LayerNorm(x))\n"
    "\n"
    "  残差在「里面」，LayerNorm 在「外面」      LayerNorm 在「里面」，残差在「外面」\n"
    "```\n"
    "\n"
    "**记忆口诀：**\n"
    "- Post-Norm：先做子层 + 残差，再 LayerNorm（Norm 在后）\n"
    "- Pre-Norm：先 LayerNorm，再做子层，再残差（Norm 在前）\n"
    "\n"
    "**为什么现代模型都用 Pre-Norm？**\n"
    "```\n"
    "Post-Norm：梯度必须穿过 LayerNorm 才能回传，深层网络容易训练不稳定\n"
    "Pre-Norm：残差路径是「高速公路」，梯度直接回传，不受 LayerNorm 干扰 → 更稳定\n"
    "```\n"
    "> ⚠️ 本笔记 5.1 节的图示是 Post-Norm（原版 Transformer）。实现 GPT 时应使用 Pre-Norm。\n"
    "\n"
    "---\n"
    "\n"
    "### 5.6 Encoder 多层数据流"
)
assert old3 in content, 'old3 not found'
content = content.replace(old3, new3, 1)
print('Change 3 done')

# ── Change 4: Add label shift vs residual note in 8.1 ────────────────────────
old4 = (
    "### 8.1 完整训练流程\n"
    "\n"
    "```\n"
    "1. 拿一批翻译对（法语→英语）\n"
    "2. 法语 → Encoder×6 → R（前向传播）\n"
    "3. 英语右移一位 → Decoder×6 → Z → Linear → Softmax → 概率（前向传播）\n"
    "4. 算 Loss（交叉熵）\n"
    "5. Loss.backward()（PyTorch 自动反向传播，算出所有梯度）\n"
    "6. optimizer.step()（用梯度更新所有参数）\n"
    "7. 重复，直到 Loss 足够小\n"
    "```"
)
new4 = (
    "### 8.1 完整训练流程\n"
    "\n"
    "```\n"
    "1. 拿一批翻译对（法语→英语）\n"
    "2. 法语 → Encoder×6 → R（前向传播）\n"
    "3. 英语右移一位 → Decoder×6 → Z → Linear → Softmax → 概率（前向传播）\n"
    "4. 算 Loss（交叉熵）\n"
    "5. Loss.backward()（PyTorch 自动反向传播，算出所有梯度）\n"
    "6. optimizer.step()（用梯度更新所有参数）\n"
    "7. 重复，直到 Loss 足够小\n"
    "```\n"
    "\n"
    "**⚠️ 术语澄清：「标签右移」≠「残差连接」**\n"
    "```\n"
    "「标签右移一位」（Label Shift）：\n"
    "  输入 [t0, t1, t2, t3]  对应  标签 [t1, t2, t3, t4]\n"
    "  是训练数据的构造方式：让每个位置「预测下一个词」\n"
    "  → 这是数据对齐，和网络结构无关\n"
    "\n"
    "「残差连接」（Residual Connection）：\n"
    "  output = x + SubLayer(x)，把子层输入直接加回到输出\n"
    "  → 这是 Transformer Block 内部的结构，作用是稳定梯度传播\n"
    "\n"
    "两者字面上都有「偏移/差」的含义，但完全不同，不要混淆。\n"
    "```"
)
assert old4 in content, 'old4 not found'
content = content.replace(old4, new4, 1)
print('Change 4 done')

open(path, 'w', encoding='utf-8').write(content)
print('All done, file saved.')
