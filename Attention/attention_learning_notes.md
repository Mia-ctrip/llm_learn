# Attention & Transformer 学习笔记

> 基于 Jay Alammar "Illustrated Transformer" 精读 + Q&A 整理
https://jalammar.github.io/illustrated-transformer/
> 学习日期：2026-06-04 ~ 06-05

---

## 第一章：Transformer 整体架构

### 1.1 原版 Transformer（2017）
```
输入（法语）→ Encoder(×6) → Decoder(×6) → 输出（英语）
```
- Encoder 和 Decoder 各堆叠 6 层（数字可调）
- 每层 Encoder = Self-Attention + FFN
- 每层 Decoder = Masked Self-Attention + Cross-Attention + FFN

### 1.2 现在的三种变体

| 架构 | 代表模型 | 结构 | 用途 |
|------|---------|------|------|
| Encoder-Decoder | T5, BART | 完整 Encoder + Decoder | 翻译、摘要 |
| **Decoder-Only** | **GPT, Claude, Qwen, LLaMA** | **只有 Decoder（无 Encoder、无 Cross-Attention）** | **通用大模型（主流）** |
| Encoder-Only | BERT | 只有 Encoder | 文本理解（不生成） |

> 广义"Transformer"= 以 Self-Attention 为核心的所有变体。GPT 没有 Encoder 和 Cross-Attention，但仍属于 Transformer 家族。

### 1.3 Transformer vs RNN

| | RNN | Transformer |
|---|---|---|
| 记忆方式 | 隐藏状态 h 逐步累积，距离越远衰减越严重 | 每个词直接访问所有词，无衰减 |
| 计算方式 | 串行（必须按顺序，GPU 空转） | 并行（矩阵运算，GPU 满载） |
| 远距离依赖 | 差 | 好（一步直达） |

---

## 第二章：输入处理（从文本到向量）

### 2.1 输入的完整流程

```
原始文本 "i am a student"
       ↓ tokenize（查词表）
[5, 6, 7, 8]                ← token id（常量整数）
       ↓ embedding（查 E）
(4×d_model) 语义向量       ← 会随 E 更新而变化
       ↓ 加位置编码（固定 sin/cos）
(4×d_model) 最终输入       ← 送进第1层 Encoder
```

### 2.2 词表（Vocabulary）与 Token ID

训练前扫描所有训练数据，给每个词分配一个编号：

```
vocab = {
  "<PAD>":  0,   ← 填充
  "<BOS>":  1,   ← 句子开始
  "<EOS>":  2,   ← 句子结束
  "<UNK>":  3,   ← 未知词
  "i":      5,
  "am":     6,
  ...
  "xyz":    49999
}
vocab_size = 50000

实际中用 BPE 等子词分词，本质都是"建立词→编号的映射表"
```

**Token ID 是常量**：这才是真正的"输入"（类似独热编码的查表索引，只是不展开）。

### 2.3 Embedding 矩阵 E

```
E 是一个 (vocab_size × d_model) 的矩阵，随机初始化

例：E = (50000 × 512)
E[0] = [0.02, -0.15, ...]   ← 第0行：<PAD> 的向量
E[5] = [0.31, 0.05, ...]   ← 第5行："i" 的向量
E[6] = [-0.19, 0.42, ...]  ← 第6行："am" 的向量
...

总参数量：50000 × 512 = 2560万（和 W_Q/W_K/W_V 一样是训练参数）
```

**E 是参数，会被反向传播更新**：
```
查表本质是矩阵乘法：
  E[5] = E × one_hot(5)
  既然是矩阵乘法 → 能算梯度 → 能反向传播

反向传播时：
  只有 batch 里出现过的词，对应的 E 行才会被更新
  没出现的词对应行不动

训练后：语义相近的词（cat/dog）在 E 中对应行会靠近
  原因：相似语境 → 梯度更新方向相似 → 向量靠近
```

### 2.4 d_model 的含义

**d_model = 模型内部所有向量的"通用维度"**（Embedding、各层输入输出、位置编码都是这个维度）。

| 模型 | d_model |
|------|---------|
| 原版 Transformer | 512 |
| BERT-base / GPT-2 | 768 |
| LLaMA-2-7B | 4,096 |
| LLaMA-2-70B | 8,192 |
| GPT-3 | 12,288 |

```
d_model 翻倍 → 模型参数近似 4 倍增长（W_Q/W_K/W_V/FFN 都是 d×d 矩阵）
d_k = d_model / num_heads（例 512/8=64，是每个头的 Q/K/V 维度）
```

### 2.5 位置编码（Positional Encoding）

Self-Attention 本身"无序"——"cat sat" 和 "sat cat" 输出一样。需要位置编码给模型"位置感"。

**位置向量公式**（固定值，非训练参数）：
```
t_i 的第 2k 维   = sin(i / 10000^(2k/d))
t_i 的第 2k+1 维 = cos(i / 10000^(2k/d))
i = 词位置，k = 维度索引，d = d_model
```

### 2.6 输入 = 语义 + 位置（直接相加）

```
token_ids = [5, 6, 7, 8]
  ↓ 查 E
E[5], E[6], E[7], E[8]      ← d_model 维语义向量
  ↓ 加位置编码
x₁ = E[5] + t₀
x₂ = E[6] + t₁
x₃ = E[7] + t₂
x₄ = E[8] + t₃

输入矩阵 X = (4 × d_model) → 送入第1层 Encoder

只在第0层加位置编码，后续层不需要（位置信息已融入向量）
Embedding 阶段词与词独立，交叉发生在 Self-Attention 阶段
```

| | 语义 Embedding | 位置 Encoding |
|---|---|---|
| 来源 | 查矩阵 E（训练参数） | sin/cos 公式（固定值） |
| 是否训练 | 是 | 否 |
| 维度 | d_model | d_model |
| 存储 | (vocab_size, d_model) | (max_seq_len, d_model) |

### 2.7 两种范式对比：预训练编码 vs 端到端训练

| 范式 | E 是什么 | E 是否可训练 |
|------|----------|-----------------|
| 传统（BERT 编码 + GRU/LSTM） | 用 BERT 输出当输入 | 冻结（常量） |
| **Transformer 论文 / GPT 训练** | **随机初始化的参数矩阵** | **可训练（变量）** |
| 调用别人训练好的 LLM | 已训练好的 E | 常量（推理不更新） |

```
判断方法：
  "这个模型是别人训练好的，我直接拿来用？"
  YES → E 对你是常量
  NO，我要从零训练 → E 是要学的参数
```

### 2.8 Embedding API 的本质

```
Chat API:    输入 → 整个模型跑 → 输出生成文本
Embedding API: 输入 → 跑部分/全部层 → 输出句子级向量

重要区别：Embedding API 返回的不是 E 中查表的原始向量！
         而是整个模型处理后的"句子级"向量（含上下文）

  E["bank"]              ← 原始向量，分不清"银行"还是"河岸"
  模型输出的 "river bank" 中 bank 的向量  ← 知道是"河岸"
```

**用途**：语义搜索、RAG、聚类分类、推荐系统。

### 2.9 Embedding 显存占用

```
Embedding 参数量 = vocab_size × d_model。与训练集大小无关（vocab 预先由分词器出限定）。

主流模型 Embedding 占总参数 1~3%，训练显存大头是：
  模型参数 + 梯度 + 优化器状态（Adam 要存 m、v） + 激活值 + KV Cache

例：LLaMA-2-7B 的 Embedding 仅 ~250MB（FP16），占总参数 1.9%
```

---

## 第三章：Self-Attention 核心机制

### 3.1 核心思想

让每个词在编码时能"看到"句子里的所有其他词，把相关信息融入自己的表示。

例："The animal didn't cross the street because **it** was too tired"
→ 处理 "it" 时，发现 "it" 和 "animal" 最相关，把 "animal" 的信息融入 "it"

### 3.2 Q/K/V 的本质

**两个视角：**

```
矩阵视角（层级别，训练参数）：
  W_Q, W_K, W_V ← 每层一套，随机初始化，训练后固定
  初始化和计算方式完全一样，区别在公式中的角色不同：
    Q = "提问"，K = "被匹配"，V = "被取出的内容"
  角色不同 → 梯度不同 → 训练后学到不同的值

向量视角（词级别，中间计算结果）：
  N 个词 → N 个 q、N 个 k、N 个 v
  q_i = W_Q · x_i，k_i = W_K · x_i，v_i = W_V · x_i
```

### 3.3 计算步骤（6步）

```
Step 1: 生成 Q、K、V      q = W_Q·x, k = W_K·x, v = W_V·x
Step 2: 计算注意力分数     score = q · k（点积，越大越相关）
Step 3: 缩放              score / √d_k（防止值太大导致 softmax 梯度消失）
Step 4: Softmax           归一化为概率（所有权重为正，加起来=1）
Step 5: 加权乘 V          attention_weight × v
Step 6: 求和              output = Σ(weighted_v)
```

**核心公式：**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

### 3.4 矩阵形式（一次算完所有词）

```
Q (N×d_k) × K^T (d_k×N) = 注意力分数矩阵 (N×N)

每行 = 一个词对所有词的注意力分数
÷ √d_k → softmax → 注意力权重矩阵 (N×N)
注意力权重矩阵 (N×N) × V (N×d_k) = 输出矩阵 (N×d_k)

K 要转置：矩阵乘法规则（A的行 × B的列），转置后 K 的列 = 每个词的 k 向量
```

**输出结果**：N×d_k 矩阵，每行是对应词的"加强向量"（融合了其他词的信息）。

### 3.5 维度说明

```
输入 embedding: 512 维
W_Q, W_K, W_V: 512 × 512（即 d_model × d_model）
Multi-Head 8 头: 每个头的 d_k = d_model / 8 = 64
详见第四章 4.5 节（实现机制详解）
```

---

## 第四章：Multi-Head Attention（多头注意力）

### 4.1 核心思想

在一层 Attention 里用 8 套独立的 W_Q/W_K/W_V，算 8 次 Attention，最后合并。

### 4.2 计算流程

```
输入 X (N×512)
  ├→ Head 1: 独立的 W_Q¹/W_K¹/W_V¹ → Z₁ (N×64)
  ├→ Head 2: 独立的 W_Q²/W_K²/W_V² → Z₂
  ├→ ...
  └→ Head 8: 独立的 W_Q⁸/W_K⁸/W_V⁸ → Z₈
  ↓
Concat(Z₁,...,Z₈) → (N×512)
  ↓
× W_O (512×512)  ← 输出权重矩阵（训练参数，学会混合8个头的信息）
  ↓
最终输出 (N×512) → 传给 FFN
```

### 4.3 为什么多头能学到不同模式？

公式相同，但每套 W 的**随机初始值不同** → 输出不同 → 梯度不同 → 越来越不同。每个 Head 走到不同的局部最优，学到不同类型的关系。**故意利用随机初始化的差异性**。

### 4.4 Head 的定义 & 参数数量

- 一个 Head = 一整套独立计算单元（W_Q/W_K/W_V + 计算过程 + 输出 Z）
- 参数：8 × 3 × 512 × 64 + 512 × 512 ≈ **105 万**

### 4.5 实现视角：多头注意力的真实计算方式（重点）

4.1-4.2 节是**概念视角**（每个头有独立的 W），本节是**工程实现视角**（实际代码怎么算的）。

#### 核心认知：3 个大矩阵，不是 num_heads 组小矩阵

```
❌ 误解：每个头有独立的 W_Q¹(512×64), W_K¹(512×64), W_V¹(512×64)...
✅ 实际：只有 3 个大矩阵 W_Q, W_K, W_V，每个都是 (d_model, d_model) = (512, 512)
         通过 reshape 切分成 8 个头
```

#### 真实计算过程（4 步）

```
Step 1: 3 次大矩阵乘法（不是 8×3=24 次小矩阵乘法）

  输入 X: (seq_len, 512)
  W_Q: (512, 512)    ← 一个完整的矩阵
  W_K: (512, 512)
  W_V: (512, 512)

  Q_full = X @ W_Q   → (seq_len, 512)   一次矩阵乘法
  K_full = X @ W_K   → (seq_len, 512)
  V_full = X @ W_V   → (seq_len, 512)

Step 2: Reshape 分头 — 这就是"多头"的来源

  Q_full: (seq_len, 512)
      → reshape → (seq_len, 8, 64)
      → transpose → (8, seq_len, 64)
                     ↑
                  8个头，每头拿到 64 维的 Q

  K_full, V_full 同理

  ⚡ 本质：W_Q 的前 64 列给了头1，接下来 64 列给了头2...
     每个头拿到的是不同的权重列 → 算出不同的 Q → 关注不同的方面

Step 3: 每个头独立做 Attention（和单头完全一样）

  头 1: Attention(Q₁, K₁, V₁) → (seq_len, 64)
  头 2: Attention(Q₂, K₂, V₂) → (seq_len, 64)
  ...
  头 8: Attention(Q₈, K₈, V₈) → (seq_len, 64)

  拼接: (seq_len, 64) × 8 → reshape → (seq_len, 512)

Step 4: W_O 投影 — 信息融合层

  拼接结果: (seq_len, 512)
      ↓ @ W_O (512, 512)
  输出: (seq_len, 512)
```

#### 实现 vs 数学等价

```python
# 实际实现（1次大矩阵乘法 + reshape）：
Q = X @ W_Q                    # (seq_len, 512)
Q = Q.reshape(seq_len, 8, 64)  # 分头

# 数学上完全等价于（8次小矩阵乘法）：
Q1 = X @ W_Q[:, :64]           # 头1用前64列
Q2 = X @ W_Q[:, 64:128]        # 头2用接下来64列
...
Q8 = X @ W_Q[:, 448:512]       # 头8用最后64列
```

工程上用大矩阵方式实现，因为**一次大矩阵乘法比多次小矩阵乘法快得多**（GPU 擅长大规模并行计算）。

#### W_Q/W_K/W_V 的维度（纠正易混淆点）

| 矩阵 | 维度 | 说明 |
|------|------|------|
| W_Q | (d_model, d_model) = (512, 512) | 不是 (d_model, d_k) |
| W_K | (d_model, d_model) = (512, 512) | 同上 |
| W_V | (d_model, d_model) = (512, 512) | 同上 |
| W_O | (d_model, d_model) = (512, 512) | 输出投影矩阵 |
| 每个头的 Q/K/V | (seq_len, d_k) = (seq_len, 64) | 切分后的中间结果 |

#### W_O 的真正作用（纠正误解）

```
❌ 误解：W_O 使得不同头关注不同的 aspect
✅ 正确：不同头关注不同 aspect 的能力来自 W_Q/W_K/W_V 被切分后，
         每个头拿到的权重列不同（训练时各自学到了不同的值）

W_O 的作用：把 8 个头的输出融合起来，投影回 d_model 维度
            是"信息混合层"，不是"注意力分配器"

                    W_Q (512 × 512)
                    ┌───┬───┬───┬───┐
                    │ 64│ 64│...│ 64│  ← 按列切成 8 份
                    └─┬─┴─┬─┴───┴─┬─┘
                      │   │       │
                     头1  头2    头8   ← 每头用不同的权重列
                      │   │       │       → 自然关注不同 aspect
                     Q₁  Q₂     Q₈

                     ↓    ↓       ↓
                  Attn  Attn   Attn    ← 每头独立做 attention
                     ↓    ↓       ↓
                    Z₁   Z₂     Z₈    ← 每头输出 (seq_len, 64)
                     ↓    ↓       ↓
              Concat → (seq_len, 512)
                     ↓
                  × W_O               ← 信息融合（不是分配注意力）
                     ↓
              最终输出 (seq_len, 512)
```

#### 单层 MHSA 参数量（以 d_model=512, num_heads=8 为例）

```
W_Q: 512 × 512 = 262,144
W_K: 512 × 512 = 262,144
W_V: 512 × 512 = 262,144
W_O: 512 × 512 = 262,144
bias: 4 × 512 = 2,048
─────────────────────────
MHSA 小计 ≈ 105 万

FFN（中间维度 = 4 × d_model = 2048）:
W₁: 512 × 2048 = 1,048,576
W₂: 2048 × 512 = 1,048,576
bias ≈ 2,560
─────────────────────────
FFN 小计 ≈ 210 万  ← 注意力的 2 倍！

单层总参数 ≈ 315 万（MHSA 占 1/3，FFN 占 2/3）
```

### 4.6 多层模型总参数量分布（以 GPT-2 Small 为例）

```
GPT-2 Small: d_model=768, num_heads=12, num_layers=12, vocab_size=50257

┌─────────────────────────────────────────────────────────┐
│ Embedding 层: 50257 × 768 ≈ 38.6M                      │
├─────────────────────────────────────────────────────────┤
│ 12 层 Decoder Block:                                     │
│   每层 MHSA: 4 × (768×768) + bias ≈ 2.36M              │
│   每层 FFN:  2 × (768×3072) + bias ≈ 4.72M             │
│   12 层小计: 12 × (2.36M + 4.72M) ≈ 84.9M             │
├─────────────────────────────────────────────────────────┤
│ LM Head (输出 Linear): 与 Embedding 共享权重，不额外计数  │
├─────────────────────────────────────────────────────────┤
│ 总计 ≈ 117M                                              │
└─────────────────────────────────────────────────────────┘

关键发现：
  FFN 参数量 ≈ MHSA 的 2 倍（因为中间维度是 4×d_model）
  Embedding 占比约 1/3
  num_layers 只数 Decoder Block，不含 Embedding 和 LM Head
```

---

## 第五章：Transformer 层的完整结构

### 5.1 Encoder 单层结构

```
输入 x ────────────────────────────┐
  ↓                                │ 残差连接
Multi-Head Self-Attention → z      │
  ↓                                │
z + x ←────────────────────────────┘
  ↓
LayerNorm(z + x) = h
  ↓
h ─────────────────────────────────┐
  ↓                                │ 残差连接
FFN → f                             │
  ↓                                │
f + h ←────────────────────────────┘
  ↓
LayerNorm(f + h) = 输出 → 传给下一层
```

### 5.2 FFN（Feed-Forward Network）

两层全连接层：`FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂`
- 512 → 2048（升维）+ ReLU → 2048 → 512（降维，无激活）
- 逐位置独立（每个词单独做，词间不影响）
- 分工：Self-Attention = 词间信息交流，FFN = 每个词自身信息加工

### 5.3 残差连接（Residual Connection）

`output = x + SubLayer(x)`
- 把子层的输入和输出相加
- 目的：防止信息丢失，即使子层学得不好，原始信息 x 仍保留
- 类比：做笔记时"在旁边补充"而非"替换"

### 5.4 层归一化（Layer Normalization）

```
x̂ = (x - μ) / √(σ² + ε)    ← 标准化到均值0方差1
output = γ · x̂ + β           ← γ、β 是可学习参数
```

- 目的：防止数值越层越大/越小（梯度爆炸/消失），稳定训练
- 残差 + LayerNorm 配合：既保留信息，又稳定训练 → 才能堆叠几十上百层

### 5.5 Encoder 多层数据流

```
第0层: (embedding + 位置编码) → Self-Attn → FFN → R₀
第1层: R₀ → Self-Attn → FFN → R₁
...
第5层: R₄ → Self-Attn → FFN → R₅（最终输出）

每层有独立的 W_Q/W_K/W_V/FFN 参数
底层学到局部关系，高层学到语义和全局关系
```

---

## 第六章：Decoder

### 6.1 Decoder 单层结构（比 Encoder 多一个 Cross-Attention）

```
Decoder 输入（embedding + 位置编码）
  ↓
① Masked Self-Attention（Q/K/V 全来自 Decoder 自己）
  ↓ + Add & Norm
② Cross-Attention
     Q ← Decoder 自己（"我在生成什么，需要什么信息"）
     K = Encoder输出 × W_K（Decoder 自己的 W_K）
     V = Encoder输出 × W_V（Decoder 自己的 W_V）
  ↓ + Add & Norm
③ FFN
  ↓ + Add & Norm
Decoder 输出 → 传给下一层 Decoder 或 Linear+Softmax
```

### 6.2 Decoder 的两路输入

| 输入 | 来源 | 用在哪个子层 |
|------|------|------------|
| 路1：已生成词的 embedding + 位置编码 | Decoder 自己 | Masked Self-Attention |
| 路2：Encoder 最终输出 R | Encoder（所有 Decoder 层共享同一份 R） | Cross-Attention（提供 K/V） |

- 路1 在 Decoder 层间逐层传递（D₁→D₂→D₃...）
- 路2 不变（每层都用同一份 R，但每层有自己的 W_K/W_V → 算出不同的 K/V）

### 6.3 Cross-Attention 详解

```
Q = Decoder上一层输出 × W_Q_dec    ← Decoder 的参数
K = Encoder输出 R       × W_K_dec    ← Decoder 的参数（不是 Encoder 的！）
V = Encoder输出 R       × W_V_dec    ← Decoder 的参数

W_K_dec/W_V_dec 通过训练学会：如何从 Encoder 输出中提取"标签"和"内容"
R 是"原料"，用什么 W 去加工，由 Decoder 自己学
```

没有神秘的"转换"——就是 `R × W_K = K`，和 Self-Attention 算 K/V 的方式完全一样，只是输入从 x 换成了 R，W 换成了 Decoder 的。

### 6.4 Masked Self-Attention（掩码机制）

```
Q/K/V 计算：完全并行，和 Encoder 一样（无顺序依赖）

Mask 加在注意力分数矩阵上（softmax 之前）：

           "i"    "love"   "study"
"i"       [ 2.1    -∞      -∞  ]    ← 只看自己
"love"    [ 1.5    3.2     -∞  ]    ← 看前面+自己
"study"   [ 0.3    1.8     2.9 ]    ← 看所有

Mask 把"看到未来"的位置设为 -∞
softmax 后：-∞ → 0，该词不会被关注

全程仍是并行矩阵运算，Mask 不改变计算方式，只限制"谁能看到谁"
```

### 6.5 GPT 为什么不需要 Cross-Attention？

GPT 只有 Masked Self-Attention + FFN：
- 翻译：输入（法语）≠ 输出（英语）→ 需要 Cross-Attention 连接两个序列
- GPT：前面的词 = 已生成的输出 → 一个序列就够

### 6.6 Linear + Softmax 输出层（LM Head 详解）

#### LM Head 的本质

LM Head（Language Model Head）= 一个 Linear 层 = `nn.Linear(d_model, vocab_size, bias=False)`，没有激活函数，就是一次矩阵乘法。

```
第 12 层 Decoder 输出: (seq_len, d_model) = (10, 768)
        │
        ↓ LM Head = Linear 层: W_lm (768, 50257)  ← vocab_size=50257
        │  就是一次矩阵乘法: (10, 768) @ (768, 50257) = (10, 50257)
        │
logits: (10, 50257)  ← 每个位置对 50257 个词的原始分数（可正可负，无范围）
        │
        ↓ Softmax（不改变维度！只是数值归一化）
        │
probs:  (10, 50257)  ← 每个位置的概率分布（0~1，和=1）
        │
        ↓ argmax（取概率最大的词）
        │
输出:   每个位置 1 个 token
```

**Logits = 未经 Softmax 的原始分数**，是 Linear 层的直接输出。分数越高 → Softmax 后概率越大 → 越可能被选中。

#### 输入和输出访问词表的方式不同

```
输入（Embedding 查表）：
  Token IDs [464, 1820, 3877, ...]  ← 整数索引
  每个 ID 作为 Index，从 E(vocab_size×d_model) 中"捷"对应行
  不是矩阵乘法，是查表操作
  结果: (seq_len, d_model) = (10, 768)

输出（LM Head 矩阵乘法）：
  (10, 768) @ (768, 50257) = (10, 50257)
  是真正的矩阵乘法
  结果: 每个位置对词表中每个词的得分
```

#### Weight Tying（权重共享）

```
GPT-2 的 LM Head 权重和 Embedding 矩阵是同一个矩阵：

  Embedding 查表: 输入 ID → 取 E 的第 ID 行 → (d_model,)
  LM Head 打分:   (10, d_model) @ E^T → (10, vocab_size)

  E: (vocab_size, d_model) = (50257, 768)  ← 同一个矩阵，两个用途

→ 省掉了一整份参数（约 38.6M）！
→ 这也是为什么 4.6 节里 LM Head "不额外计数"
```

#### 推理时为什么只取最后一行 logits？

```
10 个位置各自有一个 (1, 50257) 的概率分布，各自预测"我的下一个词是谁"：

  位置 1 ("The"):   预测下一个词 → "quick"（概率最高）  ← 我们不关心
  位置 2 ("quick"):  预测下一个词 → "brown"（概率最高） ← 我们不关心
  ...
  位置 10 ("then"):  预测下一个词 → "jumps"（概率最高） ← 只取这个！

为什么前 9 个位置不用？
  输入是我们自己写的 prompt，不需要模型告诉我们 "The" 后面是 "quick"
  我们唯一想知道的是：看完整个 prompt 之后，下一个词该是什么

代码：next_token_logits = logits[:, -1, :]  ← 只取最后一行

⚠ 但这 10 个位置的预测在训练时全都用！
  每个位置都和正确答案算 Loss（Teacher Forcing）
```

#### 推理输出不是 10 个词，而是 1 个词

```
❌ 误解：Prefill 输入 10 个 token → 输出 10 个词
✅ 正确：Prefill 输入 10 个 token → 只输出 1 个词（下一个词）
          之后由 Decode 阶段逐词追加，拼成完整回复

模型每次只能预测"下一个词"，没有能力一次输出多个词
ChatGPT "流式输出"一个字一个字蹦出来，就是因为底层在一步步 Decode
```

#### 完整输出维度链路（推理视角）

```
Token IDs: (1, 10)           ← 输入是整数索引，不是向量
    ↓ Embedding 查表
(10, 768)
    ↓ 12 层 Decoder Block（MHSA + FFN + 残差 + LayerNorm）
    │ 每层 W_O: (768, 768)，保证输入输出维度一致（残差连接需要）
(10, 768)
    ↓ LM Head = Linear (768, 50257)
(10, 50257) = logits
    ↓ Softmax（维度不变）
(10, 50257) = probs
    ↓ 取 logits[:, -1, :] 再 argmax
1 个 token（Prefill 的唯一输出）
    ↓ Decode 阶段逐词追加
完整回复

停止条件（满足任一即停）：
  1. 模型生成了 <EOS> token（模型自己学会的"结束信号"）
  2. 达到 max_length / max_new_tokens 上限（防止无限生成）
```

### 6.7 Decoder 完整工作流程

```
训练时（Teacher Forcing）:
  输入: <BOS>, "i", "am", "a"       ← 正确答案右移一位
  目标: "i", "am", "a", "student"    ← 正确答案
  每个位置都算 Loss → 一次性反向传播

推理时（自回归生成）:
  Encoder 先跑完: 法语 → Encoder×6 → R（固定不变）
  生成第1词: <BOS> → Decoder → Linear → Softmax → "i"
  生成第2词: <BOS>,"i" → Decoder → Linear → Softmax → "am"
  生成第3词: ... → "a"
  生成第4词: ... → "student"
  生成第5词: ... → <EOS> → 停止
```

### 6.8 Decoder 训练 vs 推理的差异（重点）

#### 重要认知纠正

```
❌ 误解：Teacher Forcing 的差异在 attention score 计算过程中
✅ 正确：Attention score 计算过程训推完全一样
           差异在"喂什么进 Decoder"和"跑几次 forward"
```

只要喂进去的序列一样，attention 公式、W_Q/W_K/W_V、Mask 全部一模一样，算出来的结果字节级一致。

#### 五个维度的差异对比

| 维度 | 训练时 | 推理时 |
|------|-------|-------|
| Attention 计算 | softmax(QK^T/√d_k+Mask)·V | 一模一样 |
| **输入来源** | 正确答案右移（Teacher Forcing） | 模型自己上一步生成的词 |
| **forward 次数** | 1 次（所有位置并行） | N 次（串行逐词生成） |
| **用哪些位置的输出** | 全部位置（每个位置都算 Loss） | 只用最后一个位置 |
| **是否更新参数** | 是（反向传播） | 否（参数冻结） |

#### Teacher Forcing 举例

```
训练任务：法语 "je suis étudiant" → 英语 "i am a student"

Decoder 输入:  <BOS>, "i", "am", "a"          ← 4 个位置
Decoder 目标:  "i", "am", "a", "student"      ← 4 个位置

一次 forward 并行计算 4 个位置（Mask 保证不偷看未来）:
  位置1看到 <BOS>           → 预测 → P("i")=0.7        → Loss₁
  位置2看到 <BOS>,"i"       → 预测 → P("am")=0.9       → Loss₂
  位置3看到 <BOS>,"i","am"  → 预测 → P("a")=0.5        → Loss₃
  位置4看到 <BOS>,"i","am","a" → 预测 → P("student")=0.6 → Loss₄

Loss 平均 → 反向传播 → 更新参数

⚠ 即使位置2 预测错了（比如猜成 "is"），位置3 输入仍是正确的 "am"
   → 老师强制纠错，防止错误传播拖垮训练
```

#### 推理时的自回归生成

```
Encoder 先跑完一次："je suis étudiant" → R（整个推理过程不变）

第1次 forward:
  Decoder 输入: <BOS>                    ← 1 个位置
  取最后位置输出 → argmax → "i"

第2次 forward:
  Decoder 输入: <BOS>, "i"               ← 2 个位置（追加刚生成的词）
  取最后位置输出 → argmax → "am"

第3次 forward:
  Decoder 输入: <BOS>, "i", "am"         ← 3 个位置
  取最后位置输出 → argmax → "a"

...直到生成 <EOS> 或达到 max_length

⚠ 模型自己生成的词作为下一次输入，错了也没人纠正 → exposure bias
```

#### Mask 的真正作用

```
训练时把 N 个位置一起喂进去并行算，但要保证：
  位置1 计算时只能看到位置1
  位置2 计算时只能看到位置1, 2
  ...
  位置N 计算时只能看到位置1 ~ N

→ 用 Mask 把"未来位置"的 attention 分数设为 -∞
→ softmax 后这些位置的权重 = 0
→ 一次矩阵运算 = N 次串行 forward 的等价效果

推理时也用同一个 Mask（保持训练和推理的计算逻辑一致）
```

#### 为什么训练能并行而推理不能？

```
训练时：
  已有完整答案 [<BOS>, i, am, a]
  全塞进去一次性算 4 个位置 → 并行

推理时：
  没有答案，必须等位置 i 生成完才知道位置 i+1 的输入
  位置2 的输入 = 位置1 生成的词，必须串行
```

#### KV Cache 的等价性

```
推理第 N 次 forward 时，前 N-1 个 token 的 K/V 和上一次完全一样
（因为 W_K/W_V 不变，前 N-1 个 token 的输入也不变）

→ 缓存复用前 N-1 个 K/V 是数学上完全等价的
→ 时间复杂度从 O(N²) 降到 O(N)
→ 这就是 KV Cache 优化的理论依据
```

#### 现代 LLM 推理流程（Decoder-Only，串起所有概念）

```
用户输入: "今天天气真"
  ↓ tokenize（查词表）
[5234, 8721, 9012, 4567]
  ↓ 第1次 forward（建立 KV Cache）
预测下一个词: "好"
  ↓ 追加到输入
[5234, 8721, 9012, 4567, 1234]
  ↓ 第2次 forward（用 KV Cache，只算新 token）
预测下一个词: "，"
  ↓ ...
直到生成 <EOS> 或达到 max_length
```

### 6.9 常见误区：Decoder 是否“天生”为预测下一个词而生？

❌ 误区：Decoder 就是预测下一个词的专用结构，不论什么任务都是这样。

✅ 正确：Decoder 架构只是“因果掩码下的序列上下文编码器”，输出的是 hidden states（向量序列）。
“预测下一个词”是 Decoder + LM Head + Softmax + 交叉熵 Loss 这一整套组合的结果，不是 Decoder 单独决定的。

#### 三层拆分框架

| 层次 | 内容 | 是否固定 |
|------|------|----------|
| 架构 | Masked Self-Attention + FFN | 结构固定 |
| 输出头（Head） | 把 hidden states 映射到任务目标 | **可变** |
| 训练目标（Loss） | 监督信号 | **可变** |

#### 同样 Decoder 架构，换 Head 和 Loss 可以做完全不同的事

```
语言建模（GPT）：
  Head: LM Head（投影到词表）
  Loss: 交叉熵（下一个词）
  → 文本生成

情感分类（GPT 微调）：
  Head: 分类头（取最后一个 token 的 hidden state → 类别数）
  Loss: 交叉熵（类别标签）
  → 文本分类

奖励模型（Reward Model）：
  Head: 标量输出头（hidden state → 一个分数）
  Loss: 偏好排序 Loss（Bradley-Terry）
  → 给回复打分
```

#### Causal Mask 的真正作用

```
Mask 保证位置 i 只能看到 ≤i 的信息
  → 这是注意力约束，不是任务定义
  → 保证自回归生成的合理性（推理时不偷看未来）
  → 但不强制规定输出必须是词表概率
```

#### 为什么大家“感觉”Decoder = 预测下一个词？

1. **预训练范式的统治地位**：海量无标注文本天然提供“下一个词是什么”的监督信号，不需人工标注，可无限扩展数据
2. **Causal Mask 和自回归生成天然契合**：Mask 让位置 i 只看过去，用“预测下一个”的方式能并行训练整个序列（Teacher Forcing）
3. **GPT 系列的成功**让“Decoder = 生成式语言模型”成为主流认知

#### 比喻

> Decoder = 理解了前文的“大脑”，输出 hidden states（理解结果）
> LM Head = “嘴巴”，把理解结果变成词表概率
> 预测下一个词 = 大脑 + 嘴巴 + 交叉熵 Loss 这一整套组合的表达
>
> 同一个大脑，换不同的“输出器官”，可以做完全不同的事

---

## 第七章：Attention 类型总结

| 类型 | Q 来源 | K/V 来源 | 用在哪 |
|------|--------|----------|--------|
| Self-Attention | 输入自身 | 输入自身 | Encoder + Decoder |
| Masked Self-Attention | 输入自身 | 输入自身（+遮罩） | Decoder |
| Cross-Attention | Decoder 自身 | Encoder 输出 | Decoder 中间层 |

核心：**Q 和 K/V 是不是来自同一个序列**。Self = "自己看自己"，Cross = "我看你"

---

## 第八章：训练过程

### 8.1 完整训练流程

```
1. 拿一批翻译对（法语→英语）
2. 法语 → Encoder×6 → R（前向传播）
3. 英语右移一位 → Decoder×6 → Z → Linear → Softmax → 概率（前向传播）
4. 算 Loss（交叉熵）
5. Loss.backward()（PyTorch 自动反向传播，算出所有梯度）
6. optimizer.step()（用梯度更新所有参数）
7. 重复，直到 Loss 足够小
```

### 8.2 Loss 计算（交叉熵）

```
Loss = 每个位置上 -log(模型给正确答案的概率) 的平均值

例：翻译 "i am student"

  位置1: 正确答案"i"，模型给 P=0.7  → -log(0.7) = 0.36
  位置2: 正确答案"am"，模型给 P=0.9  → -log(0.9) = 0.11
  位置3: 正确答案"student"，模型给 P=0.5 → -log(0.5) = 0.69

  Loss = (0.36 + 0.11 + 0.69) / 3 = 0.39

  概率越高 → -log越小 → Loss越小
  完美预测(概率=1) → Loss=0
```

### 8.3 所有训练参数

| 参数 | 说明 |
|------|------|
| Embedding 矩阵 E | 随机初始化，端到端训练 |
| 每层 W_Q, W_K, W_V, W_O | Multi-Head Attention 参数 |
| 每层 FFN W₁, W₂, b₁, b₂ | 两层全连接 |
| 每层 LayerNorm γ, β | 归一化缩放和平移 |
| 最终 LM Head (Linear 层) | d_model → vocab_size，与 Embedding 共享权重 (GPT-2) |

### 8.4 反向传播与残差连接

```
残差连接的梯度流：
  ∂Loss/∂x = ∂Loss/∂output × (1 + ∂SubLayer/∂x)
                  ↑                    ↑
              "高速公路"             正常路径
              即使子层梯度很小，梯度仍能通过"1"直接传回去

没有残差：梯度逐层相乘 → 6层连乘6次 → 爆炸或消失
有残差+LayerNorm：梯度有高速公路 → 能堆96层甚至更多
```

### 8.5 关键理解

- 前向传播简单（矩阵乘法），反向传播复杂（链式法则），但 PyTorch 自动求导
- 不同随机初始化 → 不同局部最优。Multi-Head 故意利用差异性
- Transformer 从零端到端训练，不依赖预训练组件
- 推理时不算 Loss，直接 argmax 或采样选词

---

## 第九章：学习进度

- [x] Transformer 架构 & 三种变体
- [x] Embedding + Positional Encoding
- [x] Self-Attention（逐词 + 矩阵 + Q/K/V 本质）
- [x] Multi-Head Attention（概念 + 实现机制 + W_O 作用 + 参数量）
- [x] FFN + Residual + LayerNorm
- [x] Encoder 完整数据流
- [x] Decoder（Masked Self-Attn + Cross-Attn + Mask + LM Head 详解 + Linear+Softmax）
- [x] Decoder 本质辨析（架构/Head/Loss 三层拆分，Decoder ≠ 只能预测下一个词）
- [x] 训练过程（Loss 计算 + 反向传播 + 残差梯度）
- [x] Attention 类型总结
- [ ] 读 "Attention is All You Need" 论文
- [ ] PyTorch 代码实现 Self-Attention

---

## 附录：疑问记录（Q&A）

### Q/K/V 相关

**Q: W_Q/W_K/W_V 到底有什么区别？**
A: 初始化和计算一样，区别在公式角色（Q查K取V）→ 梯度不同 → 训练后不同。

**Q: 训练时是在更新 QKV 吗？**
A: 不是，更新的是 W_Q/W_K/W_V。q/k/v 是临时中间结果。

**Q: 每层 W_Q/W_K/W_V 独立吗？**
A: 是，每层独立随机初始化，训练后各自学不同东西。

**Q: 矩阵和向量怎么区分？**
A: W_Q/W_K/W_V = 层级别参数（固定）。q/k/v = 词级别中间结果（N个词=N个向量）。

### 计算相关

**Q: 点积怎么算？** A: 对应位置相乘再求和，结果是标量。[1,4,8,0]·[2,3,3,9]=38。

**Q: 除以 √d_k？** A: 防止点积太大导致 softmax 梯度消失。

**Q: 输出是词还是句子？** A: 每个词一个向量，结果 N×d_k 矩阵。一次矩阵运算全搞定。

**Q: Q·K^T 维度？** A: 始终 N×N（N=词数），长序列吃显存的原因。

### Multi-Head 相关

**Q: 8个Head公式一样，为什么学不同？** A: 随机初始值不同→梯度不同→越来越不同。故意利用差异性。

**Q: Head 指什么？** A: 一整套计算单元（W_Q/W_K/W_V + 计算 + 输出 Z）。

**Q: W_O？** A: 输出权重矩阵，训练参数，融合8个Head。

### Embedding & 位置编码

**Q: Embedding 预训练的吗？** A: 不是，端到端训练，随机初始化。

**Q: 位置编码叠加时交叉吗？** A: 不交叉，Embedding阶段独立。

### Decoder 相关

**Q: Encoder输出Z怎么变成K/V？** A: 就是 Z×W_K_dec 和 Z×W_V_dec，W是Decoder自己的参数。

**Q: 多层Decoder都用同一份Encoder输出？** A: 是的，R共享，但每层W_K/W_V不同→算出不同K/V。

**Q: GPT没有Encoder怎么做Cross-Attention？** A: 不做，GPT只有Masked Self-Attention+FFN。

**Q: Linear层是降维吗？** A: 不是，是升维（d_model→vocab_size）。传统分类的Linear是降维（特征→类别数），本质都是把"理解"映射到"选项得分"。

**Q: 输出词数和输入词数必须一样吗？** A: 不需要。Decoder逐词生成，输出<EOS>时停止，长度由模型自己决定。

### LM Head & Logits 相关

**Q: LM Head 是什么？** A: 就是模型最后一个 Linear 层 (d_model → vocab_size)，没有激活函数，只做一次矩阵乘法。

**Q: Logits 是什么？** A: LM Head 的原始输出，未经 Softmax 的分数 (seq_len, vocab_size)。分数越高 → Softmax 后概率越大。

**Q: Softmax 会改变维度吗？** A: 不会。输入 (10, 50257) → Softmax → 输出仍是 (10, 50257)，只是数值归一化到 0~1。

**Q: 推理时为什么只取最后一行 logits？** A: 只有最后一行"看完"整个 prompt，前几行只看到部分前文，对我们没用。

**Q: Prefill 输入 10 个 token，能输出 10 个词吗？** A: 不能，只输出 1 个词。10 个位置各自的预测在推理时只有最后一个有用，其余只在训练时参与 Loss 计算。

**Q: GPT-2 的 LM Head 和 Embedding 是同一个矩阵？** A: 是。这叫 Weight Tying，省一份参数（约 38.6M）。

### Decoder 本质 & 任务相关性

**Q: Decoder 架构是否“天生”为预测下一个词而生？** A: 不是。Decoder 只是“因果掩码下的序列上下文编码器”，输出的是 hidden states（向量序列）。预测下一个词是 Decoder + LM Head + Softmax + 交叉熵 Loss 这一整套组合的结果，不是 Decoder 架构单独决定的。

**Q: 同样是 Decoder，为什么能做不同任务？** A: 因为架构、输出头、Loss 是三个独立层次。架构固定（Masked Self-Attention + FFN），换 Head 和 Loss 就能做分类、回归、打分等任意任务。

**Q: Causal Mask 的作用是什么？** A: 保证位置 i 只能看到 ≤i 的信息，是注意力约束（保证自回归生成时不偷看未来），不是任务定义（不强制输出必须是词表概率）。

**Q: 为什么大家都认为 Decoder = 预测下一个词？** A: 因为预训练范式的统治地位——海量无标注文本天然提供“下一个词”的监督信号，不需人工标注，可无限扩展，加上 GPT 系列的成功，让这成为主流认知。但这只是 Decoder 最常见、最自然的用法，不是唯一用法。

### 训练相关

**Q: Loss怎么算？** A: 交叉熵。每个位置取正确答案的概率 → 取-log → 求平均。

**Q: 训练时和推理时Decoder处理有什么不同？** A: 训练时用Teacher Forcing，所有位置同时算Loss；推理时只用最后一个位置，逐词生成。
