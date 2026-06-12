# Attention & Transformer 学习笔记

> 基于 Jay Alammar "Illustrated Transformer" 精读 + Q&A 整理
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

## 第二章：输入处理（Embedding + 位置编码）

### 2.1 语义 Embedding

- Embedding 矩阵 E（vocab_size × 512），是**训练参数**（随机初始化，端到端训练更新）
- 每个词查表得到 512 维语义向量
- 训练后语义相近的词向量会靠近（如 cat 和 dog）
- **不是**预训练的 Word2Vec，是 Transformer 自己从头学的

### 2.2 位置编码（Positional Encoding）

Self-Attention 本身"无序"——"cat sat" 和 "sat cat" 输出一样。需要位置编码给模型"位置感"。

**位置向量公式**（固定值，非训练参数）：
```
t_i 的第 2k 维   = sin(i / 10000^(2k/d))
t_i 的第 2k+1 维 = cos(i / 10000^(2k/d))
i = 词位置，k = 维度索引，d = 512
```

### 2.3 输入 = 语义 + 位置（直接相加）

```
第 0 层输入：
  "Je"       → E["Je"] + t₀ → x₁
  "suis"     → E["suis"] + t₁ → x₂
  "étudiant" → E["étudiant"] + t₂ → x₃

然后用 (x+t) 去算 Q/K/V：  q = W_Q · (x + t)

只在第 0 层加位置编码，后续层不需要（位置信息已融入向量）
Embedding 阶段词与词独立，交叉发生在 Self-Attention 阶段
```

| | 语义 Embedding | 位置 Encoding |
|---|---|---|
| 来源 | 查矩阵 E（训练参数） | sin/cos 公式（固定值） |
| 是否训练 | 是 | 否 |
| 维度 | 512 | 512 |

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
W_Q, W_K, W_V: 512 × 64（降维，省计算量）
q, k, v: 64 维
Multi-Head 8 × 64 = 512 维，信息量还原
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

### 6.6 Linear + Softmax 输出层

```
Decoder 最终输出 Z (N×512)
  ↓ 训练时：每行都算，推理时：只取最后一行
z (512维)
  ↓ Linear层 (512 → vocab_size，如50000)  ← 升维！不是降维
logits (50000维)  ← 每个位置对应词表里一个词的得分
  ↓ Softmax
概率 (50000维)   ← 所有值 0~1，加起来=1
  ↓ argmax（取最大值）
"student" → 查词表得到输出
```

- Linear 层是**升维**（512→50000），让每个位置对应词表里每个词的得分
- 对比传统分类：Linear 是降维（特征→2~3个类别）。本质一样：把"理解"映射到"选项得分"
- 输出长度不固定：模型自己决定什么时候输出 `<EOS>`（结束符），训练时学会了这个能力

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
| 最终 Linear 层 | 512 → vocab_size |

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
- [x] Multi-Head Attention
- [x] FFN + Residual + LayerNorm
- [x] Encoder 完整数据流
- [x] Decoder（Masked Self-Attn + Cross-Attn + Mask + Linear+Softmax）
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

**Q: Linear层是降维吗？** A: 不是，是升维（512→50000词表）。传统分类的Linear是降维（特征→类别数），本质都是把"理解"映射到"选项得分"。

**Q: 输出词数和输入词数必须一样吗？** A: 不需要。Decoder逐词生成，输出<EOS>时停止，长度由模型自己决定。

### 训练相关

**Q: Loss怎么算？** A: 交叉熵。每个位置取正确答案的概率 → 取-log → 求平均。

**Q: 训练时和推理时Decoder处理有什么不同？** A: 训练时用Teacher Forcing，所有位置同时算Loss；推理时只用最后一个位置，逐词生成。
