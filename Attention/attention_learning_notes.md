# Attention & Transformer 学习笔记

> 基于 Jay Alammar "Illustrated Transformer" 精读 + Q&A 整理
> 学习日期：2026-06-04 起

---

## 1. Transformer 整体架构

### 1.1 原版 Transformer（2017）
```
输入（法语）→ Encoder(×6) → Decoder(×6) → 输出（英语）
```
- Encoder 和 Decoder 各堆叠 6 层（数字可调）
- 每层 Encoder = Self-Attention + FFN
- 每层 Decoder = Masked Self-Attention + Cross-Attention + FFN

### 1.2 现在的三种变体
| 架构 | 代表模型 | 用途 |
|------|---------|------|
| Encoder-Decoder | T5, BART | 翻译、摘要 |
| **Decoder-Only** | **GPT, Claude, Qwen, LLaMA** | **通用大模型（主流）** |
| Encoder-Only | BERT | 文本理解（不生成） |

> 现在说的"Transformer"是广义的——以 Self-Attention 为核心的所有变体。GPT 类模型算 Transformer，只是去掉了 Encoder 和 Cross-Attention。

---

## 2. Self-Attention（自注意力）

### 2.1 为什么需要 Self-Attention？
- RNN 的问题：信息一步步传递（h₁→h₂→h₃...），距离越远衰减越严重
- Self-Attention：每个词直接看所有其他词，无衰减，可并行

### 2.2 核心思想
让每个词在编码时能"看到"句子里的所有其他词，把相关信息融入自己的表示中。

例子："The animal didn't cross the street because **it** was too tired"
- 处理 "it" 时，Self-Attention 发现 "it" 和 "animal" 关系最大
- 把 "animal" 的信息融入到 "it" 的表示中

### 2.3 计算步骤（6步）

```
Step 1: 生成 Q、K、V
  q = W_Q · x    （Query - "我在找什么？"）
  k = W_K · x    （Key   - "我是什么？"）
  v = W_V · x    （Value - "我的实际内容"）

Step 2: 计算注意力分数
  score_i = q · k_i    （用 Query 和每个 Key 做点积）
  点积越大 = 越相关 = 注意力越高

Step 3: 缩放
  score = score / √d_k    （d_k = 64，防止值太大导致 softmax 梯度消失）

Step 4: Softmax 归一化
  attention_weights = softmax(scores)    （所有权重为正，加起来=1）

Step 5: 用权重乘 Value
  weighted_v_i = attention_weight_i × v_i

Step 6: 求和
  output = Σ(weighted_v_i)
```

### 2.4 核心公式（一句话版本）
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

### 2.5 Q/K/V 的本质

**矩阵 vs 向量（两个视角）：**

```
矩阵视角（Attention 层的参数，层级别）：
  W_Q, W_K, W_V  ← 每层一套，训练前随机初始化，训练后固定

向量视角（由输入决定，词级别）：
  句子有 N 个词 → 就有 N 个 q、N 个 k、N 个 v 向量
  q_i = W_Q · x_i    （第 i 个词的 Query 向量）
  k_i = W_K · x_i    （第 i 个词的 Key 向量）
  v_i = W_V · x_i    （第 i 个词的 Value 向量）

例："Thinking Machines"（2个词）→ q₁,q₂ / k₁,k₂ / v₁,v₂
```

**已理解的关键点：**

- W_Q, W_K, W_V 是**训练参数**（类比传统 NN 的 W 和 b）
- q, k, v 是**中间计算结果**（每次前向传播从输入重新算出来）
- 初始化和计算方式上，三个矩阵完全一样（都是随机初始化，都是矩阵乘法）
- **区别在于公式中的角色不同**：
  - Q 用来"查"（提问）
  - K 用来"被查"（被匹配）
  - V 是"被取出的内容"
- 因为角色不同 → 反向传播时梯度不同 → 训练后学到不同的值

**类比**：就像传统 NN 每层都有独立的 W 和 b，初始化都是随机，但训练后不同层的参数学到不同的东西。

### 2.6 训练过程（和传统 NN 完全一样）
```
前向传播 → 算 loss → 反向传播 → 梯度下降更新 W_Q, W_K, W_V（+ FFN参数）
```
- 训练前：所有层的 W_Q, W_K, W_V 随机初始化
- 每层的参数独立，不复用
- 训练后：底层 Encoder 学到局部关系，高层学到语义和全局关系

### 2.7 维度说明
```
输入 embedding: 512 维
W_Q, W_K, W_V: 512 × 64 的矩阵
q, k, v: 64 维（压缩了，为了省计算量）

Multi-Head 用 8 个 64 维 = 512 维，信息量一样
```

---

## 3. Attention 的三种类型

| 类型 | Q 来源 | K/V 来源 | 用在哪 |
|------|--------|----------|--------|
| Self-Attention | 输入自身 | 输入自身 | Encoder + Decoder |
| Cross-Attention | Decoder | Encoder | Decoder 中间那层 |
| 传统 Attention | Decoder 隐藏状态 | Encoder 隐藏状态 | 早期 Seq2Seq |

核心区别：**Q 和 K/V 是不是来自同一个序列**
- Self-Attention = "自己看自己"
- Cross/传统 Attention = "我看你"

---

## 4. Encoder 的数据流

```
第1层: embedding → 算Q₁K₁V₁ → Self-Attention → FFN → 中间表示₁
第2层: 中间表示₁ → 算Q₂K₂V₂ → Self-Attention → FFN → 中间表示₂
...
第6层: 中间表示₅ → 算Q₆K₆V₆ → Self-Attention → FFN → 最终表示

每层的 W_Q/W_K/W_V 是该层独有的、训练好的、固定的参数
输入不同 → 算出的 Q/K/V 不同 → 输出不同
```

---

## 5. FFN（Feed-Forward Network）

就是两层全连接层：
```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

第一层：512维 → 2048维 + ReLU（升维 + 非线性变换）
第二层：2048维 → 512维（降维回来，无激活）
```

- 逐位置独立应用（每个词单独做，词间不互相影响）
- 分工：Self-Attention 负责词间信息交流，FFN 负责每个词自身的信息加工

---

## 6. Transformer vs RNN

| | RNN | Transformer |
|---|---|---|
| 记忆方式 | 隐藏状态 h 不断累积 | 每个词直接访问所有词 |
| 远距离依赖 | 差（信息衰减） | 好（一步直达） |
| 计算方式 | 串行（必须按顺序） | 并行（矩阵运算） |
| GPU 利用率 | 低（GPU 空转等待） | 高（全是矩阵乘法） |

Self-Attention 层的 dependency：每个词计算时需要所有词的 K 和 V 参与
FFN 层无 dependency：每个词独立计算
→ 两者都可以并行（矩阵运算一次性算完所有位置）

---

## 7. 待学习（明天继续）

- [x] Self-Attention 的 6 步计算（逐词视角 + 矩阵视角）
- [ ] Matrix Calculation of Self-Attention（矩阵形式的完整计算）
- [ ] Multi-Head Attention（多头注意力）
- [ ] Positional Encoding（位置编码）
- [ ] Residual Connection + Layer Norm（残差连接 + 层归一化）
- [ ] Decoder 部分（Masked Self-Attention + Cross-Attention）
- [ ] 最终的 Linear + Softmax 输出层
- [ ] 训练过程回顾

---

## 8. 个人疑问记录

> 学习过程中遇到的疑问和解答记录在这里

### Q: W_Q W_K W_V 到底有什么区别？看起来一模一样
A: 初始化和计算方式确实一样。区别在公式中的角色（Q查K取V），导致训练时梯度不同，最终学到不同的值。

### Q: 训练时是在更新 QKV 吗？
A: 不是。更新的是 W_Q/W_K/W_V（权重矩阵），q/k/v 是每次前向传播临时算出的中间结果。

## Q: 每层 Encoder 的 W_Q/W_K/W_V 是独立的吗？
A: 是的，每层有独立的参数，训练前全部随机初始化，训练后各自学到不同的东西。

### Q: QKV 的矩阵和向量怎么区分？
A: 矩阵（W_Q/W_K/W_V）是层级别的训练参数，每层一套，固定不变。向量（q/k/v）是词级别的中间结果，有多少个输入词就有多少个 q/k/v 向量，每次前向传播重新计算。

### Q: 6步计算得到的是某个词的向量还是整句话的？
A: 每个词都当一次核心词，每个词都得到一个输出向量。最终结果是矩阵（N×64），每行是一个词的“加强版”表示（融合了其他词的信息）。实际实现中不需要走 N 次，一次矩阵运算全部搞定：softmax(Q·K^T/√d)·V。

### Q: 点积怎么算？
A: q 和 k 维度相同（都是64维），对应位置相乘再求和。结果是标量，值越大表示越相关。例：[1,4,8,0]·[2,3,3,9] = 2+12+24+0 = 38。

### Q: 除以 √d_k 的目的是什么？
A: 防止点积值太大。64维向量点积可能到 64~128，值太大会让 softmax 输出过于极端（如 0.999），导致梯度接近 0，训练学不动。除以 8 让数值在合理范围内。
