# 📘 PyTorch 学习笔记 

## 1. 参考文献
- https://github.com/chenyuntc/pytorch-book
- https://zh-v2.d2l.ai/chapter_introduction/index.html

---

## 2. 张量（Tensor）

### 2.1 基本概念

张量是 PyTorch 中定义的**多维数组**，是标量、向量、矩阵在任意维度上的推广：

| 维度 | 名称 | 举例 |
|------|------|------|
| 0维 | 标量（Scalar） | `5` |
| 1维 | 向量（Vector） | `[1, 2, 3]` |
| 2维 | 矩阵（Matrix） | `[[1,2],[3,4]]` |
| 3维及以上 | 张量（Tensor） | 批量文本嵌入、图像等 |

**PyTorch 张量的两大核心优势**：
1. ✅ **GPU 加速并行计算**
2. ✅ **自动求导（Autograd）**，深度学习训练的核心

张量通过 `torch.Tensor` 类定义。

### 2.2 张量的核心属性（"身份证"信息）

| 属性 | 含义 | 示例 |
|------|------|------|
| `.dim()` | 维数（几维） | 3 |
| `.shape` / `.size()` | 形状（每维大小） | `[2, 3, 4]` |
| `.numel()` | 元素总数 | 24 |
| `.dtype` | 数据类型 | `torch.float32` |
| `.device` | 所在设备 | `cpu` / `cuda:0` |
| `.requires_grad` | 是否需要求导 | `True` / `False` |

### 2.3 张量的创建

```python
torch.zeros(shape)      # 全0张量
torch.ones(shape)       # 全1张量
torch.rand(shape)       # [0,1) 均匀分布
torch.randn(shape)      # 标准正态分布
torch.arange(start, end, step)   # 等差数列
torch.tensor([...])     # 从列表创建
```

### 2.4 GPU 计算

```python
x.to(device)   # 推荐，灵活指定设备
x.cuda()       # 转到 GPU
x.cpu()        # 转到 CPU
```

### 2.5 张量的基本操作

- **索引 / 切片**：类似 NumPy
- **形状变换**：`reshape` / `view` 改变形状，不改变元素数量和元素值
- **维度操作**：
  - `cat()` 拼接
  - `squeeze()` 去除大小为1的维度
  - `unsqueeze()` 增加一个维度
  - `transpose()` / `permute()` 调换维度顺序

### 2.6 张量的运算（⭐ 核心）

深度学习中的张量运算分为**两大类**，场景和目的完全不同：

#### 🔸 类型一：矩阵/向量变换（维度变换型）
- **代表**：矩阵乘法 `@` / `torch.matmul`
- **核心目的**：让不同维度的信息**交流和融合**，产生新的特征空间
- **应用**：特征提取、维度扩展/压缩、空间投影

```python
y = x @ W.T     # [B, L, D_in] @ [D_in, D_out] → [B, L, D_out]
```

**规则**：**最后两维**符合矩阵乘法规则（内部维度相等），前面的维度当作"批量"并行处理。

#### 🔸 类型二：按元素运算（数值调节型）
- **代表**：`+`、`-`、`*`（Hadamard 积）、`/`、`**`
- **核心目的**：保持形状不变，对每一维**独立地调节数值**
- **应用**：门控、mask 屏蔽、缩放、残差连接

```python
y = a * x       # 对应位置相乘，叫 Hadamard 积（数学里1899年就有了）
y = x + b       # 按元素加法
```

**⚠️ 重要区分**：
| 运算 | 符号 | 本质 |
|------|------|------|
| 矩阵乘法 | `@` | 维度间"混合" → 产生新特征 |
| Hadamard 积 | `*` | 维度间"独立" → 调节数值 |

#### 🔸 常用聚合运算
```python
x.sum()      # 求和
x.mean()     # 均值
x.max()      # 最大值
torch.softmax(x, dim=-1)   # 概率归一化
```

### 2.7 广播机制（Broadcasting）

当两个张量**形状不同**做按元素运算时，PyTorch 会自动"扩展"较小的张量来对齐。

**广播规则**（从**最后一维**往前比较）：
1. ✅ 两个维度相等
2. ✅ 其中一个维度为 1（会被复制扩展）
3. ✅ 其中一个维度缺失（自动补1再扩展）
4. ❌ 否则报错

**示例**：
```python
[2, 5, 16] + [16]      ✅ → [2, 5, 16]   （b 广播）
[2, 5, 16] + [5, 1]    ✅ → [2, 5, 16]
[2, 5, 8]  + [5]       ❌ 报错（8 vs 5 不匹配）
```

**广播的意义**：逻辑上扩展、物理上不复制 → **高效、简洁**，且支持**参数共享**。

### 2.8 Tensor 与 NumPy 的互转

```python
tensor = torch.from_numpy(ndarray)   # NumPy → Tensor
ndarray = tensor.numpy()              # Tensor → NumPy
```

- **共享内存**：转换快，但修改一方会影响另一方
- 对 PyTorch 不支持的运算，可以转到 NumPy 处理后再转回

### 2.9 自动求导（Autograd）

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)   # 查看梯度
```

- 设置 `requires_grad=True` 后，PyTorch 会自动构建**计算图**
- 调用 `.backward()` 自动计算梯度
- 是深度学习训练的核心机制

---

## 3. NLP 场景下的张量

### 3.1 NLP 张量的标准形态："批长维"三件套

```
[batch_size, seq_len, embed_dim]
  ↑            ↑         ↑
 批量大小    序列长度   每个词的向量维度
```

**举例**：`[32, 128, 768]` 表示一次处理 32 条样本、每条 128 个 token、每个 token 用 768 维向量表示（BERT 的维度）。

### 3.2 数据流演变过程

| 阶段 | 形态 | Shape 示例 |
|------|------|-----------|
| 原始文本 | 字符串 | `"I love NLP"` |
| 分词 | Token 列表 | `["I", "love", "NLP"]` |
| 转 ID | 整数序列（1维） | `[5, 238, 1024]` → `[3]` |
| 组 batch + padding | 2维 | `[batch, seq_len]` = `[32, 128]` |
| Embedding 编码 | 3维 | `[batch, seq_len, embed_dim]` = `[32, 128, 768]` |

### 3.3 Embedding 的两层含义（易混淆 ⚠️）

| 对象 | Shape | 说明 |
|------|-------|------|
| Embedding 层**参数矩阵**（查找表） | `[vocab_size, embed_dim]` | 词表有多大 |
| 单句**编码输出** | `[seq_len, embed_dim]` | 这句话多长 |
| 批量句子**编码输出** | `[batch_size, seq_len, embed_dim]` | NLP 标配 |


### 3.4 Batch 的概念

**Batch（批量）** = 一次喂给模型的一组样本，是深度学习工程化的基础。

**为什么要用 Batch？**
1. 🚀 **GPU 并行加速**：一次处理 32 条和 1 条几乎一样快
2. 📊 **梯度更稳定**：比单样本 SGD 噪声小，比全量 BGD 计算可控
3. 💾 **内存可控**：常见 batch size: 16/32/64/128

**Batch 维度通常占据张量的第一维**。

### 3.5 参数共享（重要设计哲学）

> **关键规律**：参数（W、b）的维度与 batch **无关**，所有 batch 共用同一套参数。

以 Linear 层为例：
```python
x:       [batch, seq_len, in_dim]   # 3维（数据张量，有 batch 维）
W:       [out_dim, in_dim]          # 2维（参数张量，无 batch 维）
b:       [out_dim]                  # 1维（参数张量，无 batch 维）
y = x @ W.T + b  →  [batch, seq_len, out_dim]
```

**为什么参数不需要 batch 维？**
- 参数表达的是"**通用的特征变换规律**"，对所有样本都一样
- 如果每个 batch 都有独立参数，参数量爆炸 + 无法泛化
- 通过**广播机制**，同一套参数自动作用于所有 batch、所有位置

**规律总结**：
```
数据张量：维度高（含 batch、seq 等）
参数张量：维度低（只关心特征变换）
中间通过：矩阵乘法（批量版）+ 广播机制 完成计算
```

---

## 4. torch.nn - 神经网络核心模块

### 4.1 torch.nn 是什么

`torch.nn` 是 PyTorch 的**神经网络工具包**，提供了构建神经网络的所有基础组件：

| 组件类型 | 示例 | 说明 |
|---------|------|------|
| **神经网络层** | `nn.Linear`、`nn.Conv2d`、`nn.LSTM` | 全连接、卷积、循环层等 |
| **激活函数** | `nn.Sigmoid()`、`nn.ReLU()`、`nn.Tanh()` | 非线性变换 |
| **损失函数** | `nn.MSELoss()`、`nn.CrossEntropyLoss()` | 计算预测与真实值的差距 |
| **容器** | `nn.Module`、`nn.Sequential`、`nn.ModuleList` | 组织和管理网络结构 |

---

### 4.2 nn.Module - 所有网络层的基类

**`nn.Module` 是所有神经网络层和模型的基类**，提供了：
- ✅ 自动追踪和管理参数
- ✅ 自动构建计算图
- ✅ 设备管理（CPU/GPU）
- ✅ 训练/评估模式切换
- ✅ 模型保存/加载

#### 使用 nn.Module 构建模型

**你只需要重写两个部分：**

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()  # ✅ 必须调用父类构造函数
        # 1. 定义网络结构（层、参数）
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):  # ✅ 必须重写 forward
        # 2. 定义前向传播逻辑
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x
```

**为什么只写 `forward()` 就能自动反向传播？**

- `nn.Module` 会自动将层的参数注册到 `self.parameters()`
- `forward()` 中的运算会被 PyTorch **自动构建计算图**
- 调用 `loss.backward()` 时，PyTorch 沿计算图反向传播，自动计算所有参数的梯度
- **你不需要（也不应该）手写 `backward()` 方法**

#### 继承 nn.Module 后自动获得的能力

```python
model = MyNetwork()

# ✅ 自动管理所有参数
for name, param in model.named_parameters():
    print(name, param.shape)

# ✅ 自动设备转移
model.to('cuda')  # 把所有参数移到 GPU

# ✅ 自动训练/评估模式切换
model.train()  # 训练模式（启用 Dropout、BatchNorm 等）
model.eval()   # 评估模式（关闭 Dropout、BatchNorm 等）

# ✅ 自动保存/加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

---

### 4.3 nn.Linear - 全连接层

**`nn.Linear` = Fully Connected Layer（全连接层）**

```python
nn.Linear(in_features, out_features, bias=True)
```

**内部实现（等价于）：**
```python
# 参数
W: [in_features, out_features]  # 权重矩阵
b: [out_features]                # 偏置向量

# 前向传播
output = input @ W.T + b
```

**为什么叫 Linear？**
- 执行的是**线性变换**（Affine Transformation）：`y = Wx + b`
- 对应你手搓代码中的：`z = numpy.dot(x, w) + b`

**对比手搓版本：**

| 手搓 NumPy 版本 | PyTorch 版本 |
|----------------|-------------|
| `w = numpy.random.randn(in_dim, out_dim)` | `layer = nn.Linear(in_dim, out_dim)` |
| `b = numpy.zeros(out_dim)` | （自动包含在 Linear 内） |
| `z = numpy.dot(x, w) + b` | `z = layer(x)` |

**示例：**
```python
layer = nn.Linear(10, 5)  # 输入10维 → 输出5维

x = torch.randn(32, 10)   # [batch_size, in_features]
y = layer(x)               # [batch_size, out_features] = [32, 5]

print(layer.weight.shape)  # [5, 10] - 注意是转置的
print(layer.bias.shape)    # [5]
```

---

### 4.4 容器：nn.Sequential vs nn.ModuleList

#### 4.4.1 nn.ModuleList - 参数注册的列表

**`nn.ModuleList` = 能让 PyTorch 识别参数的特殊列表**

```python
# ❌ 错误：普通列表，参数不会被注册
self.layers = [nn.Linear(2, 4), nn.Linear(4, 1)]

# ✅ 正确：ModuleList，参数会自动注册
self.layers = nn.ModuleList([
    nn.Linear(2, 4),
    nn.Linear(4, 1)
])
```

**关键区别：**
- **普通 `list`**：`model.parameters()` 找不到里面层的参数 → 无法训练！
- **`nn.ModuleList`**：`model.parameters()` 能自动找到所有层的参数

**使用场景：** 动态构建网络（层数不固定）

```python
class DynamicNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 根据 layer_sizes 动态创建层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x

# 灵活创建不同深度的网络
model1 = DynamicNetwork([2, 4, 1])       # 2层
model2 = DynamicNetwork([2, 8, 6, 4, 1]) # 4层
```

**⚠️ 注意：使用 `nn.ModuleList` 仍需要手写 `forward()`**

---

#### 4.4.2 nn.Sequential - 自动顺序执行的容器

**`nn.Sequential` = 自动按顺序执行的容器，不需要写 `forward()`**

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# 直接使用，自动按顺序执行
output = model(input)
```

**也可以在类中使用：**
```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)  # 一行搞定
```

---

### 4.5 三种构建网络的方式对比

#### 方式 1️⃣ **手动定义（最灵活）**
```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):  # ✅ 需要手写
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

#### 方式 2️⃣ **nn.ModuleList（需要手写 forward）**
```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 4),
            nn.Linear(4, 1)
        ])
    
    def forward(self, x):  # ✅ 需要手写
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x
```

#### 方式 3️⃣ **nn.Sequential（不需要写 forward）**
```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
# ✅ 不需要写 forward，自动顺序执行
```

---

### 4.6 对比总结

| 特性 | 手动定义 | `nn.ModuleList` | `nn.Sequential` |
|------|---------|----------------|-----------------|
| **需要写 `forward()`** | ✅ | ✅ | ❌ |
| **自动顺序执行** | ❌ | ❌ | ✅ |
| **灵活性** | 最高 | 高 | 低 |
| **适用场景** | 复杂网络 | 动态层数 | 简单顺序网络 |
| **支持跳跃连接** | ✅ | ✅ | ❌ |
| **支持条件分支** | ✅ | ✅ | ❌ |

**选择建议：**
- ✅ **简单顺序网络**（一层接一层）→ 用 `nn.Sequential`
- ✅ **动态层数网络** → 用 `nn.ModuleList`
- ✅ **复杂逻辑**（跳跃连接、残差、分支）→ 手动定义

---

### 4.7 自动求导与反向传播

#### 手搓代码 vs PyTorch

**手搓 NumPy 版本（model_train_v2.py）：**
```python
for epoch in range(epochs):
    # 1. 前向传播
    activations = self.forward(X)
    predictions = activations[-1]
    
    # 2. 计算损失
    loss = self.compute_loss(predictions, y)
    
    # 3. 反向传播（手写45行链式法则）
    weight_gradients, bias_gradients = self.compute_gradients(activations, y)
    
    # 4. 参数更新（手写梯度下降）
    self.update_parameters(weight_gradients, bias_gradients, learning_rate)
```

**PyTorch 版本（等价代码）：**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # 1. 前向传播
    predictions = model(X)
    
    # 2. 计算损失
    loss = criterion(predictions, y)
    
    # 3. 反向传播（自动！）
    optimizer.zero_grad()  # 清空上一轮的梯度
    loss.backward()        # ✨ 自动计算所有梯度
    
    # 4. 参数更新（自动！）
    optimizer.step()       # 自动更新所有参数
```

**对应关系：**

| NumPy 手搓版本 | PyTorch 等价代码 | 代码量 |
|---------------|-----------------|-------|
| `compute_gradients()` | `optimizer.zero_grad()` + `loss.backward()` | 45行 → 2行 |
| `update_parameters()` | `optimizer.step()` | 10行 → 1行 |

---

#### PyTorch 自动求导原理

**loss.backward() 做了什么？**

1. **前向传播时**：PyTorch 自动记录所有运算，构建**计算图**
   ```python
   x → [Linear] → h → [Sigmoid] → a → [Linear] → y → [Loss] → loss
   ```

2. **backward() 时**：沿计算图反向传播，自动应用**链式法则**
   ```python
   dloss/dW1 = dloss/dy × dy/da × da/dh × dh/dW1
   ```

3. **梯度存储**：每个参数的梯度存在 `param.grad` 中

**示例验证：**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3 + 5
loss = y ** 2

loss.backward()  # 自动计算梯度

print(x.grad)  # dloss/dx = 2*y*3 = 2*(3*2+5)*3 = 66
```

---

#### 为什么需要 optimizer.zero_grad()？

**关键区别：**

```python
# NumPy 版本：每次创建新列表，旧梯度自动丢弃
def compute_gradients(self, ...):
    weight_gradients = []  # ← 每次都是新列表
    return weight_gradients

# PyTorch 版本：梯度累加到 param.grad 上
loss.backward()  # ← 梯度会累加，不会自动清空
```

**如果不清空梯度：**
```python
for i in range(3):
    loss = compute_loss()
    # optimizer.zero_grad()  # ← 忘记清空
    loss.backward()
    # 第1轮：grad = g1
    # 第2轮：grad = g1 + g2  ← 累加了！
    # 第3轮：grad = g1 + g2 + g3
```

**正确的训练循环（牢记）：**
```python
optimizer.zero_grad()  # 1. 清空旧梯度
loss.backward()        # 2. 计算新梯度（会累加）
optimizer.step()       # 3. 更新参数
```

**为什么设计成累加？**
- 支持**梯度累积**（Gradient Accumulation）
- 当 batch 太大、显存不够时，可以拆成多个小 batch 累加梯度：
  ```python
  optimizer.zero_grad()
  for mini_batch in large_batch:
      loss = compute_loss(mini_batch)
      loss.backward()  # 梯度累加
  optimizer.step()     # 一次更新所有累积的梯度
  ```

---

### 4.8 核心总结

#### torch.nn 知识地图

```
torch.nn
├── 基础组件
│   ├── nn.Linear        → 全连接层
│   ├── nn.Sigmoid       → 激活函数
│   └── nn.MSELoss       → 损失函数
│
├── 容器（组织网络）
│   ├── nn.Module        → 所有层的基类（必须继承）
│   ├── nn.Sequential    → 顺序容器（不需要写 forward）
│   └── nn.ModuleList    → 参数注册列表（需要写 forward）
│
└── 自动机制
    ├── 自动参数注册    → model.parameters()
    ├── 自动求导        → loss.backward()
    └── 自动优化        → optimizer.step()
```

#### 你只需要做的事

| 你需要做 | PyTorch 自动给你 |
|---------|-----------------|
| 继承 `nn.Module` | 自动参数管理 |
| 定义 `__init__()` 中的层 | 自动注册到 `parameters()` |
| 定义 `forward()` 的逻辑 | 自动构建计算图 |
| - | 自动支持 `loss.backward()` |
| - | 自动支持 `optimizer.step()` |
| - | 自动支持 GPU/CPU 转移 |

**核心哲学：** `nn.Module` 把"参数管理、计算图构建、反向传播"都自动化了，你只需专注于**网络结构设计**和**前向传播逻辑**。

---