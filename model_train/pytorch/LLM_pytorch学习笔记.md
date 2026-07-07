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

# 5 DataSet
在PyTorch中，数据加载可以通过自定义的数据集对象实现。数据集对象被抽象为Dataset类，实现自定义的数据集需要继承Dataset，并实现以下两个Python魔法方法。

__getitem__()：返回一条数据，或一个样本。obj[index]等价于obj.__getitem__(index)。
__len__()：返回样本的数量。len(obj)等价于obj.__len__()。


# 6 DataLoader
Dataset只负责数据的抽象，调用一次__getitem__返回一个样本。然而，在训练神经网络时，一次处理的对象是一个batch的数据，同时还需要对一批数据进行打乱顺序和并行加速等操作。考虑到这一点，PyTorch提供了DataLoader实现这些功能。

DataLoader的定义如下：

DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
它主要有以下几个参数。

dataset：加载的数据集（Dataset对象）。
batch_size：一个batch的大小。
shuffle：是否将数据打乱。
sampler：样本抽样，后续会详细介绍。
batch_sampler：与sampler类似，一次返回一个batch的索引（该参数与batch_size、shuffle、sampler和drop_last不兼容）。
num_workers：使用多进程加载的进程数，0代表不使用多进程。
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可。
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转移到GPU速度更快。
drop_last：dataset中的数据个数可能不是batch_size的整数倍，若drop_last为True，则将多出来不足一个batch的数据丢弃。
timeout：进程读取数据的最大时间，若超时则丢弃数据。
worker_init_fn：每个worker的初始化函数。
prefetch_factor：每个 worker 预先加载的样本数。

---

# 7 实现 nn.Module 的工程规范

> 记录时间：2026-07-03  
> 第一次手写 GPT 模型时暴露的 PyTorch 知识盲区

## 7.1 nn.Module 三件套

所有自定义模型层只需要做三件事，没有第四件：

```python
class MyLayer(nn.Module):
    # 1. 继承 nn.Module
    def __init__(self, ...):
        super().__init__()   # 必须调用
        # 2. __init__ 里定义带参数的子模块
        self.linear = nn.Linear(...)

    # 3. 实现 forward
    def forward(self, x):
        return self.linear(x)

    # backward 不用写，PyTorch 自动处理
```

**backward 不需要自己写**：PyTorch 前向传播时自动记录计算图，`loss.backward()` 自动对每个运算反向推导。

## 7.2 参数定义在哪里

| 做什么 | 在哪里做 |
|--------|----------|
| 定义参数（`nn.Linear` 等） | **每个 class 自己的 `__init__`** |
| 覆盖初始化数値（可选） | 顶层模型的 `_init_weights` + `self.apply()` |

参数必须在所属类的 `__init__` 里定义，PyTorch 才能追踪它、自动求梯度、`model.parameters()` 才能收集到它。

## 7.3 注册子模块的三种方式

| 方式 | 用途 | 示例 |
|--------|------|---------|
| `self.xxx = 子模块` | 单个有名字的层 | `self.norm = nn.LayerNorm(d)` |
| `nn.ModuleList([...])` | N 个同类层堆叠 | `self.blocks = nn.ModuleList([Block() for _ in range(n)])` |
| `nn.Sequential(...)` | 无分支的线性串联 | `self.ffn = nn.Sequential(nn.Linear(...), nn.GELU(), nn.Linear(...))` |

**必须用 `nn.ModuleList`，不能用普通 list：**
```python
self.blocks = [Block() for _ in range(4)]         # ✗ PyTorch 追踪不到参数
self.blocks = nn.ModuleList([Block() for _ in range(4)])  # ✓
```

`Sequential` 适合简单串联，但 Transformer Block 有残差连接（`x = x + sublayer(x)`）无法用 Sequential，必须自己写 forward。

## 7.4 nn.Linear 的调用方式

```python
# 错误：手动取权重进行矩阵乘法（漏掉了 bias）
out = torch.matmul(x, self.linear.weight.t())   # ✗

# 正确：直接调用（自动包含 weight 和 bias）
out = self.linear(x)   # ✓
```

`nn.Linear` 是一个可调用的模块，直接加括号调用即可。手动取 `.weight.t()` 会漏掉 bias，导致训练结果不正确。

## 7.5 register_buffer：不参与训练的固定张量

有些张量不是可训练参数，但希望跟随模型一起保存、一起移动到 GPU，用 `register_buffer`：

```python
# 示例：位置编码（固定就不参与训练）
self.register_buffer('pos_enc', positional_encoding(embed_size, max_length))

# 之同：
self.pos_enc = positional_encoding(...)   # ✗ 普通属性，.to('cuda') 时不会自动移动
self.register_buffer('pos_enc', ...)      # ✓ 自动跟随模型移动和保存
```

| | 普通属性 | register_buffer |
|--|------|-----------------|
| 参与训练 | ✗ | ✗ |
| 跟随 `.to(device)` | ✗ | ✓ |
| 保存到 checkpoint | ✗ | ✓ |
| `model.parameters()` 可见 | ✗ | ✗ |

## 7.6 张量多维切片

```python
# 切片语法：[dim0, dim1, dim2, ...]
# : 表示这个维度全取： start:end 表示取一部分

x = torch.zeros(4, 10, 64)   # (batch=4, seq_len=10, embed=64)

x[:, :5, :]     # 取所有 batch，seq_len 前 5 个，所有 embed
                # 结果：(4, 5, 64)

x[0, :, :]      # 取第 0 个 batch
                # 结果：(10, 64)

x[:, -1, :]     # 取每个 batch 的最后一个位置（推理时取最后 token 的 logits）
                # 结果：(4, 64)
```

**常见错误**：3 维张量切少了维度
```python
pos_enc[:seq_len, :]      # ✗ pos_enc 是 (1, max_len, embed)，需要 3 个维度
pos_enc[:, :seq_len, :]   # ✓
```

## 7.7 从张量读取维度信息

```python
# 两种方式等价
batch = x.size(0)          # 取第 0 维的大小
seq_len = x.size(1)

batch, seq_len, d = x.shape    # 同时解包所有维度，_ 表示不关心的维度
batch, seq_len, _ = x.shape
```

**在 `forward` 里永远用运行时读取，不要在 `__init__` 里固定 batch_size。**

---

# 8 LLM 训练代码实战卡点与认知盲区

> 记录时间：2026-07-06  
> 来源：手写 mini_gpt 训练脚本时反复纠正的问题

## 8.1 LLM 训练数据的本质：自监督学习

**核心认知：LLM 的训练没有外部标签，x 和 y 是从纯文本自己构造的。**

```
传统任务：  x = "这部电影太好看了"    y = "正面情感"   ← 人工标注
LLM 训练：  纯文本 → 编码 → 滑动窗口切分 → 自动构造 x 和 y
```

构造方式：
```
文本 ID：  [t0, t1, t2, t3, t4, t5]
输入 x：   [t0, t1, t2, t3, t4]      ← 前 block_size 个
标签 y：   [t1, t2, t3, t4, t5]      ← 整体右移一位
```

模型的任务：看到前面的 token，预测下一个 token。这就是**自监督学习（Self-Supervised Learning）**。

## 8.2 滑动窗口切分：样本数 = 总 token 数 - block_size

**卡点：一开始完全无法理解这个公式。**

把整段文本编码成一个长 token 列表，然后用固定长度的滑动窗口切：

```
总 token 数 = 10，block_size = 3

[t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]

样本0: x=[t0,t1,t2]  y=[t1,t2,t3]   ← 从 t0 开始
样本1: x=[t1,t2,t3]  y=[t2,t3,t4]   ← 从 t1 开始
...
样本6: x=[t6,t7,t8]  y=[t7,t8,t9]   ← 最后一个
样本数 = 7 = 10 - 3 = 总token数 - block_size
```

**易错点：不要按行切分文本！** 按行切会导致每行长度不同，无法转成 tensor。整段编码 + 滑动窗口才是正确做法。

## 8.3 Tokenizer 设计卡点

| 卡点 | 错误写法 | 正确写法 |
|------|---------|----------|
| 词表重复 | 每行去重后 extend | 全部 extend 后整体去重 |
| `sort()` 返回 None | `list(set(x)).sort()` | `sorted(list(set(x)))` |
| 空字符串 split | `text.split("")` 报错 | `list(text)` 拆成字符 |
| Tokenizer 耦合文件 | 内部硬编码 `load_text()` | 外部传入 text 参数 |

## 8.4 Dataset 实现卡点

| 卡点 | 错误写法 | 正确写法 |
|------|---------|----------|
| `__len__` 返回什么 | 返回文本字符数 | 返回 `len(id_list) - block_size` |
| `__getitem__` 返回类型 | Python list | `torch.Tensor`（dtype=torch.long） |
| 从哪个变量切片 | 从 Python list 切 | 从 tensor 切（`self.id_token_list`） |
| 重复编码 | 每次 `__getitem__` 都调 encoder | `__init__` 里编码一次，存起来 |

## 8.5 训练循环卡点

| 卡点 | 错误写法 | 正确写法 |
|------|---------|----------|
| 前向传播 | `model.forward(x)` | `logits = model(x)` |
| 计算损失 | `model.loss(x, y)` | `nn.functional.cross_entropy(...)` |
| 梯度清零 | `torch.zero_grad()` | `optimizer.zero_grad()` |
| 反向传播 | `loss.backward(l)` | `loss.backward()` |
| CrossEntropy 维度 | 直接传 3D logits | `.view(-1, vocab_size)` 展平成 2D |

## 8.6 模型保存/加载卡点

**核心认知：`state_dict()` 只保存参数值，不保存模型结构。**

```
state_dict = {"embedding.weight": tensor(...), "w_q.weight": tensor(...), ...}
```

所以加载时必须先创建**结构相同**的空模型，再灌入参数。

| 卡点 | 错误写法 | 正确写法 |
|------|---------|----------|
| 只保存权重 | `torch.save(model.state_dict(), ...)` | 同时保存 vocab、model_config |
| eval 用新词表 | `tokenizer = Tokenizer(eval_text)` | 从 checkpoint 恢复训练时的 vocab |
| predict 没传 vocab_size | `model_load(vocab_size)` 外部未定义 | `model_load()` 从文件读取 |

**必须保存的信息：**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': tokenizer.vocab,           # 词表（保证推理和训练一致）
    'vocab_size': tokenizer.vocab_size,
    'model_config': {...},              # 模型超参数
}, 'model.pth')
```

---

# 9 PyTorch 训练生态不熟悉的地方

> 记录时间：2026-07-06  
> 来源：第一次独立实现训练脚本时暴露的 PyTorch 知识盲区

## 9.1 DataLoader 是迭代器，不是列表

```python
# 错误：DataLoader 不支持下标访问
x, y = dataloader[i]           # ✗ TypeError

# 正确：用 for 遍历
for x, y in dataloader:        # ✓
    # 每次迭代自动返回一个 batch
```

DataLoader 内部自动完成：
1. 从 Dataset 取 `batch_size` 条样本
2. 自动拼成 batch tensor
3. `shuffle=True` 时每个 epoch 打乱顺序

## 9.2 model.eval() 不是"评估专用"，是"切换层行为模式"

| 层 | train() 模式 | eval() 模式 |
|---|---|---|
| Dropout | 随机丢弃神经元 | 全部保留 |
| BatchNorm | 用当前 batch 统计量 | 用训练时累积的统计量 |

**推理/评估时都必须先 `model.eval()`**，否则结果不稳定。

## 9.3 torch.no_grad() 和 model.eval() 是两件事

```python
model.eval()                  # 切换层的行为模式
with torch.no_grad():         # 关闭梯度计算，省内存
    logits = model(x)
```

- `model.eval()` → 影响 Dropout/BatchNorm 等层的行为
- `torch.no_grad()` → 不记录计算图，不追踪梯度，省内存

两者作用不同，推理时通常一起用。

## 9.4 CrossEntropyLoss 的维度要求

`nn.functional.cross_entropy` 期望：
- 预测值：`(N, C)` — 二维，N 是样本数，C 是类别数
- 标签：`(N,)` — 一维

LLM 输出是三维 `(batch, seq_len, vocab_size)`，必须展平：

```python
loss = nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (batch*seq_len, vocab_size)
    y.view(-1),                          # (batch*seq_len,)
)
```

**CrossEntropyLoss 内部自动做了 softmax + NLLLoss**，不需要自己写 softmax。

## 9.5 state_dict 只保存参数，不保存结构

```
保存的 = {参数名: 参数值}
不保存 = 模型架构、vocab、超参数
```

所以加载时需要：
1. 先创建结构相同的空模型
2. 再 `load_state_dict()` 灌入参数
3. 超参数和 vocab 要额外保存和恢复

## 9.6 训练/推理时 vocab 必须一致

模型和 tokenizer 是**一对绑定关系**：
- 模型的 embedding 层维度 = 训练时 vocab_size
- tokenizer 的字符→ID 映射必须和训练时完全相同
- eval/predict 时**必须从 checkpoint 恢复训练时的 vocab**

---

# 10 PyTorch 模型构建 & 训练流程完整模板

> 记录时间：2026-07-06  
> 用途：下次写训练脚本时对照着写，减少卡点

## 10.1 完整流程模板

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 第一步：Tokenizer（文本 → 数字 ID）
# ============================================================
class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.vocab = self.build_vocab(text)
        self.vocab_size = len(self.vocab)

    def encoder(self):
        """文本 → token ID 列表"""
        tokens = ...  # 分词逻辑
        return [self.vocab.index(t) for t in tokens]

    def decoder(self, ids):
        """token ID 列表 → 文本"""
        return ''.join([self.vocab[i] for i in ids])

    def build_vocab(self, text):
        """从文本构建词表"""
        vocab = []
        # ... 分词并收集所有不重复的 token
        return sorted(list(set(vocab)))

# ============================================================
# 第二步：Dataset（定义"一条训练样本"）
# ============================================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # 整段文本编码一次
        self.id_list = tokenizer.encoder()
        self.id_tensor = torch.tensor(self.id_list, dtype=torch.long)

    def __len__(self):
        return len(self.id_list) - self.block_size

    def __getitem__(self, index):
        # 滑动窗口切分，返回 Tensor
        x = self.id_tensor[index : index + self.block_size]
        y = self.id_tensor[index + 1 : index + 1 + self.block_size]
        return x, y

# ============================================================
# 第三步：训练函数
# ============================================================
def train(model, dataloader, epochs, lr, device):
    model.to(device)
    model.train()                                    # 训练模式
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in dataloader:                      # DataLoader 自动给 batch
            x, y = x.to(device), y.to(device)       # 数据移到设备

            logits = model(x)                        # 前向传播
            loss = nn.functional.cross_entropy(       # 计算损失
                logits.view(-1, logits.size(-1)),     # (batch*seq, vocab)
                y.view(-1),                           # (batch*seq,)
            )

            optimizer.zero_grad()                     # 清零梯度
            loss.backward()                           # 反向传播
            optimizer.step()                          # 更新参数

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# ============================================================
# 第四步：评估函数
# ============================================================
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()                                      # 评估模式
    total_loss = 0
    with torch.no_grad():                             # 不计算梯度
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))    # 困惑度 = exp(loss)
    print(f'Eval Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}')

# ============================================================
# 第五步：推理/生成函数
# ============================================================
def generate(model, tokenizer, prompt, max_new_tokens, max_length, device):
    model.to(device)
    model.eval()
    tokenizer.text = prompt
    id_list = tokenizer.encoder()
    input_tensor = torch.tensor([id_list], dtype=torch.long)  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = input_tensor[:, -max_length:]       # 截断防超限
            logits = model(input_tensor.to(device))
            next_logits = logits[:, -1, :]                     # 取最后位置
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    text = tokenizer.decoder(input_tensor[0].tolist())
    return text

# ============================================================
# 第六步：保存/加载
# ============================================================
def save_checkpoint(model, tokenizer, model_config, path='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'vocab_size': tokenizer.vocab_size,
        'model_config': model_config,
    }, path)

def load_checkpoint(path='model.pth'):
    ckpt = torch.load(path)
    model = MyModel(**ckpt['model_config'])
    model.load_state_dict(ckpt['model_state_dict'])
    vocab = ckpt['vocab']
    return model, vocab

# ============================================================
# 第七步：主程序串联
# ============================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 准备数据
    text = load_text('train.txt')
    tokenizer = Tokenizer(text)
    dataset = TextDataset(text, tokenizer, block_size=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. 创建模型
    model = MyModel(vocab_size=tokenizer.vocab_size, ...)

    # 3. 训练
    train(model, dataloader, epochs=100, lr=1e-4, device=device)

    # 4. 保存
    save_checkpoint(model, tokenizer, model_config={...})

    # 5. 评估
    model, vocab = load_checkpoint()
    eval_dataset = TextDataset(eval_text, ...)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    evaluate(model, eval_loader, device)

    # 6. 推理
    text = generate(model, tokenizer, "今天天气", max_new_tokens=100, ...)
    print(text)
```

## 10.2 关键检查清单

写训练脚本时逐项对照：

| 检查项 | 要点 |
|--------|------|
| Tokenizer | 词表是否去重？encoder/decoder 是否对称？ |
| Dataset | `__getitem__` 返回 Tensor？dtype 是 long？ |
| DataLoader | batch_size？shuffle=True？ |
| 模型 | 参数是否和 vocab_size 对齐？ |
| 训练循环 | zero_grad → backward → step 顺序对吗？ |
| Loss | logits 是否 view 成 2D？ |
| 设备 | model.to(device)？x, y 也 .to(device)？ |
| 保存 | 是否保存了 vocab 和 model_config？ |
| 加载 | 是否从 checkpoint 恢复 vocab？ |
| eval | model.eval() + torch.no_grad()？ |
| generate | 是否截断到 max_length？keepdim=True？ |

---

# 11 训练实战踩坑记录

> 记录时间：2026-07-07  
> 场景：在 H20 (47.5GB) K8s Pod 上训练手写 mini_gpt，遇到的工程问题及解决

## 11.1 GPU 利用率 0% 但 CPU 1000%

**现象**：模型和数据都 `.to(device)` 了，但 `nvidia-smi` 显示 GPU 利用率 0%、显存 0，CPU 却拉满。

**原因**：模型太小（batch_size=8, embed=256, 10层），GPU 每次计算只需几微秒，**内核启动开销比计算本身还长**。

**解决**：增大 batch_size（8→16/32/64），让 GPU 每次做更多工作。

**认知**：GPU 利用率 ≠ 模型是否在 GPU 上跑。太小的计算量喂不饱 GPU，利用率就是 0。

## 11.2 vocab_size 爆炸导致 OOM

**现象**：160M 参数模型，训练时占用 48GB 显存的 60%。重启后立刻又占满，不是僵尸进程。

**原因**：jieba 分词 5000 条中文维基百科 → vocab_size=296,317。Embedding + Output 两层占参数 95%，logits 张量 (16×128×296k×4B = 2.4GB) 每个 batch 都要创建/销毁。

**解决**：按词频截断词表，`MAX_VOCAB_SIZE=50000`，低频词映射到 `<UNK>`。

```python
from collections import Counter

def build_vocab(self, text, max_vocab_size):
    words = jieba.lcut(text)
    word_counts = Counter(words)
    vocab = ['<UNK>'] + [w for w, _ in word_counts.most_common(max_vocab_size - 1)]
    return vocab

# encoder 中用 get 而不是 if in，确保未登录词映射到 UNK
return [self.word2id.get(token, self.unk_id) for token in text_list]
```

**认知**：
- vocab_size 不影响模型“深度”（内部维度 embed_size），但决定了“出口宽度”
- 参数量不能只看数字，要看**参数分布在哪**。Embedding/Output 与 vocab 正比，Decoder 与 hidden_size 正比
- logits 张量 = batch × seq × vocab，可能比模型参数本身还大

## 11.3 num_workers 在 K8s/Windows 下引发内存翻倍

**现象**：设置 `num_workers=4` 后 OOM。

**原因**：DataLoader 的每个 worker 进程会**复制整个 Dataset 内存**。5 个进程（主进程 + 4 worker）= 5 份数据副本。

**解决**：`num_workers=0`，在主进程加载数据。

**认知**：`num_workers` 不是万能的。Dataset 很大时（如 jieba 分词百万 token），多进程反而吃爆内存。Linux 下可以用，但 K8s Pod / Windows 下优先设为 0。

## 11.4 OOM 后 GPU 僵尸进程

**现象**：训练 OOM 崩溃后，`nvidia-smi` 显示进程占 46GB 显存，但 `ps aux | grep <PID>` 找不到进程。

**原因**：进程崩溃时 GPU 驱动未正确释放 CUDA context，形成孤儿上下文。

**解决**：
- 有 root：`sudo nvidia-smi --gpu-reset`
- K8s Pod：`kubectl delete pod` 重建（容器运行时会自动回收 GPU 资源）
- 终极：重启机器

**预防**：
```python
try:
    train()
except Exception as e:
    print(f"训练异常: {e}")
    raise
finally:
    torch.cuda.empty_cache()  # 确保异常时释放显存
```

## 11.5 训练太慢：数据量过大

**现象**：521,365 个 batch/epoch，每 epoch 要 39 小时。

**原因**：5000 条中文维基百科 ≈ 834 万 token，滑动窗口生成样本太多。

**解决**：
- 减少训练数据：5000 条 → 1000 条
- 增大 batch_size：16 → 32（利用剩余显存）

**认知**：学习用 mini_gpt 不需要太多数据，1000 条足够学到基本模式。

## 11.6 训练显存排查清单

下次训练前先过一遍：

| 检查项 | 怎么查 | 预期 |
|--------|--------|------|
| vocab_size | `print(tokenizer.vocab_size)` | 30k-100k 合理，超过则截断 |
| 模型参数量 | `sum(p.numel() for p in model.parameters())` | 和 vocab_size 对比看分布 |
| 单 batch logits | `batch × seq × vocab × 4B` | 不应超过几 GB |
| nvidia-smi | 训练时监控 | 不应超过 80% |
| num_workers | 根据环境设置 | K8s/Windows 用 0 |