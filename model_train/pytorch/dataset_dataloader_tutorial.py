"""
PyTorch 数据预处理完整教程
Dataset 和 DataLoader 的核心概念

核心理解：
- Dataset: 对数据集的抽象，定义"如何获取一条数据"
- DataLoader: 负责批量加载、打乱、并行处理

类比：
Dataset  = 图书馆的书架（知道每本书在哪）
DataLoader = 图书管理员（帮你一次拿多本书，还能打乱顺序）
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd


# ============================================================================
# 第1部分：理解 Dataset 的核心
# ============================================================================

print("=" * 80)
print("第1部分：Dataset - 数据集的抽象")
print("=" * 80)

class SimpleDataset(Dataset):
    """
    自定义Dataset的最小示例

    必须实现三个方法：
    1. __init__: 初始化（加载数据、预处理等）
    2. __len__: 返回数据集大小
    3. __getitem__: 根据索引返回一条数据
    """

    def __init__(self, data, labels):
        """
        初始化数据集

        参数:
            data: 特征数据 (N, features)
            labels: 标签 (N,)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        返回数据集大小
        DataLoader 会调用这个方法来知道有多少数据
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回一条数据

        参数:
            idx: 索引（0 到 len-1）

        返回:
            (x, y): 一条样本的特征和标签
        """
        # 获取第 idx 条数据
        x = self.data[idx]
        y = self.labels[idx]

        # 这里可以做数据增强、转换等
        # 例如: x = self.transform(x)

        return x, y


# 示例：创建一个简单的数据集
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
y = np.array([0, 1, 0, 1, 0], dtype=np.int64)

dataset = SimpleDataset(X, y)

print(f"\n数据集大小: {len(dataset)}")
print(f"第0条数据: {dataset[0]}")
print(f"第2条数据: {dataset[2]}")

print("\n✅ 关键理解:")
print("   Dataset只是定义了'如何获取数据'")
print("   但不负责批量加载、打乱等操作")


# ============================================================================
# 第2部分：理解 DataLoader 的核心
# ============================================================================

print("\n" + "=" * 80)
print("第2部分：DataLoader - 批量加载和打乱")
print("=" * 80)

# 使用 DataLoader 包装 Dataset
dataloader = DataLoader(
    dataset,
    batch_size=2,      # 每次取2条数据
    shuffle=True,      # 打乱顺序
    num_workers=0      # 单进程加载（多进程加载用于加速）
)

print(f"\nDataLoader 配置:")
print(f"  batch_size=2  → 每次返回2条数据")
print(f"  shuffle=True  → 每个epoch都会打乱顺序")

print("\n遍历 DataLoader（相当于一个epoch）:")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  batch_x shape: {batch_x.shape}")  # [batch_size, features]
    print(f"  batch_x:\n{batch_x}")
    print(f"  batch_y: {batch_y}")

print("\n✅ 关键理解:")
print("   DataLoader 自动完成:")
print("   1. 分批（batch）")
print("   2. 打乱（shuffle）")
print("   3. 并行加载（num_workers）")


# ============================================================================
# 第3部分：DataLoader 的核心参数
# ============================================================================

print("\n" + "=" * 80)
print("第3部分：DataLoader 核心参数详解")
print("=" * 80)

# 参数1: batch_size
print("\n【参数1】batch_size - 批量大小")
print("-" * 40)

for bs in [2, 3, 5]:
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    print(f"\nbatch_size={bs}:")
    for i, (x, y) in enumerate(loader):
        print(f"  Batch {i}: x.shape={x.shape}")

print("\n说明: batch_size 控制每次返回多少条数据")
print("      最后一个batch可能不足batch_size（如果数据总数不能整除）")


# 参数2: shuffle
print("\n【参数2】shuffle - 是否打乱")
print("-" * 40)

print("\nshuffle=False (顺序读取):")
loader_no_shuffle = DataLoader(dataset, batch_size=2, shuffle=False)
for i, (x, y) in enumerate(loader_no_shuffle):
    print(f"  Batch {i}: y={y.tolist()}")

print("\nshuffle=True (打乱顺序):")
loader_shuffle = DataLoader(dataset, batch_size=2, shuffle=True)
for i, (x, y) in enumerate(loader_shuffle):
    print(f"  Batch {i}: y={y.tolist()}")

print("\n说明: shuffle=True 每个epoch都会重新打乱")
print("      训练集一般用shuffle=True，验证集/测试集用shuffle=False")


# 参数3: drop_last
print("\n【参数3】drop_last - 是否丢弃最后不足一批的数据")
print("-" * 40)

print("\ndrop_last=False (保留最后的batch):")
loader = DataLoader(dataset, batch_size=2, drop_last=False)
print(f"  总batch数: {len(loader)}")  # 5条数据 / batch_size=2 = 3个batch

print("\ndrop_last=True (丢弃最后的batch):")
loader = DataLoader(dataset, batch_size=2, drop_last=True)
print(f"  总batch数: {len(loader)}")  # 5条数据 / batch_size=2 = 2个batch（丢弃最后1条）

print("\n说明: drop_last=True 保证所有batch大小一致")
print("      一般训练集用False，某些场景（如BatchNorm）可能用True")


# ============================================================================
# 第4部分：TensorDataset - 快速创建Dataset
# ============================================================================

print("\n" + "=" * 80)
print("第4部分：TensorDataset - 快速创建Dataset")
print("=" * 80)

print("\n如果数据已经是Tensor，可以用TensorDataset快速创建:")

X_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
y_tensor = torch.tensor([0, 1, 0], dtype=torch.long)

# 快速创建Dataset（不需要自己写类）
tensor_dataset = TensorDataset(X_tensor, y_tensor)

print(f"\n数据集大小: {len(tensor_dataset)}")
print(f"第0条数据: {tensor_dataset[0]}")

# 配合DataLoader使用
loader = DataLoader(tensor_dataset, batch_size=2, shuffle=True)
print("\n遍历DataLoader:")
for batch_x, batch_y in loader:
    print(f"  batch_x: {batch_x}, batch_y: {batch_y}")

print("\n✅ TensorDataset 适用场景:")
print("   数据量小，可以一次性加载到内存")
print("   数据已经预处理好，直接转成Tensor")


# ============================================================================
# 第5部分：NLP场景 - 文本数据集示例
# ============================================================================

print("\n" + "=" * 80)
print("第5部分：NLP场景 - 文本数据集")
print("=" * 80)

class TextDataset(Dataset):
    """
    文本分类数据集

    处理流程：
    1. 加载原始文本和标签
    2. 分词
    3. 转换为ID序列
    4. Padding到统一长度
    """

    def __init__(self, texts, labels, vocab, max_len=50):
        """
        参数:
            texts: 文本列表 ["I love NLP", "PyTorch is great", ...]
            labels: 标签列表 [1, 0, ...]
            vocab: 词表字典 {"word": id, ...}
            max_len: 序列最大长度（padding/截断）
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        返回一条处理好的数据
        """
        # 1. 获取原始文本和标签
        text = self.texts[idx]
        label = self.labels[idx]

        # 2. 分词（简化版：按空格分）
        tokens = text.lower().split()

        # 3. 转换为ID序列
        token_ids = [self.vocab.get(token, 0) for token in tokens]  # 0是<UNK>

        # 4. Padding/截断到max_len
        if len(token_ids) < self.max_len:
            # Padding：补0到max_len
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        else:
            # 截断：只保留前max_len个
            token_ids = token_ids[:self.max_len]

        # 5. 转换为Tensor
        x = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


# 示例数据
texts = [
    "I love machine learning",
    "PyTorch is awesome",
    "Deep learning is cool",
    "NLP is interesting"
]
labels = [1, 1, 1, 0]

# 构建简单词表
all_words = set()
for text in texts:
    all_words.update(text.lower().split())

vocab = {word: idx+1 for idx, word in enumerate(sorted(all_words))}  # 从1开始，0留给<PAD>/<UNK>
vocab['<PAD>'] = 0
vocab['<UNK>'] = 0

print(f"\n词表大小: {len(vocab)}")
print(f"词表示例: {dict(list(vocab.items())[:5])}")

# 创建数据集
text_dataset = TextDataset(texts, labels, vocab, max_len=10)

print(f"\n数据集大小: {len(text_dataset)}")
print(f"\n第0条数据:")
x, y = text_dataset[0]
print(f"  输入序列: {x}")
print(f"  标签: {y}")

# 使用DataLoader
text_loader = DataLoader(text_dataset, batch_size=2, shuffle=True)

print("\n遍历一个batch:")
for batch_x, batch_y in text_loader:
    print(f"  batch_x shape: {batch_x.shape}")  # [batch_size, max_len]
    print(f"  batch_x:\n{batch_x}")
    print(f"  batch_y: {batch_y}")
    break  # 只看第一个batch


# ============================================================================
# 第6部分：训练循环中的使用
# ============================================================================

print("\n" + "=" * 80)
print("第6部分：训练循环中的典型用法")
print("=" * 80)

print("""
# 标准训练循环

# 1. 创建数据集和DataLoader
train_dataset = TextDataset(train_texts, train_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TextDataset(val_texts, val_labels, vocab)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证集不打乱

# 2. 训练循环
for epoch in range(num_epochs):
    # ===== 训练阶段 =====
    model.train()  # 切换到训练模式

    for batch_x, batch_y in train_loader:  # 自动分批、打乱
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ===== 验证阶段 =====
    model.eval()  # 切换到评估模式

    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            # 计算验证指标...
""")


# ============================================================================
# 第7部分：核心总结
# ============================================================================

print("\n" + "=" * 80)
print("核心总结")
print("=" * 80)

summary = """
📚 Dataset vs DataLoader

┌─────────────────────────────────────────────────────────────┐
│ Dataset（数据集）                                            │
│ ─────────────────────────────────────────────────────────── │
│ 职责：定义"如何获取一条数据"                                 │
│                                                             │
│ 必须实现：                                                   │
│   __init__(self, ...):     初始化，加载数据                  │
│   __len__(self):           返回数据集大小                     │
│   __getitem__(self, idx):  根据索引返回一条数据               │
│                                                             │
│ 适合在这里做：                                               │
│   ✅ 数据加载（读文件、数据库等）                              │
│   ✅ 数据预处理（分词、转ID、归一化）                          │
│   ✅ 数据增强（随机翻转、裁剪等）                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DataLoader（数据加载器）                                     │
│ ─────────────────────────────────────────────────────────── │
│ 职责：批量加载、打乱、并行处理                                │
│                                                             │
│ 核心参数：                                                   │
│   dataset:      Dataset对象                                 │
│   batch_size:   批量大小（一次取几条）                        │
│   shuffle:      是否打乱（训练True，验证/测试False）           │
│   num_workers:  并行加载的进程数（0=单进程）                   │
│   drop_last:    是否丢弃最后不足一批的数据                     │
│                                                             │
│ 自动完成：                                                   │
│   ✅ 分批（自动把数据分成一个个batch）                         │
│   ✅ 打乱（每个epoch重新打乱）                                 │
│   ✅ 并行（多进程加速数据加载）                                │
│   ✅ 自动Collate（把多条数据堆叠成batch）                      │
└─────────────────────────────────────────────────────────────┘

📋 使用流程

1. 定义Dataset
   class MyDataset(Dataset):
       def __init__(self, ...):
           self.data = load_data(...)

       def __len__(self):
           return len(self.data)

       def __getitem__(self, idx):
           return self.data[idx], self.label[idx]

2. 创建DataLoader
   train_loader = DataLoader(
       MyDataset(...),
       batch_size=32,
       shuffle=True
   )

3. 训练循环
   for epoch in range(epochs):
       for batch_x, batch_y in train_loader:
           # 训练代码...

🎯 关键理解

Dataset  → "数据在哪，怎么拿"（单条）
DataLoader → "批量拿，打乱拿"（批次）

类比：
Dataset  = 图书馆书架（知道每本书在哪）
DataLoader = 管理员（帮你批量拿书，还能打乱顺序）

🔑 常见场景

┌──────────────┬──────────────┬──────────────┐
│              │ 训练集        │ 验证/测试集   │
├──────────────┼──────────────┼──────────────┤
│ shuffle      │ True         │ False        │
│ drop_last    │ False        │ False        │
│ num_workers  │ 2-4          │ 0-2          │
└──────────────┴──────────────┴──────────────┘

✨ 快速上手

如果数据已经是Tensor：
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

如果需要自定义处理：
    继承Dataset，实现__len__和__getitem__
"""

print(summary)

print("\n" + "=" * 80)
print("教程结束！现在你可以处理真实的数据集了 🎉")
print("=" * 80)
