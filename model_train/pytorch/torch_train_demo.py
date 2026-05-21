import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # ✅ 添加画图库


# ==================== 第1步：定义 LSTM 模块 ====================
class myLSTM(nn.Module):
    """自定义LSTM层"""
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output[:, -1, :]  # 取最后一个时间步


# ==================== 第2步：定义完整模型 ====================
class myFirstModel(nn.Module):
    """情感分类模型"""
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128, num_classes=2):
        super(myFirstModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = myLSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)    # [batch, seq] -> [batch, seq, 64]
        x = self.lstm(x)          # [batch, seq, 64] -> [batch, 128]
        x = self.fc(x)            # [batch, 128] -> [batch, 2]
        return x


# ==================== 第3步：组织数据 ====================
class MovieDataset(Dataset):    
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


def load_data(data_path, min_freq=2):
    """
    加载 IMDB 数据集并构建词表

    参数:
        data_path: CSV 文件路径（需要有 'review' 和 'sentiment' 列）
        min_freq: 词频阈值，低于此频率的词会被过滤

    返回:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab
    """
    # 1. 读取 CSV
    df = pd.read_csv(data_path)  # ✅ 修正：load_csv → read_csv

    # 2. 转换标签为数字（positive=1, negative=0）
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 3. 构建词表
    from collections import Counter
    word_counts = Counter()

    for i in range(len(df)):
        review = df.iloc[i]['review']  # ✅ 修正：sentence[review] → sentence['review']
        tokens = review.lower().split()
        word_counts.update(tokens)

    # 只保留出现>=min_freq次的词
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    print(f"词表大小: {len(vocab)}")

    # 4. 拆分数据集：训练 70% / 验证 15% / 测试 15%
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # 5. 提取文本和标签
    train_texts = train['review'].tolist()  # ✅ 修正：train[review] → train['review']
    train_labels = train['label'].tolist()

    val_texts = val['review'].tolist()
    val_labels = val['label'].tolist()

    test_texts = test['review'].tolist()
    test_labels = test['label'].tolist()

    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab


# ==================== 第4步：训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """
    训练模型
    """
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n开始训练，共 {num_epochs} 个 epoch...")
    print("=" * 60)

    # ✅ 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # ✅ 记录指标
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✅ 最佳模型！验证准确率: {best_val_acc*100:.2f}%")

        print("-" * 60)

    # ✅ 训练结束后画图
    plot_training_history(history)

    return model


# ✅ 新增：画图函数
def plot_training_history(history):
    """
    画出训练历史曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Acc')
    plt.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)  # 保存图片
    print(f"\n📊 训练曲线已保存到: training_history.png")
    plt.show()  # 显示图片





# ==================== 第5步：评估函数 ====================
def evaluate_model(model, test_loader):
    """
    在测试集上评估模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_acc = correct / total
    print(f"\n{'='*60}")
    print(f"测试集最终准确率: {test_acc*100:.2f}%")
    print(f"{'='*60}")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PyTorch 情感分类模型训练 Demo")
    print("="*60)

    # 1. 加载数据（需要先下载 IMDB 数据集到本地）
    data_path = "/home/powerop/work/data/IMDB Dataset.csv"  # ✅ 修改为你的 CSV 文件路径

    print("\n正在加载数据...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab = load_data(data_path)

    # 2. 创建 Dataset
    train_dataset = MovieDataset(train_texts, train_labels, vocab, max_len=512)
    val_dataset = MovieDataset(val_texts, val_labels, vocab, max_len=512)
    test_dataset = MovieDataset(test_texts, test_labels, vocab, max_len=512)

    # 3. 创建 DataLoader
    print("\n创建 DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 4. 创建模型
    print("\n创建模型...")
    model = myFirstModel(
        vocab_size=len(vocab),  # ✅ 使用实际词表大小
        embed_dim=128,
        hidden_dim=256,
        num_classes=2
    )
    print(f"模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 5. 训练模型
    model = train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001)

    # 6. 测试模型
    evaluate_model(model, test_loader)

    print("\n训练完成！✅")
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes



