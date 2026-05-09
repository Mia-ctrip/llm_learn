import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


# ==================== 第3步：生成随机训练数据 ====================
def generate_random_data(num_samples=1000, seq_len=50, vocab_size=1000):
    """
    生成随机数据用于测试
    返回：训练集、验证集、测试集
    """
    print("正在生成随机数据...")

    # 生成随机 token IDs
    X_train = torch.randint(0, vocab_size, (num_samples, seq_len))
    y_train = torch.randint(0, 2, (num_samples,))

    X_val = torch.randint(0, vocab_size, (200, seq_len))
    y_val = torch.randint(0, 2, (200,))

    X_test = torch.randint(0, vocab_size, (200, seq_len))
    y_test = torch.randint(0, 2, (200,))

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


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

        # 打印结果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f}, Val Acc:   {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✅ 最佳模型！验证准确率: {best_val_acc*100:.2f}%")

        print("-" * 60)

    return model


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

    # 1. 生成数据
    X_train, y_train, X_val, y_val, X_test, y_test = generate_random_data()

    # 2. 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 3. 创建模型
    model = myFirstModel(vocab_size=1000, embed_dim=64, hidden_dim=128, num_classes=2)
    print(f"\n模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 4. 训练模型
    model = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)

    # 5. 测试模型
    evaluate_model(model, test_loader)

    print("\n训练完成！✅")
    print("\n下一步可以尝试:")
    print("  1. 修改 hidden_dim (如 256)")
    print("  2. 修改 learning_rate (如 0.0001)")
    print("  3. 修改 num_epochs (如 20)")
    print("  4. 加载真实数据集")




