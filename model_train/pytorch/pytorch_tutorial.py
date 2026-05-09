"""
PyTorch快速入门指南

基于你的model_train_v2.py，理解PyTorch的核心概念
"""

import torch
import torch.nn as nn
import numpy

print("=" * 80)
print("PyTorch Quick Start Guide")
print("=" * 80)

# ============================================================================
# 1. Tensor基础：PyTorch的"numpy数组"
# ============================================================================
print("\n1. Tensor Basics")
print("-" * 80)

# 从numpy转换
x_numpy = numpy.array([[1, 2], [3, 4]])
x_torch = torch.from_numpy(x_numpy).float()

print(f"NumPy array:\n{x_numpy}")
print(f"PyTorch tensor:\n{x_torch}")
print(f"Tensor shape: {x_torch.shape}")
print(f"Tensor dtype: {x_torch.dtype}")

# 转回numpy
x_back = x_torch.numpy()
print(f"Back to NumPy:\n{x_back}")

# ============================================================================
# 2. 自动求导：requires_grad=True
# ============================================================================
print("\n2. Automatic Differentiation (Autograd)")
print("-" * 80)

# 对比：手动求导 vs 自动求导
print("\n--- Manual calculation ---")
w_val = 2.0
x_val = 3.0
y_val = w_val * x_val  # y = 6
loss_val = (y_val - 5.0) ** 2  # loss = (6-5)^2 = 1

# 手动求导
dloss_dy = 2 * (y_val - 5.0)  # = 2 * 1 = 2
dy_dw = x_val  # = 3
dloss_dw = dloss_dy * dy_dw  # = 2 * 3 = 6

print(f"w = {w_val}, x = {x_val}")
print(f"y = w * x = {y_val}")
print(f"loss = (y - 5)^2 = {loss_val}")
print(f"dloss/dw (manual) = {dloss_dw}")

print("\n--- PyTorch autograd ---")
w = torch.tensor([2.0], requires_grad=True)  # 需要梯度
x = torch.tensor([3.0])
y = w * x
loss = (y - 5.0) ** 2

loss.backward()  # 自动计算梯度！

print(f"w = {w.item()}, x = {x.item()}")
print(f"y = w * x = {y.item()}")
print(f"loss = (y - 5)^2 = {loss.item()}")
print(f"dloss/dw (autograd) = {w.grad.item()}")
print("\nSame result! No manual chain rule needed!")

# ============================================================================
# 3. nn.Module：组织模型
# ============================================================================
print("\n3. Building Models with nn.Module")
print("-" * 80)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # nn.Linear = 一个全连接层（W·x + b）
        self.fc1 = nn.Linear(2, 4)  # 输入2维 → 输出4维
        self.fc2 = nn.Linear(4, 1)  # 输入4维 → 输出1维
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

model = SimpleNet()
print(f"Model structure:\n{model}")

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape {param.shape}")

# 查看权重
print(f"\nFirst layer weight:\n{model.fc1.weight.data}")
print(f"First layer bias:\n{model.fc1.bias.data}")

# ============================================================================
# 4. 训练循环：对比NumPy和PyTorch
# ============================================================================
print("\n4. Training Loop")
print("-" * 80)

# XOR数据
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

print(f"Training data:")
print(f"X:\n{X}")
print(f"y:\n{y}")

# 固定随机种子保证可重复
torch.manual_seed(42)
numpy.random.seed(42)

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

print("\nTraining for 5000 epochs...")
for epoch in range(5000):
    # === NumPy版本的三个步骤 ===
    # 1. forward
    predictions = model(X)

    # 2. compute_gradients (自动！)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()

    # 3. update_parameters (自动！)
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 4999:
        print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

print("\nFinal predictions:")
with torch.no_grad():
    preds = model(X)
    for i in range(len(X)):
        print(f"  {X[i].tolist()} -> {preds[i].item():.4f} (expected {y[i].item()})")

# ============================================================================
# 5. PyTorch vs NumPy 代码对比
# ============================================================================
print("\n5. Code Comparison: NumPy vs PyTorch")
print("-" * 80)

comparison = """
NumPy版本 (model_train_v2.py):
    activations = self.forward(X)
    predictions = activations[-1]

    # 手写反向传播（30行代码）
    weight_gradients, bias_gradients = self.compute_gradients(activations, y)

    # 手写参数更新
    self.update_parameters(weight_gradients, bias_gradients, learning_rate)

PyTorch版本:
    predictions = self.forward(X)

    # 自动反向传播（1行！）
    loss.backward()

    # 自动参数更新（1行！）
    optimizer.step()

核心区别：
1. NumPy: 你手写了compute_gradients()中的链式法则
2. PyTorch: loss.backward()自动完成所有梯度计算
3. NumPy: 你手写了参数更新 W -= lr * grad
4. PyTorch: optimizer.step()自动完成参数更新
"""

print(comparison)

# ============================================================================
# 6. GPU加速（如果有的话）
# ============================================================================
print("\n6. GPU Acceleration")
print("-" * 80)

if torch.cuda.is_available():
    print("GPU available! Moving model to GPU...")

    device = torch.device('cuda')
    model_gpu = SimpleNet().to(device)
    X_gpu = X.to(device)
    y_gpu = y.to(device)

    print(f"Model is on: {next(model_gpu.parameters()).device}")
    print(f"Data is on: {X_gpu.device}")

    # 在GPU上训练
    predictions_gpu = model_gpu(X_gpu)
    print(f"Predictions computed on GPU: {predictions_gpu.device}")
else:
    print("No GPU available (CPU-only PyTorch installation)")
    print("\nYour system:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("  NVIDIA GPU detected:", result.stdout.strip())
            print("  But PyTorch was installed without CUDA support")
            print("\nTo enable GPU:")
            print("  pip uninstall torch")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("  No NVIDIA GPU detected")
    except:
        print("  Could not detect GPU (nvidia-smi not found)")

# ============================================================================
# 7. 总结
# ============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

summary = """
你现在理解的PyTorch核心概念：

1. Tensor = NumPy数组 + 自动求导能力
   - torch.FloatTensor() 创建tensor
   - .numpy() 转回numpy
   - requires_grad=True 启用梯度跟踪

2. nn.Module = 组织模型的标准方式
   - __init__(): 定义层
   - forward(): 定义前向传播
   - .parameters(): 自动管理所有权重

3. 自动求导 (Autograd)
   - loss.backward() 自动计算所有梯度
   - 不需要手写 compute_gradients()
   - 不需要手写链式法则

4. 优化器 (Optimizer)
   - optimizer.step() 自动更新参数
   - 不需要手写 W -= lr * grad

5. GPU加速（如果有GPU）
   - model.to('cuda') 把模型移到GPU
   - X.to('cuda') 把数据移到GPU
   - 计算自动在GPU上并行

对比你的NumPy版本：
- NumPy: 你手写了所有的反向传播和参数更新（教学价值高）
- PyTorch: 框架自动完成（工程效率高）

下一步：
1. 理解PyTorch的自动求导机制
2. 学习使用GPU加速（如果有GPU）
3. 用PyTorch实现Attention层（不再手写反向传播！）
"""

print(summary)

print("\n" + "=" * 80)
print("你已经掌握了PyTorch基础！")
print("现在可以开始学习Attention和Transformer了。")
print("=" * 80)
