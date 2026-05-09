"""
NumPy vs PyTorch 反向传播对比
展示 PyTorch 如何用 loss.backward() 替代你手写的链式法则
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 80)
print("NumPy vs PyTorch: 反向传播对比")
print("=" * 80)

# ============================================================================
# 你的 NumPy 版本：82-126 行的 compute_gradients() 方法
# ============================================================================
print("\n[NumPy 版本] 你手写的反向传播代码片段:")
print("-" * 80)

numpy_code = """
def compute_gradients(self, activations, y):
    '''反向传播：计算损失函数对所有权重和偏置的梯度'''
    m = y.shape[0]

    # 存储每层的误差梯度
    deltas = [None] * self.num_layers

    # 步骤1: 计算损失函数对输出的梯度（反向传播的起点）
    output = activations[-1]
    loss_gradient = self.compute_loss_gradient(output, y)

    # 步骤2: 输出层的梯度 = 损失梯度 × 激活函数导数（链式法则）
    deltas[-1] = loss_gradient * self.sigmoid_derivative(output)

    # 步骤3: 反向传播梯度到各隐藏层（链式法则逐层传播）
    for i in range(self.num_layers - 2, 0, -1):
        # 当前层的误差 = 后一层的梯度 × 后一层的权重转置
        error = deltas[i+1].dot(self.weights[i].T)
        deltas[i] = error * self.sigmoid_derivative(activations[i])

    # 步骤4: 计算权重和偏置的梯度
    weight_gradients = []
    bias_gradients = []

    for i in range(self.num_layers - 1):
        # 权重梯度 = 前一层激活值 × 当前层梯度
        weight_grad = activations[i].T.dot(deltas[i+1]) / m
        bias_grad = numpy.sum(deltas[i+1], axis=0, keepdims=True) / m

        weight_gradients.append(weight_grad)
        bias_gradients.append(bias_grad)

    return weight_gradients, bias_gradients

def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
    '''参数更新：使用梯度下降更新权重和偏置'''
    for i in range(self.num_layers - 1):
        # 梯度下降：沿梯度反方向更新
        self.weights[i] -= learning_rate * weight_gradients[i]
        self.biases[i] -= learning_rate * bias_gradients[i]
"""

print(numpy_code)
print(f"\n>>> 你手写了 {len(numpy_code.splitlines())} 行代码来实现反向传播和参数更新")

# ============================================================================
# PyTorch 版本：只需要 2 行
# ============================================================================
print("\n[PyTorch 版本] 等价的代码:")
print("-" * 80)

pytorch_code = """
# 反向传播：自动计算所有梯度
loss.backward()

# 参数更新：自动更新所有权重
optimizer.step()
"""

print(pytorch_code)
print(">>> PyTorch 用 2 行代码替代了你手写的 45 行链式法则！")

# ============================================================================
# 完整训练循环对比
# ============================================================================
print("\n" + "=" * 80)
print("完整训练循环对比")
print("=" * 80)

print("\n[NumPy 版本] model_train_v2.py 中的训练循环:")
print("-" * 80)

numpy_training = """
for epoch in range(epochs):
    # 步骤1: 前向传播
    activations = self.forward(X)
    predictions = activations[-1]

    # 步骤2: 计算损失
    loss = self.compute_loss(predictions, y)

    # 步骤3: 反向传播（手写链式法则，45行代码）
    weight_gradients, bias_gradients = self.compute_gradients(activations, y)

    # 步骤4: 参数更新（手写梯度下降）
    self.update_parameters(weight_gradients, bias_gradients, learning_rate)
"""

print(numpy_training)

print("\n[PyTorch 版本] 等价的训练循环:")
print("-" * 80)

pytorch_training = """
for epoch in range(epochs):
    # 步骤1: 前向传播
    predictions = model(X)

    # 步骤2: 计算损失
    loss = criterion(predictions, y)

    # 步骤3: 反向传播（自动！）
    optimizer.zero_grad()  # 清空上一轮的梯度
    loss.backward()        # 自动计算所有梯度

    # 步骤4: 参数更新（自动！）
    optimizer.step()       # 自动更新所有权重
"""

print(pytorch_training)

# ============================================================================
# 实际运行对比：解决同样的 XOR 问题
# ============================================================================
print("\n" + "=" * 80)
print("实际运行对比：XOR 问题")
print("=" * 80)

# 准备数据
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

# 定义模型（对应你的 [2, 4, 3, 1] 结构）
class PyTorchNN(nn.Module):
    def __init__(self):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

torch.manual_seed(42)
model = PyTorchNN()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

print("\n[PyTorch 训练中...]")
print("使用完全相同的网络结构: [2, 4, 3, 1]")
print()

for epoch in range(10001):
    # === 你手写的 3 个步骤在 PyTorch 中只需要 4 行 ===

    # 1. 前向传播
    predictions = model(X)

    # 2. 计算损失
    loss = criterion(predictions, y)

    # 3. 反向传播（替代你的 compute_gradients）
    optimizer.zero_grad()
    loss.backward()

    # 4. 参数更新（替代你的 update_parameters）
    optimizer.step()

    if epoch % 2000 == 0 or epoch == 10000:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("\n[训练完成] 最终预测:")
with torch.no_grad():
    final_predictions = model(X)
    for i in range(len(X)):
        print(f"  输入: {X[i].tolist()}  ->  预测: {final_predictions[i].item():.4f}  (期望: {y[i].item()})")

# ============================================================================
# 幕后解析：loss.backward() 做了什么
# ============================================================================
print("\n" + "=" * 80)
print("幕后解析：loss.backward() 做了什么？")
print("=" * 80)

explanation = """
你手写的 compute_gradients() 方法：
  1. 计算输出层梯度：loss_gradient * sigmoid_derivative(output)
  2. 逐层反向传播：error = deltas[i+1].dot(weights[i].T)
  3. 计算每层权重梯度：activations[i].T.dot(deltas[i+1])
  4. 返回所有层的梯度

PyTorch 的 loss.backward() 做了完全相同的事情：
  1. 自动追踪计算图（每次运算都记录依赖关系）
  2. 从 loss 开始，沿着计算图反向传播
  3. 自动应用链式法则到每个运算
  4. 把梯度存储到每个参数的 .grad 属性中

差异：
  NumPy: 你需要手写链式法则的每一步
  PyTorch: 框架自动追踪并应用链式法则

结果完全一样，但 PyTorch 帮你省了大量工作！
"""

print(explanation)

# ============================================================================
# 验证梯度确实被计算了
# ============================================================================
print("\n" + "=" * 80)
print("验证：梯度确实被计算并存储了")
print("=" * 80)

# 重新运行一次前向和反向传播
predictions = model(X)
loss = criterion(predictions, y)
optimizer.zero_grad()
loss.backward()

print("\n查看第一层权重的梯度:")
print(f"  fc1.weight.grad.shape: {model.fc1.weight.grad.shape}")
print(f"  fc1.weight.grad:\n{model.fc1.weight.grad}")
print("\n>>> 这些梯度就是 loss.backward() 自动计算出来的！")
print(">>> 等价于你手写的 compute_gradients() 的返回值")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

summary = """
问：在 PyTorch 中只用 loss.backward() 就行了吗？

答：是的！但训练循环完整步骤是：

    optimizer.zero_grad()  # 1. 清空旧梯度
    loss.backward()        # 2. 自动计算新梯度（替代你的 compute_gradients）
    optimizer.step()       # 3. 自动更新参数（替代你的 update_parameters）

对比你的 NumPy 版本：
  ┌────────────────────────────────────────────────────────────────┐
  │ NumPy (v2.py)          │ PyTorch 等价代码                      │
  ├────────────────────────────────────────────────────────────────┤
  │ forward(X)             │ model(X)                              │
  │ compute_loss()         │ criterion(predictions, y)             │
  │ compute_gradients()    │ optimizer.zero_grad() + loss.backward()│
  │   (45 行链式法则)      │   (2 行自动求导)                      │
  │ update_parameters()    │ optimizer.step()                      │
  │   (手写梯度下降)       │   (自动更新)                          │
  └────────────────────────────────────────────────────────────────┘

你手写反向传播的价值：
  ✓ 深刻理解了链式法则
  ✓ 知道梯度是怎么计算的
  ✓ 现在能看懂 PyTorch 在做什么

现在用 PyTorch 的好处：
  ✓ 不需要手写链式法则
  ✓ 支持任意复杂的网络（Attention、Transformer 等）
  ✓ 自动 GPU 加速
  ✓ 更少的 bug（手写反向传播容易出错）

下一步：
  用 PyTorch 实现 Attention 机制，体验"不用手写反向传播"的爽感
"""


print(summary)

# ============================================================================
# 补充：为什么 PyTorch 需要 zero_grad()，但 NumPy 不需要？
# ============================================================================
print("\n" + "=" * 80)
print("补充：为什么需要 optimizer.zero_grad()？")
print("=" * 80)

print("\n[关键区别]")
print("-" * 80)
print("""
NumPy 版本（你的 compute_gradients）:
    def compute_gradients(self, activations, y):
        weight_gradients = []  # <- 每次都创建新列表
        bias_gradients = []    # <- 旧梯度自动丢弃

        # ... 计算梯度 ...

        return weight_gradients, bias_gradients

PyTorch 版本:
    loss.backward()  # <- 梯度累加到 param.grad 上

    # param.grad 不会自动清空，需要手动清零
    optimizer.zero_grad()
""")

print("\n[实际演示] 不清空梯度会怎样？")
print("-" * 80)

# 简单模型
class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        self.fc.weight.data = torch.tensor([[1.0, 1.0]])

    def forward(self, x):
        return self.fc(x)

demo_model = TinyModel()
X_demo = torch.tensor([[1.0, 2.0]])
y_demo = torch.tensor([[5.0]])

print("\n场景1: 正确做法 - 每次清空梯度")
demo_model.zero_grad()
for i in range(3):
    pred = demo_model(X_demo)
    loss = (pred - y_demo) ** 2

    demo_model.zero_grad()  # 清空旧梯度
    loss.backward()

    print(f"第 {i+1} 轮梯度: {demo_model.fc.weight.grad.data.numpy()}")

print("\n场景2: 错误做法 - 不清空梯度")
demo_model.zero_grad()
for i in range(3):
    pred = demo_model(X_demo)
    loss = (pred - y_demo) ** 2

    # demo_model.zero_grad()  # <- 注释掉了
    loss.backward()

    print(f"第 {i+1} 轮梯度: {demo_model.fc.weight.grad.data.numpy()} <- 越来越大！")

print("\n>>> NumPy 每次创建新列表，所以不需要清空")
print(">>> PyTorch 梯度累加在 param.grad 上，必须手动清空")

print("\n[为什么设计成累加？]")
print("-" * 80)
zero_grad_reason = """
原因：支持梯度累积（Gradient Accumulation）

当 batch 太大，显存不够时：
    for mini_batch in large_batch:
        loss = compute_loss(mini_batch)
        loss.backward()  # 梯度累加
    optimizer.step()     # 一次更新所有累积的梯度
    optimizer.zero_grad()

标准训练循环（牢记）：
    optimizer.zero_grad()  # 1. 清空旧梯度
    loss.backward()        # 2. 计算新梯度（会累加）
    optimizer.step()       # 3. 更新参数
"""
print(zero_grad_reason)

print("\n" + "=" * 80)
print("补充2：autograd 计算图（不需要深入，知道概念就行）")
print("=" * 80)

print("\n[什么是计算图？]")
print("-" * 80)
print("""
计算图 = PyTorch 自动记录的运算流程图

例子：
    x = torch.tensor([2.0], requires_grad=True)
    y = x * 3        # 运算1
    z = y + 5        # 运算2
    loss = z ** 2    # 运算3

PyTorch 自动记录：
    x → [×3] → y → [+5] → z → [^2] → loss

当你调用 loss.backward()，PyTorch 反向走一遍，用链式法则算梯度
""")

print("\n[实际演示] 看看计算图怎么工作")
print("-" * 80)

x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y + 5
loss = z ** 2

print(f"x = {x.item()}")
print(f"y = x * 3 = {y.item()}")
print(f"z = y + 5 = {z.item()}")
print(f"loss = z^2 = {loss.item()}")

# 反向传播
loss.backward()

print(f"\nloss 对 x 的梯度: {x.grad.item()}")
print("\n手动验证:")
print("  dloss/dx = dloss/dz × dz/dy × dy/dx")
print(f"           = 2z × 1 × 3")
print(f"           = 2×{z.item()}×3 = {2*z.item()*3}")
print(f"  PyTorch 计算的: {x.grad.item()}")
print("  >>> 完全一致！")

print("\n[你只需要知道这些]")
print("-" * 80)
autograd_summary = """
1. requires_grad=True 告诉 PyTorch "追踪这个变量"
2. backward() 沿计算图反向传播，自动算梯度
3. with torch.no_grad(): 推理时不建计算图（省内存）

不需要懂：
  - autograd 引擎的 C++ 实现
  - 计算图的数据结构
  - 梯度如何存储

就像开车不需要懂发动机原理一样
"""
print(autograd_summary)

print("\n" + "=" * 80)
print("运行完成！你已经理解了 PyTorch 的自动求导、zero_grad() 和计算图")
print("=" * 80)
