import torch
import torch.nn as nn
import numpy
import time

"""
PyTorch版神经网络 - 对应 model_train_v2.py

核心区别：
1. 自动求导：不需要手写反向传播
2. GPU加速：tensor可以放在GPU上并行计算
3. 内置优化器：不需要手写参数更新
4. 模块化：用nn.Module组织代码
"""


class NeuralNetworkPyTorch(nn.Module):
    """
    PyTorch版本的神经网络

    和NumPy版本的对应关系：
    - NumPy的weights列表 → PyTorch的nn.Linear层
    - NumPy的forward → PyTorch的forward
    - NumPy的compute_gradients → PyTorch的loss.backward()（自动！）
    - NumPy的update_parameters → PyTorch的optimizer.step()
    """

    def __init__(self, layer_sizes):
        """
        初始化神经网络

        参数:
            layer_sizes: 列表，定义每层的节点数
                       例如 [2, 4, 3, 1]
        """
        super(NeuralNetworkPyTorch, self).__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # 创建所有层
        # nn.Linear(in_features, out_features) 自动创建权重W和偏置b
        self.layers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            # nn.Linear内部会自动用Xavier初始化
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)

        # 激活函数
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入tensor, shape (batch_size, input_dim)

        返回:
            输出tensor, shape (batch_size, output_dim)
        """
        # 逐层计算
        for i in range(len(self.layers)):
            x = self.layers[i](x)  # 线性变换 W·x + b
            x = self.activation(x)  # 激活函数

        return x

    def train_model(self, X, y, epochs=5000, learning_rate=0.5, device='cpu', verbose=True):
        """
        训练模型

        参数:
            X: 训练数据 (numpy数组或tensor)
            y: 训练标签 (numpy数组或tensor)
            epochs: 训练轮数
            learning_rate: 学习率
            device: 'cpu' 或 'cuda'
            verbose: 是否打印训练过程

        返回:
            losses: 损失历史
            training_time: 训练耗时（秒）
        """
        # 转换为tensor并移到指定设备
        if isinstance(X, numpy.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, numpy.ndarray):
            y = torch.FloatTensor(y)

        X = X.to(device)
        y = y.to(device)

        # 把模型移到指定设备
        self.to(device)

        # 定义损失函数（MSE）
        criterion = nn.MSELoss()

        # 定义优化器（SGD梯度下降）
        # 这个优化器会自动更新self.parameters()中的所有参数
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        losses = []
        start_time = time.time()

        for epoch in range(epochs):
            # ===== 步骤1: 前向传播 =====
            predictions = self.forward(X)

            # ===== 步骤2: 计算损失 =====
            loss = criterion(predictions, y)

            # ===== 步骤3: 反向传播（自动计算梯度！）=====
            optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()        # 自动计算所有参数的梯度！

            # ===== 步骤4: 参数更新（优化器自动完成）=====
            optimizer.step()       # 根据梯度更新参数

            # 记录和打印
            if epoch % 1000 == 0 or epoch == epochs - 1:
                loss_value = loss.item()
                losses.append(loss_value)
                if verbose:
                    print(f"Epoch {epoch}, Loss: {loss_value:.6f}")

        training_time = time.time() - start_time

        return losses, training_time

    def predict(self, X, device='cpu'):
        """
        预测

        参数:
            X: 输入数据
            device: 设备

        返回:
            预测结果（numpy数组）
        """
        if isinstance(X, numpy.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(device)
        self.to(device)

        with torch.no_grad():  # 预测时不需要计算梯度
            predictions = self.forward(X)

        return predictions.cpu().numpy()  # 转回numpy数组


def compare_numpy_vs_pytorch():
    """
    对比NumPy版本和PyTorch版本
    """
    print("=" * 80)
    print("NumPy vs PyTorch Comparison")
    print("=" * 80)

    # XOR训练数据
    X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = numpy.array([[0], [1], [1], [0]])

    # 网络结构
    layer_sizes = [2, 4, 3, 1]
    epochs = 5000
    learning_rate = 0.5

    print(f"\nNetwork structure: {layer_sizes}")
    print(f"Training epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

    # ===== PyTorch CPU版本 =====
    print("\n" + "=" * 80)
    print("PyTorch (CPU)")
    print("=" * 80)

    nn_pytorch_cpu = NeuralNetworkPyTorch(layer_sizes)
    losses_cpu, time_cpu = nn_pytorch_cpu.train_model(
        X, y, epochs=epochs, learning_rate=learning_rate,
        device='cpu', verbose=False
    )

    print(f"\nTraining time: {time_cpu:.4f} seconds")
    print(f"Final loss: {losses_cpu[-1]:.6f}")

    print("\nPredictions:")
    predictions_cpu = nn_pytorch_cpu.predict(X, device='cpu')
    for i in range(len(X)):
        print(f"  Input {X[i]} -> Prediction {predictions_cpu[i][0]:.4f} (Expected {y[i][0]})")

    # ===== PyTorch GPU版本（如果有GPU）=====
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("PyTorch (GPU)")
        print("=" * 80)

        nn_pytorch_gpu = NeuralNetworkPyTorch(layer_sizes)
        losses_gpu, time_gpu = nn_pytorch_gpu.train_model(
            X, y, epochs=epochs, learning_rate=learning_rate,
            device='cuda', verbose=False
        )

        print(f"\nTraining time: {time_gpu:.4f} seconds")
        print(f"Final loss: {losses_gpu[-1]:.6f}")
        print(f"Speedup: {time_cpu / time_gpu:.2f}x")

        print("\nPredictions:")
        predictions_gpu = nn_pytorch_gpu.predict(X, device='cuda')
        for i in range(len(X)):
            print(f"  Input {X[i]} -> Prediction {predictions_gpu[i][0]:.4f} (Expected {y[i][0]})")
    else:
        print("\n" + "=" * 80)
        print("No GPU available (CUDA not found)")
        print("=" * 80)
        print("\nTo use GPU:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("2. Make sure you have NVIDIA GPU with CUDA support")

    # ===== 对比NumPy版本（如果可用）=====
    try:
        from model_train_v2 import NeuralNetwork as NeuralNetworkNumPy

        print("\n" + "=" * 80)
        print("NumPy (CPU)")
        print("=" * 80)

        nn_numpy = NeuralNetworkNumPy(layer_sizes)
        start_time = time.time()
        losses_numpy = nn_numpy.train(X, y, epochs=epochs, learning_rate=learning_rate, verbose=False)
        time_numpy = time.time() - start_time

        print(f"\nTraining time: {time_numpy:.4f} seconds")
        print(f"Final loss: {losses_numpy[-1]:.6f}")

        print("\nPredictions:")
        predictions_numpy = nn_numpy.predict(X)
        for i in range(len(X)):
            print(f"  Input {X[i]} -> Prediction {predictions_numpy[i][0]:.4f} (Expected {y[i][0]})")

        # 速度对比
        print("\n" + "=" * 80)
        print("Speed Comparison")
        print("=" * 80)
        print(f"NumPy (CPU):   {time_numpy:.4f}s")
        print(f"PyTorch (CPU): {time_cpu:.4f}s")
        if torch.cuda.is_available():
            print(f"PyTorch (GPU): {time_gpu:.4f}s")
            print(f"\nGPU is {time_cpu / time_gpu:.2f}x faster than PyTorch CPU")
            print(f"GPU is {time_numpy / time_gpu:.2f}x faster than NumPy CPU")

    except ImportError:
        print("\n(model_train_v2.py not found, skipping NumPy comparison)")


def check_gpu_info():
    """
    检查GPU信息
    """
    print("=" * 80)
    print("GPU Information")
    print("=" * 80)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

            # 显存使用情况
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("\nNo CUDA-capable GPU found.")
        print("\nTo enable GPU:")
        print("1. Check if you have NVIDIA GPU: run 'nvidia-smi' in terminal")
        print("2. Install CUDA toolkit")
        print("3. Install PyTorch with CUDA support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")


def demonstrate_autograd():
    """
    演示PyTorch的自动求导
    """
    print("=" * 80)
    print("PyTorch Autograd (Automatic Differentiation)")
    print("=" * 80)

    print("\n--- Example 1: Simple computation ---")

    # 创建需要梯度的tensor
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)

    # 前向计算
    y = w * x + b  # y = 3*2 + 1 = 7
    z = y ** 2     # z = 7^2 = 49

    print(f"x = {x.item()}")
    print(f"w = {w.item()}")
    print(f"b = {b.item()}")
    print(f"y = w*x + b = {y.item()}")
    print(f"z = y^2 = {z.item()}")

    # 反向传播（自动计算梯度）
    z.backward()

    print("\nAfter z.backward():")
    print(f"dz/dx = {x.grad.item():.1f}  (Expected: 2*y*w = 2*7*3 = 42)")
    print(f"dz/dw = {w.grad.item():.1f}  (Expected: 2*y*x = 2*7*2 = 28)")
    print(f"dz/db = {b.grad.item():.1f}  (Expected: 2*y*1 = 2*7*1 = 14)")

    print("\n--- Example 2: Matrix multiplication ---")

    X = torch.tensor([[1.0, 2.0]], requires_grad=True)  # (1, 2)
    W = torch.tensor([[0.5], [0.3]], requires_grad=True)  # (2, 1)

    y = X @ W  # 矩阵乘法
    loss = (y - 1.0) ** 2

    print(f"X = {X.tolist()}")
    print(f"W = {W.tolist()}")
    print(f"y = X @ W = {y.item():.2f}")
    print(f"loss = (y - 1)^2 = {loss.item():.4f}")

    loss.backward()

    print("\nGradients:")
    print(f"dLoss/dX = {X.grad.tolist()}")
    print(f"dLoss/dW = {W.grad.tolist()}")

    print("\nKey insight: PyTorch automatically computes all gradients!")
    print("No need to manually implement backpropagation with chain rule.")


if __name__ == "__main__":
    # 1. 检查GPU信息
    check_gpu_info()

    # 2. 演示自动求导
    print("\n\n")
    demonstrate_autograd()

    # 3. 对比NumPy和PyTorch
    print("\n\n")
    compare_numpy_vs_pytorch()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Key takeaways:
1. PyTorch autograd: No need to manually write backpropagation code
2. GPU acceleration: Move tensors to GPU with .to('cuda')
3. nn.Module: Organize model architecture cleanly
4. Optimizer: Automatic parameter updates

Next steps:
- Practice moving data between CPU and GPU
- Try larger networks and datasets to see GPU advantage
- Learn about different optimizers (Adam, RMSprop)
- Start building Attention mechanism with PyTorch
    """)
