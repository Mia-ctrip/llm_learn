import numpy

# 改进版：支持任意层数的神经网络
#
# 核心概念：
# 1. 前向传播 (Forward Propagation): 计算预测值
# 2. 反向传播 (Backpropagation): 计算梯度（链式法则）
# 3. 参数更新 (Parameter Update): 用梯度更新权重（梯度下降）
#
# 这三个步骤是独立的 反向传播只负责计算梯度，不负责更新权重。

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        初始化神经网络

        参数:
            layer_sizes: 列表，定义每层的节点数
                       例如 [2, 4, 3, 1] 表示：
                       - 输入层：输入是2维
                       - 隐藏层1：4个节点
                       - 隐藏层2：3个节点
                       - 输出层：1个节点
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # 初始化权重和偏置
        # weights[i] 连接第i层和第i+1层
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier初始化：权重在 [-sqrt(1/n), sqrt(1/n)] 范围内
            # 权重W是一个矩阵 矩阵的每一列是一个神经元的权重向量 矩阵的shape是 (当前层节点数, 下一层节点数) 矩阵乘法实现了输入同时经过所有神经元的计算
            w = numpy.random.randn(layer_sizes[i], layer_sizes[i+1]) * numpy.sqrt(1.0 / layer_sizes[i])
            b = numpy.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-numpy.clip(x, -500, 500)))  # 防止溢出

    # 激活函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """
        前向传播

        返回:
            activations: 列表，存储每层的激活值（包括输入层）
        """
        activations = [X]  # 第0层是输入

        # 逐层计算
        for i in range(self.num_layers - 1):
            z = numpy.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

        return activations

    def compute_loss(self, predictions, y):
        """
        计算损失函数（均方误差 MSE）

        L = 0.5 * mean((y - predictions)²)
        """
        return 0.5 * numpy.mean(numpy.square(y - predictions))

    def compute_loss_gradient(self, predictions, y):
        """
        计算损失函数对输出的梯度

        ∂L/∂output = ∂/∂output[0.5 * (y - output)²] = output - y
        """
        return predictions - y

    def compute_gradients(self, activations, y):
        """
        反向传播：计算损失函数对所有权重和偏置的梯度

        这是真正的"反向传播"步骤，只计算梯度，不更新参数

        参数:
            activations: 前向传播得到的每层激活值
            y: 真实标签

        返回:
            weight_gradients: 所有层权重的梯度列表
            bias_gradients: 所有层偏置的梯度列表
        """
        m = y.shape[0]  # 样本数量

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
        """
        参数更新：使用梯度下降更新权重和偏置

        这是独立于反向传播的步骤

        参数:
            weight_gradients: 权重梯度列表
            bias_gradients: 偏置梯度列表
            learning_rate: 学习率
        """
        for i in range(self.num_layers - 1):
            # 梯度下降：沿梯度反方向更新
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]

    def train(self, X, y, epochs, learning_rate=0.1, verbose=True):
        """
        训练模型

        完整的训练流程分为三个独立步骤：
        1. 前向传播：计算预测值
        2. 反向传播：计算梯度
        3. 参数更新：用梯度更新权重

        参数:
            X: 训练数据
            y: 训练标签
            epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印训练过程
        """
        losses = []

        for epoch in range(epochs):
            # ===== 步骤1: 前向传播 =====
            activations = self.forward(X)
            predictions = activations[-1]

            # ===== 步骤2: 计算损失函数 =====
            loss = self.compute_loss(predictions, y)

            # ===== 步骤3: 反向传播（计算梯度）=====
            weight_gradients, bias_gradients = self.compute_gradients(activations, y)

            # ===== 步骤4: 参数更新（梯度下降）=====
            self.update_parameters(weight_gradients, bias_gradients, learning_rate)

            # 记录和打印
            if epoch % 1000 == 0 or epoch == epochs - 1:
                losses.append(loss)
                if verbose:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return losses

    def predict(self, X):
        """
        预测
        """
        activations = self.forward(X)
        return activations[-1]

    def visualize_activations(self, X, sample_index=0):
        """
        可视化每层的激活值

        参数:
            X: 输入数据
            sample_index: 要查看的样本索引（默认第0个样本）
        """
        activations = self.forward(X)

        print("=" * 70)
        print(f"Activation Values Visualization (Sample #{sample_index})")
        print("=" * 70)

        # 输入层
        input_vec = activations[0][sample_index]
        print(f"\nInput Layer (dim={len(input_vec)}):")
        print(f"  {input_vec}")

        # 每个隐藏层和输出层
        for i in range(1, len(activations)):
            activation_vec = activations[i][sample_index]
            layer_name = f"Hidden Layer {i}" if i < len(activations) - 1 else "Output Layer"

            print(f"\n{layer_name} (dim={len(activation_vec)}):")
            print(f"  {activation_vec}")

            # 显示权重和偏置的形状
            print(f"  [via W{i} {self.weights[i-1].shape} + b{i} {self.biases[i-1].shape}]")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # XOR训练数据
    X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = numpy.array([[0], [1], [1], [0]])

    print("=" * 50)
    print("测试1: 原来的3层网络 [2, 4, 3, 1]")
    print("=" * 50)
    nn1 = NeuralNetwork([2, 4, 3, 1])
    nn1.train(X, y, epochs=10000, learning_rate=0.5)

    print("\n训练集测试:")
    print(nn1.predict(X))

    # 泛化测试
    X_test = numpy.array([[0.1, 0.1], [0.05, 0.95], [0.9, 0.1], [0.95, 0.9]])
    print("\n泛化测试:")
    print(nn1.predict(X_test))

    # 测试不同的网络结构
    print("\n" + "=" * 50)
    print("测试2: 更深的网络 [2, 8, 6, 4, 1]")
    print("=" * 50)
    nn2 = NeuralNetwork([2, 8, 6, 4, 1])
    nn2.train(X, y, epochs=10000, learning_rate=0.5)

    print("\n训练集测试:")
    print(nn2.predict(X))

    # 测试浅层网络
    print("\n" + "=" * 50)
    print("测试3: 浅层网络 [2, 4, 1]")
    print("=" * 50)
    nn3 = NeuralNetwork([2, 4, 1])
    nn3.train(X, y, epochs=10000, learning_rate=0.5)

    print("\n训练集测试:")
    print(nn3.predict(X))

    print("\n" + "=" * 50)
    print("总结: 支持任意层数的神经网络实现成功！")
    print("=" * 50)
