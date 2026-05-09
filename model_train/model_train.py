import numpy

#我想实现一个简单的深度学习神经网络模型
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # 初始化权重（使用更好的初始化方式：Xavier初始化）
        # 每一层神经元的权重维度等于输入维度 输入经过每一层神经网络计算后的输出维度等于该层神经元的数量
        self.W1 = numpy.random.randn(input_size, hidden_size1) * 0.5
        self.W2 = numpy.random.randn(hidden_size1, hidden_size2) * 0.5
        self.W3 = numpy.random.randn(hidden_size2, output_size) * 0.5

        # 偏置项
        self.b1 = numpy.zeros((1, hidden_size1))
        self.b2 = numpy.zeros((1, hidden_size2))
        self.b3 = numpy.zeros((1, output_size))

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    # 激活函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 训练模型
    def train(self, X, y, epochs, learning_rate=0.1):
        for epoch in range(epochs):
            # 前向传播（加入偏置项）
            hidden1_input = numpy.dot(X, self.W1) + self.b1
            hidden1_output = self.sigmoid(hidden1_input)

            hidden2_input = numpy.dot(hidden1_output, self.W2) + self.b2
            hidden2_output = self.sigmoid(hidden2_input)

            final_input = numpy.dot(hidden2_output, self.W3) + self.b3
            final_output = self.sigmoid(final_input)

            # 计算梯度（损失函数对输出的梯度）
            output_gradient = final_output - y

            # 反向传播（链式法则计算各层梯度）
            d_final_output = output_gradient * self.sigmoid_derivative(final_output)

            hidden2_gradient = d_final_output.dot(self.W3.T)
            d_hidden2_output = hidden2_gradient * self.sigmoid_derivative(hidden2_output)

            hidden1_gradient = d_hidden2_output.dot(self.W2.T)
            d_hidden1_output = hidden1_gradient * self.sigmoid_derivative(hidden1_output)

            # 更新权重（梯度下降：沿梯度反方向）
            self.W3 -= learning_rate * hidden2_output.T.dot(d_final_output)
            self.W2 -= learning_rate * hidden1_output.T.dot(d_hidden2_output)
            self.W1 -= learning_rate * X.T.dot(d_hidden1_output)

            # 更新偏置
            self.b3 -= learning_rate * numpy.sum(d_final_output, axis=0, keepdims=True)
            self.b2 -= learning_rate * numpy.sum(d_hidden2_output, axis=0, keepdims=True)
            self.b1 -= learning_rate * numpy.sum(d_hidden1_output, axis=0, keepdims=True)

            # 每1000轮打印一次损失
            if epoch % 1000 == 0:
                loss = numpy.mean(numpy.square(y - final_output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        hidden1_input = numpy.dot(X, self.W1) + self.b1
        hidden1_output = self.sigmoid(hidden1_input)

        hidden2_input = numpy.dot(hidden1_output, self.W2) + self.b2
        hidden2_output = self.sigmoid(hidden2_input)

        final_input = numpy.dot(hidden2_output, self.W3) + self.b3
        final_output = self.sigmoid(final_input)

        return final_output

if __name__ == "__main__":
    # 训练数据
    X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = numpy.array([[0], [1], [1], [0]])

    # 创建神经网络实例（增加了第二个隐藏层）
    # 输入维度=2, 隐藏层1有4个神经元, 隐藏层2有3个神经元, 输出维度=1
    nn = NeuralNetwork(input_size=2, hidden_size1=4, hidden_size2=3, output_size=1)

    # 训练模型（学习率=0.5，训练10000轮）
    nn.train(X, y, epochs=10000, learning_rate=0.5)

    # 测试模型
    print("训练集测试:")
    print(nn.predict(X))

    # 泛化能力测试（接近但不等于训练数据的值）
    print("\n泛化测试（未见过的输入）:")
    X_test = numpy.array([
        [0.1, 0.1],   # 接近 [0,0] 应该输出 ~0
        [0.05, 0.95], # 接近 [0,1] 应该输出 ~1
        [0.9, 0.1],   # 接近 [1,0] 应该输出 ~1
        [0.95, 0.9]   # 接近 [1,1] 应该输出 ~0
    ])
    predictions = nn.predict(X_test)
    print("输入:")
    print(X_test)
    print("预测输出:")
    print(predictions)            