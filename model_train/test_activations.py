import numpy
from model_train_v2 import NeuralNetwork

print("=" * 70)
print("Neural Network Activation Visualization Test")
print("=" * 70)

# XOR训练数据
X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = numpy.array([[0], [1], [1], [0]])

print("\nTraining XOR problem...")
print("Network structure: [2, 4, 3, 1]")
print("  - Input: 2 dimensions")
print("  - Hidden Layer 1: 4 neurons")
print("  - Hidden Layer 2: 3 neurons")
print("  - Output: 1 dimension")

# 创建并训练神经网络
nn = NeuralNetwork([2, 4, 3, 1])
nn.train(X, y, epochs=5000, learning_rate=0.5, verbose=False)

print("\nTraining complete!\n")

# 可视化每个样本的激活值
for i in range(len(X)):
    print(f"\n{'='*70}")
    print(f"Sample {i}: Input {X[i]} -> Expected Output {y[i][0]}")
    print(f"{'='*70}")
    nn.visualize_activations(X, sample_index=i)

    prediction = nn.predict(X[i:i+1])[0][0]
    print(f"\nFinal Prediction: {prediction:.4f} (Expected: {y[i][0]})")
    print()

# 测试一个中间值
print("\n" + "=" * 70)
print("Testing with intermediate values")
print("=" * 70)

X_test = numpy.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])

for i in range(len(X_test)):
    print(f"\n{'='*70}")
    print(f"Test Sample {i}: Input {X_test[i]}")
    print(f"{'='*70}")
    nn.visualize_activations(X_test, sample_index=i)

    prediction = nn.predict(X_test[i:i+1])[0][0]
    print(f"\nPrediction: {prediction:.4f}")
    print()

print("\n" + "=" * 70)
print("Observations:")
print("=" * 70)
print("""
1. Input layer: Original 2D input values
2. Hidden layers: Values transformed through weights + bias + sigmoid
3. Output layer: Final prediction (close to 0 or 1 for XOR)

Key insights:
- Each layer transforms the input vector dimension
- Sigmoid activation keeps values in [0, 1] range
- Network learns non-linear decision boundary through multiple layers
""")
