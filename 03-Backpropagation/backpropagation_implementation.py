import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset (non-linearly separable)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)  # Reshape for compatibility

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hyperparameters
lr = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    # ----- Forward Pass -----
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # ----- Loss Calculation (Binary Cross-Entropy) -----
    m = y_train.shape[0]
    loss = -np.mean(y_train * np.log(a2 + 1e-9) + (1 - y_train) * np.log(1 - a2 + 1e-9))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # ----- Backpropagation -----
    dz2 = a2 - y_train                      # Output layer error
    dW2 = np.dot(a1.T, dz2) / m            # Gradient for W2
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(z1)  # Hidden layer error
    dW1 = np.dot(X_train.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # ----- Update Weights -----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# ----- Evaluation -----
z1_test = np.dot(X_test, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = sigmoid(z2_test)

predictions = (a2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
