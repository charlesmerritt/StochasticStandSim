import numpy as np

# Initialize inputs, weights, and biases
np.random.seed(0)
X = np.array([[0.5, 0.2]])  # Input
y_true = np.array([[1]])    # True output

# Parameters
W1 = np.random.rand(2, 3)   # Weights for the first layer
b1 = np.random.rand(1, 3)   # Biases for the first layer
W2 = np.random.rand(3, 1)   # Weights for the second layer
b2 = np.random.rand(1, 1)   # Biases for the second layer

# Learning rate
alpha = 0.01

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Forward pass
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # First layer linear
    A1 = sigmoid(Z1)         # First layer activation
    Z2 = np.dot(A1, W2) + b2 # Second layer linear
    A2 = sigmoid(Z2)         # Second layer activation (output)
    return Z1, A1, Z2, A2

# Loss function (Mean Squared Error)
def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Backpropagation
for epoch in range(10000):
    # Forward pass
    Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
    loss_value = loss(y_true, A2)

    # Backward pass
    dA2 = A2 - y_true                       # Derivative of loss w.r.t output
    dZ2 = dA2 * sigmoid_derivative(Z2)     # Backprop through activation
    dW2 = np.dot(A1.T, dZ2)                # Gradient for W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradient for b2

    dA1 = np.dot(dZ2, W2.T)                # Backprop through layer 2
    dZ1 = dA1 * sigmoid_derivative(Z1)     # Backprop through activation
    dW1 = np.dot(X.T, dZ1)                 # Gradient for W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradient for b1

    # Update parameters
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

    # Print loss every 100 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

# Final loss
print(f"Final Loss: {loss_value:.4f}")
