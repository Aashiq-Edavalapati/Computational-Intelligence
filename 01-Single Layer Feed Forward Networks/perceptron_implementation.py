import numpy as np

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=1000):
        """
        Initializes the Perceptron.
        
        Parameters:
        - input_dim (int): Number of features.
        - learning_rate (float): Step size for weight updates.
        - n_epochs (int): Number of training iterations over the dataset.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = np.zeros(input_dim)
        self.bias = 0

    def activation(self, z):
        """
        Step activation function (binary output: 0 or 1).
        """
        return 1 if z >= 0 else 0

    def predict(self, X):
        """
        Predicts the class label (0 or 1) for input X.
        
        Parameters:
        - X (np.ndarray): Input vector or matrix.
        
        Returns:
        - int or np.ndarray: Predicted class label(s).
        """
        z = np.dot(X, self.weights) + self.bias
        if len(z.shape) == 0:
            return self.activation(z)
        return np.array([self.activation(x) for x in z])

    def fit(self, X_train, y_train):
        """
        Trains the Perceptron on given data using the Perceptron learning rule.
        
        Parameters:
        - X_train (np.ndarray): Input features, shape (n_samples, n_features).
        - y_train (np.ndarray): True labels, shape (n_samples,).
        """
        for epoch in range(self.n_epochs):
            for xi, target in zip(X_train, y_train):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(z)
                error = target - y_pred
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

    def evaluate(self, X_test, y_test):
        """
        Evaluates accuracy on the test set.
        
        Returns:
        - float: Accuracy (0.0 - 1.0)
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)

# --- Example Usage ---

if __name__ == "__main__":
    # Simple AND gate example
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])  # AND output

    perceptron = Perceptron(input_dim=2, learning_rate=0.1, n_epochs=10)
    perceptron.fit(X, y)

    print("Final Weights:", perceptron.weights)
    print("Final Bias:", perceptron.bias)

    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", perceptron.evaluate(X, y))
