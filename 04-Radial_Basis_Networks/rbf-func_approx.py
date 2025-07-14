import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# --- 1. Define the Target Function ---
def target_function(x):
    """
        A non-linear function to approximate.
    """
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))

# ===============================================================

# --- 2. RBF Network Class Implementation ---
class RBFNN:
    def __init__(self, n_hidden_neurons, sigma = None):
        """
            Initializes the RBF Neural Network.

            Args:
                n_hidden_neurons (int): Number of RBF(hidden) neurons.
                sigma (float, optional): Spread parameter for Gaussian RBF.
                                If None, it's estimated during training.
        """

        self.n_hidden_neurons = n_hidden_neurons
        self.sigma = sigma
        self.centers = None # Centers of RBF neurons
        self.weights = None # Weights connecting hidden to output layer


    def _calculate_distance(self, x, center):
        """
            Calculates Euclidean distance between input x and a center.
        """
        return np.linalg.norm(x - center)

    def _gaussian_rbf(self, r):
        """
            Gaussian RBF.
        """
        if self.sigma is None:
            raise ValueError("Sigma must be set before calculating RBF.")
        return np.exp(-(r ** 2)/(2 * self.sigma ** 2))
    
    def _calculate_hidden_activations(self, X):
        """
            Calculates the activation of each hidden neuron for given input(s).

            Args:
                X(np.ndarray): Input data (n_samples, n_features).

            Returns:
                np.ndarray: Hidden layer activations (n_samples, n_hidden_neurons).
        """
        if self.centers is None:
            raise ValueError("Centers must be set before calculating activations.")
        
        num_samples = X.shape[0]
        H = np.zeros((num_samples, self.n_hidden_neurons))

        for i in range(num_samples):
            for j in range(self.n_hidden_neurons):
                distance = self._calculate_distance(X[i], self.centers[j])
                H[i, j] = self._gaussian_rbf(distance)
        
        return H
    
    def fit(self, X, y):
        """
            Trains the RBF NN.

            Args:
                X (np.ndarray): Training input data (n_samples, n_features).
                y (np.ndarray): Target output data (n_samples).
        """
        # Step 1: Determine Centers using K-Means
        kmeans = KMeans(n_clusters=self.n_hidden_neurons, random_state=42, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Step 2: Determine Spreads (Sigma)
        if self.sigma is None:
            # Heuristic: Average distance between centers, or max distance.
            # Let's use max distance between any two centers, divided by sqrt(2 * num_centers).
            # This is a common heuristic, but other methods exist.
            max_distance = 0
            for i in range(self.n_hidden_neurons):
                for j in range(i + 1, self.n_hidden_neurons):
                    dist = np.linalg.norm(self.centers[i] - self.centers[j])
                    max_distance = max(max_distance, dist)
            
            self.sigma = max_distance / np.sqrt(2 * self.n_hidden_neurons)
            if self.sigma == 0: # Avoid division by zero if all centers are identical
                self.sigma = 1.0 # Fallback to a default value
            
        # Step 3: Calculate Hidden Layer Activations for training data.
        H = self._calculate_hidden_activations(X)

        # Step 4: Train Output Layer Weights using Least Squares
        # We need to solve H @ weights = y
        # weights = (H.T@H)^-1@H.T@y
        # Use np.linalg.lstsq for numerical stability
        self.weights = np.linalg.lstsq(H, y, rcond=None)[0]
    
    def predict(self, X):
        """
            Makes predictions using the trained RBF NN.

            Args:
                X (np.ndarray): Input data for prediction (n_samples, n_features).
            
            Returns:
                np.ndarray: Predicted output values (n_samples).
        """
        if self.centers is None or self.weights is None:
            raise RuntimeError("RBF NN not trained. Call fit() first.")
        
        H = self._calculate_hidden_activations(X)
        predictions = H @ self.weights
        return predictions
    
# ===============================================================

# --- 3. Generate Data ---
np.random.seed(42) # for reproducing the same data

X_train = np.linspace(0, 1, 100).reshape(-1, 1) # 100 samples, 1 feature
y_train = target_function(X_train.flatten())

X_test = np.linspace(0, 1, 200).reshape(-1, 1) # More points for smoother plot
y_test_true = np.sin(2 * np.pi * X_test.flatten()) # True function without noise for comparison

# ===============================================================

# --- 4. Initialize and Train RBF Network ---
n_hidden = 20 # Number of RBF(hidden) neurons (can be tuned)
rbf_nn = RBFNN(n_hidden_neurons=n_hidden)
rbf_nn.fit(X_train, y_train)

print(f'RBF Network Trained!')
print(f'Number of hidden neurons: {rbf_nn.n_hidden_neurons}')
print(f'Learned Centers:\n{rbf_nn.centers[:5]}...') # Show first 5 centers
print(f'Learned Sigma: {rbf_nn.sigma:4f}')
print(f'Learned Weights:\n{rbf_nn.weights[:5]}...') # Show first 5 weights

# ===============================================================

# --- 5. Make Predictions ---
y_pred = rbf_nn.predict(X_test)

# ===============================================================

# --- 6. Evaluate and Visualize ---
mse = mean_squared_error(y_test_true, y_pred)
print(f'\nMean Squared Error on test set (vs true function): {mse:4f}')

plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'o', label='Training Data (with noise)', markersize=4, alpha=0.6)
plt.plot(X_test, y_test_true, '-', label='True Function ($sin(2\pi x)$)', color='green', linewidth=2)
plt.plot(X_test, y_pred, '--', label='RBF NN Approximation', color='red', linewidth=2)

# Plot RBF neuron centers
plt.plot(rbf_nn.centers, np.zeros_like(rbf_nn.centers), 'x', color='purple', markersize=10, label='RBF Centers')

plt.title("RBF Neural Network for Function Approximation")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

