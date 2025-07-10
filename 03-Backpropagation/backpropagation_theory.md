# Backpropagation Algorithm

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Why Backpropagation?](#2-why-backpropagation)
3.  [Prerequisites](#3-prerequisites)
4.  [Core Idea: Gradient Descent](#4-core-idea-gradient-descent)
5.  [The Backpropagation Algorithm Steps](#5-the-backpropagation-algorithm-steps)
    * [Step 1: Forward Pass](#step-1-forward-pass)
    * [Step 2: Calculate Output Layer Error](#step-2-calculate-output-layer-error)
    * [Step 3: Backpropagate Error (Hidden Layers)](#step-3-backpropagate-error-hidden-layers)
    * [Step 4: Update Weights and Biases](#step-4-update-weights-and-biases)
6.  [Key Mathematical Concepts](#6-key-mathematical-concepts)
    * [Chain Rule](#61-chain-rule)
    * [Derivative of Activation Functions](#62-derivative-of-activation-functions)
7.  [Learning Rate](#7-learning-rate)
8.  [Epochs and Batches](#8-epochs-and-batches)
9.  [Advantages and Limitations](#9-advantages-and-limitations)
10. [See Also](#10-see-also)

---

## 1. Introduction

Backpropagation (short for "backward propagation of errors") is the fundamental algorithm for training artificial neural networks, particularly Multi-Layer Perceptrons (MLPs). It's an iterative gradient descent optimization algorithm that works by calculating the gradient of the loss function with respect to the weights in the network. This gradient is then used to adjust the weights in the direction that minimizes the loss.

## 2. Why Backpropagation?

While the Perceptron could learn linearly separable problems, it lacked a mechanism to train networks with multiple layers (hidden layers) that are essential for solving complex, non-linear problems like the XOR problem. Backpropagation provides an efficient way to distribute the error from the output layer back through the hidden layers, allowing the network to learn intricate patterns.

## 3. Prerequisites

Understanding backpropagation requires familiarity with:
* **Multi-Layer Perceptrons (MLPs):** Networks with at least one hidden layer.
* **Activation Functions:** Functions applied to the weighted sum of inputs (e.g., Sigmoid, ReLU, Tanh).
* **Loss (Cost) Functions:** A function that quantifies the difference between the network's output and the true target (e.g., Mean Squared Error, Cross-Entropy Loss).
* **Gradient Descent:** An optimization algorithm used to find the minimum of a function.
* **Calculus (Chain Rule):** Essential for computing gradients across multiple layers.

## 4. Core Idea: Gradient Descent

Backpropagation uses gradient descent. The goal is to minimize a loss function $L(W)$ by iteratively adjusting the network's weights $W$. The update rule for a weight $w$ is:

$$ w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w} $$

Where $\eta$ is the learning rate, and $\frac{\partial L}{\partial w}$ is the partial derivative of the loss function with respect to the weight $w$. Backpropagation is the method used to efficiently calculate these partial derivatives for all weights and biases in the network.

## 5. The Backpropagation Algorithm Steps

For each training example:

### Step 1: Forward Pass

* Input features are fed into the network.
* The network computes the output for each neuron, layer by layer, from the input to the output.
* The activation function is applied at each neuron to produce its output.
* The final output of the network is obtained.
* The loss (error) between the predicted output and the true target is calculated.

### Step 2: Calculate Output Layer Error

* For each neuron in the output layer, calculate the error term ($\delta$) using the difference between the predicted output ($\hat{y}$) and the true target ($y$), multiplied by the derivative of its activation function.
    * For Mean Squared Error (MSE) and Sigmoid activation:
        $$ \delta_{output} = (y - \hat{y}) \cdot \hat{y}(1 - \hat{y}) $$
        (where $\hat{y}(1-\hat{y})$ is the derivative of the sigmoid function)

### Step 3: Backpropagate Error (Hidden Layers)

* The error is propagated backward from the output layer to the preceding hidden layer.
* For each neuron in a hidden layer, its error term ($\delta_{hidden}$) is calculated based on the error terms of the neurons in the next layer that it connects to, weighted by the connection weights. This involves summing the products of the next layer's $\delta$ and the connecting weights, then multiplying by the derivative of the hidden neuron's own activation function.
    * For a hidden neuron $j$:
        $$ \delta_j = (\sum_k w_{jk} \delta_k) \cdot f'(net_j) $$
        Where $\delta_k$ are error terms from the next layer's neurons $k$ connected to $j$, $w_{jk}$ are the weights connecting $j$ to $k$, and $f'(net_j)$ is the derivative of neuron $j$'s activation function with respect to its net input.
* This process continues until the error terms for all hidden layers are computed.

### Step 4: Update Weights and Biases

* Once the error terms ($\delta$) are calculated for all neurons, the weights and biases are updated using the gradient descent rule.
* For a weight $w_{ij}$ connecting neuron $i$ to neuron $j$:
    $$ \Delta w_{ij} = \eta \cdot \delta_j \cdot o_i $$
    $$ w_{ij}^{\text{new}} = w_{ij}^{\text{old}} + \Delta w_{ij} $$
    Where $o_i$ is the output of neuron $i$ (the input to neuron $j$).
* For a bias $b_j$ of neuron $j$:
    $$ \Delta b_j = \eta \cdot \delta_j $$
    $$ b_j^{\text{new}} = b_j^{\text{old}} + \Delta b_j $$
* These updates are performed for all weights and biases in the network.

These four steps are repeated for each training example (or batch of examples) for a predefined number of epochs.

## 6. Key Mathematical Concepts

### 6.1. Chain Rule

The chain rule from calculus is fundamental to backpropagation. It allows us to compute the derivative of a composite function. In neural networks, the loss depends on the output, which depends on the activations, which depend on the weighted sums, which depend on the weights. The chain rule enables us to efficiently calculate how a change in a weight in an early layer affects the final loss.

$$ \frac{dL}{dw_{ij}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial net_k} \cdot \frac{\partial net_k}{\partial o_j} \cdot \frac{\partial o_j}{\partial net_i} \cdot \frac{\partial net_i}{\partial w_{ij}} $$
(This is a simplified conceptual representation; the actual application is more detailed.)

### 6.2. Derivative of Activation Functions

The choice of activation function directly impacts the backpropagation process because its derivative is used in calculating the error terms.
* **Sigmoid:** $f'(z) = f(z)(1 - f(z))$
* **Tanh:** $f'(z) = 1 - f(z)^2$
* **ReLU:** $f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$

## 7. Learning Rate ($\eta$)

As with the Perceptron, the learning rate controls the step size during weight updates. Proper tuning is crucial for efficient training.

## 8. Epochs and Batches

* **Epoch:** One full pass through the entire training dataset.
* **Batch:** A subset of the training data used to compute the gradient and update weights.
    * **Stochastic Gradient Descent (SGD):** Batch size = 1 (updates after each example).
    * **Mini-Batch Gradient Descent:** Batch size > 1 but less than total dataset size (most common).
    * **Batch Gradient Descent:** Batch size = total dataset size (updates after one full pass).

## 9. Advantages and Limitations

**Advantages:**
* Enables training of multi-layer networks to solve complex, non-linear problems.
* Computationally efficient compared to earlier methods.
* Forms the basis for modern deep learning.

**Limitations:**
* **Vanishing/Exploding Gradients:** Can occur in very deep networks, making learning difficult.
* **Local Minima:** Gradient descent can get stuck in local minima of the loss function.
* **Hyperparameter Tuning:** Sensitive to choices of learning rate, number of layers, neurons, etc.
* Requires differentiable activation functions.

## 10. See Also

* [Multi-Layer Perceptron (MLP)](../multi_layer_perceptron/mlp_theory.md) (Potential future topic folder)
* [Gradient Descent Optimization](../optimization_algorithms/gradient_descent.md) (Potential future topic folder)

---

See the Python implementation of Backpropagation [here](backpropagation_implementation.py).