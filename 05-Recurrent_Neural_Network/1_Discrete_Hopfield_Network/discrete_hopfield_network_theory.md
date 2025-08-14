# Discrete Hopfield Network

## Table of Contents

1. [Introduction](#1-introduction)
2. [Hopfield Network Architecture](#2-hopfield-network-architecture)
3. [Encoding (Learning)](#3-encoding-learning)
4. [Recall](#4-recall)
5. [Energy Function and Stability Analysis](#5-energy-function-and-stability-analysis)
6. [Applications](#6-applications)
7. [See Also](#7-see-also)

---

## 1. Introduction

The **Discrete Hopfield Network** is a type of recurrent neural network that serves as an associative memory model.
It was introduced by **John Hopfield** in 1982 .
This network is *auto-associative*, meaning it can store a set of patterns and later retrieve the complete, original pattern from a noisy or incomplete version.
The network's dynamics are governed by discrete state equations, and it is a **single-layered network** with exhaustive bidirectional connections between its nodes.

---

## 2. Hopfield Network Architecture

A Hopfield network is a single-layer recurrent neural network.
A key architectural feature is that **every node (neuron) is connected to every other node**.
These connections are **bidirectional**, and there are **no self-loops** ($w_{ii} = 0$).

If the network has $n$ neurons, $n$ is also the dimension of the input pattern vectors.
In a discrete Hopfield network:

* Neurons have **binary states** (typically $+1$ or $-1$)
* Their dynamics are described by **discrete state equations**.

---

## 3. Encoding (Learning)

The process of storing patterns in a discrete Hopfield network is called **encoding**.
The network’s memory is stored in its **connection weights**.

For a given set of $m$ binary pattern vectors, the **Hebbian encoding rule** is used to construct the weight matrix $W$.
The weights $w_{ij}$ between neurons $i$ and $j$ are calculated as:

$$
w_{ij} = \sum_{p=1}^m P_{i,p} \, P_{j,p}, \quad i \ne j
$$

$$
w_{ii} = 0
$$

Here, $P_{i,p}$ represents the $i$-th component of the $p$-th binary pattern vector $P$.
If two neurons are in the **same state** across most patterns, $w_{ij}$ will be **positive** (excitatory);
if they are **opposite**, $w_{ij}$ will be **negative** (inhibitory).

---

## 4. Recall

The **recall process** retrieves a stored pattern from a noisy or partial input.

Steps:

1. **Initialize** the network’s state $O(t)$ with the input pattern $P$ at time $t = 0$.

2. **Update** each neuron’s state synchronously or asynchronously using:

   $$
   O_j(t+1) = h\left( \sum_{i=1}^n w_{ij} \, O_i(t) \right)
   $$

   Where $h$ is the signum activation function:

   $$
   h(x) =
   \begin{cases}
   +1, & \text{if } x > 0 \\
   -1, & \text{if } x < 0
   \end{cases}
   $$

3. **Repeat** until the network reaches an **equilibrium** state where:

   $$
   O(t+1) = O(t)
   $$

   for all neurons.

The final stable state vector $O(t+1)$ is the recalled stored pattern.

---

## 5. Energy Function and Stability Analysis

A key property of the Hopfield network is its **energy function**, which guarantees stability .
If weights are **symmetric** ($w_{ij} = w_{ji}$) and there are no self-loops ($w_{ii} = 0$), the network will always converge

The energy function is:

$$
E = -\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n w_{ij} \, O_i \, O_j
$$

Here, $O_i$ and $O_j$ are the outputs of neurons $i$ and $j$.

The network’s dynamics ensure that $E$ **never increases**, eventually reaching a **local minimum** that corresponds to a stored pattern.

---

## 6. Applications

Discrete Hopfield Networks are mainly used for:

* **Auto-associative memory** (recalling a complete pattern from noisy input)
* **Image reconstruction** from partially obscured views
* Other pattern completion and noise-reduction tasks.

---

## 7. See Also

* [Recurrent Neural Networks (RNNs)](https://www.google.com/search?q=rnn_theory.md)
* [Continuous Hopfield Network](https://www.google.com/search?q=continuous_hopfield_network_theory.md)
* [Boltzmann Machines](https://www.google.com/search?q=boltzmann_machines.md) — a probabilistic variant avoiding local minima
* [Bi-directional Associative Memory](https://www.google.com/search?q=bi_directional_associative_memory_theory.md) — a two-layer associative memory