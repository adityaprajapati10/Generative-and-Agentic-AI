## What is a Neural Network?

A **neural network** is inspired by how the **human brain** works — with interconnected **neurons** that learn from data.  
Technically, it’s called a **Perceptron** — a building block of Artificial Neural Networks (ANN).

## Basic Structure

- **Input Layer**: Takes features as input (x₁, x₂, …, xₙ).
- **Hidden Layers**: Perform computations through weighted connections.
- **Output Layer**: Produces the final prediction or class label.

Each connection has **learnable parameters**:
- **Weights (w)** – Strength of input
- **Biases (b)** – Shifts activation

---

## Flow: Forward + Backward Propagation

### Forward Propagation

- Multiply inputs by weights, add bias
- Apply activation function  

$$
z = x \cdot w + b  
$$

$$
a = f(z)
$$
```
         x
         │
         ▼
      [ Input ]
         │
     ┌───┴─────┐
     │         │
     ▼         ▼
   w1*x      w2*x
     │         │
     ▼         ▼
   [ h1 ]     [ h2 ]
     │         │
   ReLU      ReLU
     │         │
     ▼         ▼
   z1 = ReLU(h1)     z2 = ReLU(h2)

```

###  Backpropagation
- Calculates error
- Updates weights using gradients
$$
w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w}
$$
$$
b_{\text{new}} = b - \eta \cdot \frac{\partial L}{\partial b}
$$

Where:  
- $\eta$ = learning rate  
- $L$ = loss function

```
            x
            │
            ▼
         [ Input ]
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
   w1*x          w2*x
     │             │
     ▼             ▼
   [ h1 ]         [ h2 ]
     │             │
   ReLU(h1)      ReLU(h2)
     │             │
     ▼             ▼
   z1             z2
     │             │
     └──────┬──────┘
            ▼
     y_pred_input = z1·w3 + z2·w4 + b
            ▼
         [ y_pred ]

```


Output used in loss → backpropagation starts from here
## Key Components

### Activation Functions

| Function | Formula                             | Output Range |
| -------- | ----------------------------------- | ------------ |
| Sigmoid  | $\frac{1}{1 + e^{-x}}$              | (0, 1)       |
| Tanh     | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1)      |
| ReLU     | $\max(0, x)$                        | [0, ∞)       |

These add **non-linearity** so that the network can learn complex functions.

### Loss Function (Objective Function)

Measures how far the predicted value ($\hat{y}$) is from the actual value ($y$).

- **Mean Squared Error (MSE)**:  
$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### Optimizer

Used to **update weights and biases** during training using gradients.

- Formula:
$$
w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w}
$$

$$
b_{\text{new}} = b - \eta \cdot \frac{\partial L}{\partial b}
$$

###  Gradient

- Measures the **rate of change of loss** with respect to a parameter (like weight).
- Used to **minimize the loss** during backpropagation.

## Epoch

- One full pass of **entire training data** through the network.
- Multiple epochs help the model **learn better and reduce loss** gradually.

##  Summary

>A neural network mimics human brain learning using interconnected layers.  
> Each neuron learns using weights, biases, and activation functions.  
> The training happens via forward and backward propagation using **gradients** and **optimizers**.

---
