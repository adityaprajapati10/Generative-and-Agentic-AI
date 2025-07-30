## Introduction to RNN

Recurrent Neural Networks (RNNs) are a specialized type of neural network designed for **sequential data** like text, speech, and time series. They differ from traditional neural networks in their ability to retain information from previous time steps using internal memory.

In standard feedforward neural networks:
- Each input is processed independently.
- There's no memory of previous inputs.
- Only **forward propagation** and **backward propagation** are involved.

But this fails in scenarios where context matters — for instance, predicting the next word in a sentence.

In neural network we learn forward propagation and then system learn through backward propagation  but we know every time system will be consider all of these input as a new input every time. system will not be able to retain any kind of information or knowledge from the previous input. there no linkage between a memory the previous input and next input. so basically it try to pass a new input every time and based on that new input it will try to just adjust the weights in between means try to change the relationship inside the network, this is what happed in backward propagation.

## How RNN Works

RNNs maintain a **hidden state** `hₜ`, which captures information from previous inputs. At each time step `t`, the network takes:
- The current input `xₜ`
- The previous hidden state `hₜ₋₁`

and produces:
- A new hidden state `hₜ`
- An output `yₜ`

So the input is `[hₜ₋₁, xₜ]`.

This way, RNNs can process sequences of arbitrary length and capture short-term dependencies in data.

## Limitation of RNN: Vanishing Gradient

RNNs struggle with **long-term dependencies** due to the **vanishing gradient problem**. As gradients are propagated backward through time, they become very small (or sometimes explode), making it difficult to learn from earlier time steps.

As a result:
- RNNs are good at remembering recent inputs.
- But they forget inputs from many time steps ago.

---

## LSTM: Long Short-Term Memory Networks

To overcome RNN's limitations, **LSTM** networks were introduced.

###  Key Strengths of LSTM:
- Maintains **long-term memory** via a memory cell.
- Uses **gates** to control what information to forget, remember, and output.
- Effectively handles **long-range dependencies**.

##  LSTM Architecture

LSTM includes the following components:

1. **Forget Gate** `fₜ`
2. **Input Gate** `iₜ`
3. **Candidate Memory** `C̃ₜ`
4. **Output Gate** `oₜ`
5. **Cell State** `Cₜ`
6. **Hidden State** `hₜ`

All gates take as input:  
`[hₜ₋₁, xₜ]`

### 1. Forget Gate

- Determines what information to discard from the cell state.

$$
fₜ = \sigma(hₜ₋₁ \cdot Wₕ + Xₜ \cdot Wₓ + b)
$$

### 2. Input Gate and Candidate Memory

- **Input Gate** Decides **what new information to store**.
  $$
iₜ = \sigma(hₜ₋₁ \cdot Wₕᵢ + xₜ \cdot Wₓᵢ + bᵢ)
$$

- **Candidate Memory** **new candidate values** to add to the cell state.

   $$
\tilde{C}_ₜ = \tanh(hₜ₋₁ \cdot Wₕ𝚌 + xₜ \cdot Wₓ𝚌 + b𝚌)
$$

- Update cell state: Updates the cell state with what we remembered and added.

$$
Cₜ = fₜ \odot Cₜ₋₁ + iₜ \odot \tilde{C}_ₜ
$$

### 3. Output Gate 

 - Decides **what to output** as the hidden state.

   $$
oₜ = \sigma(hₜ₋₁ \cdot Wₕₒ + xₜ \cdot Wₓₒ + bₒ)
$$

$$
hₜ = oₜ \odot \tanh(Cₜ)
$$



If we want to generate a prediction:

$$
yₜ = \text{softmax}(W \cdot hₜ + b)
$$

## Summary of Gates

| Gate         | Purpose               | Activation | Equation Style                         |
|--------------|------------------------|------------|-----------------------------------------|
| Forget Gate  | What to forget         | Sigmoid    | `fₜ = σ(hₜ₋₁ Wₕ + xₜ Wₓ + b)`           |
| Input Gate   | What to add            | Sigmoid    | `iₜ = σ(hₜ₋₁ Wₕᵢ + xₜ Wₓᵢ + bᵢ)`        |
| Candidate    | Candidate memory       | Tanh       | `~Cₜ = tanh(hₜ₋₁ Wₕ𝚌 + xₜ Wₓ𝚌 + b𝚌)`     |
| Cell State   | Final memory value     | —          | `Cₜ = fₜ * Cₜ₋₁ + iₜ * ~Cₜ`             |
| Output Gate  | What to output         | Sigmoid    | `oₜ = σ(hₜ₋₁ Wₕₒ + xₜ Wₓₒ + bₒ)`        |
| Final Output | Class prediction       | Softmax    | `yₜ = softmax(W · hₜ + b)`              |
- **σ (sigmoid):** squashes values between 0 and 1  
- **tanh:** squashes values between -1 and 1  
- **softmax:** used for classification outputs

#### Memory Flow Diagram
- Cell state `Cₜ` flows horizontally across time steps  
- Gates control how much information flows in or out  
- Hidden state `hₜ` is used as output at each step

---
## Example: One-Hot Encoding with LSTM

Let’s consider the word “help” and encode each character:

| Character | One-Hot Vector |
|-----------|----------------|
| h         | [1, 0, 0, 0]   |
| e         | [0, 1, 0, 0]   |
| l         | [0, 0, 1, 0]   |
| p         | [0, 0, 0, 1]   |

Assume:

- `xₜ = [1, 0, 0, 0]`
- `hₜ₋₁ = [0, 0]`
- `Cₜ₋₁ = [0, 0]`

Suppose the gates output:

- `fₜ = [0.8, 0.3]`
- `iₜ = [0.5, 0.7]`
- `C̃ₜ = [0.2, -0.1]`
- Updated `Cₜ = [0.1, -0.07]`
- `oₜ = [0.9, 0.6]`
- Final `hₜ = [0.089, -0.041]`

## Prediction: Using Softmax

To predict the next character:


$$
Yₜ = \text{softmax}(W \cdot hₜ + b)
$$



This returns the probability of each character (e.g., `h`, `e`, `l`, `p`) being the next one.

If **context window = 3**, and the input is “hel”, LSTM predicts “p”.

## Loss Calculation and Training

LSTM is trained using backpropagation through time (BPTT). The model computes gradients and updates weights for:

- Forget gate: `∂L/∂W_f`
- Input gate: `∂L/∂W_i
- Candidate memory: `∂L/∂W_c`
- Output gate: `∂L/∂W_o`

This allows each gate to be trained **independently for its specific function**, resulting in more effective learning of sequence patterns.

## Summary Table

| Feature                 | RNN                 | LSTM                          |
| ----------------------- | ------------------- | ----------------------------- |
| Memory Type             | Short-term          | Long-term (via cell state)    |
| Handles Long Sequences? | Limited             | Yes                           |
| Architecture            | Simple              | Complex (uses gates)          |
| Problem Solved          | Sequence modeling   | Vanishing gradient            |
| Applications            | Basic sequence data | Language, speech, forecasting |

