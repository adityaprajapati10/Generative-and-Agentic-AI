GRU is a **simplified variant of LSTM** designed to solve the vanishing gradient problem in RNNs while being more efficient. It has **fewer gates and parameters** than LSTM, making it **faster** and often performing similarly.


##  GRU vs LSTM

| Component      | LSTM                    | GRU                          |
|----------------|--------------------------|-------------------------------|
| Gates          | 3 (Forget, Input, Output) | 2 (Reset, Update)            |
| Memory Cell    | Present (`Cₜ`)           | Not present                  |
| Hidden State   | `hₜ`                     | `hₜ`                         |
| Efficiency     | Slower (more params)     | Faster (fewer params)        |
| Usage          | Complex sequences        | Efficient real-time tasks    |

## GRU Gates
### 1. Reset Gate

- Controls how much of the **previous hidden state** to forget.
- If reset gate is close to 0 → forget previous memory.

$$
rₜ = \sigma(Wᵣₕ \cdot h_{t-1} + Wᵣₓ \cdot xₜ + bᵣ)
$$

### 2. Update Gate 

- Combines the role of **forget gate and input gate** in LSTM.
- Controls how much of the new state to write vs how much of the old state to keep.

$$
zₜ = \sigma(W_zₕ \cdot h_{t-1} + W_zₓ \cdot xₜ + b_z)
$$


