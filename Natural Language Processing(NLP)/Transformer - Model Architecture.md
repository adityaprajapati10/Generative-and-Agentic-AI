
The Transformer is a **deep learning model architecture** introduced in the paper **“Attention Is All You Need”**. Unlike RNNs or LSTMs, it **doesn’t use recurrence** and instead relies on a mechanism called **Self-Attention** (or Multi-Head Attention) to process sequences in parallel.

## Input Preparation

Before sending input to the model:

1. **Words → Embeddings**
2. **Add Positional Encoding** (so model knows the position/order of words)
3. Final input = `Embedding + Position`

Example:  
“My name is Tom” → `[x₁, x₂, ..., x₆]`

## Transformer Structure

The model has two main blocks:
### 1. Encoder (Stacked Layers)
Each encoder layer contains:
- **Multi-Head Self Attention**
- **Add + Norm**
- **Feed Forward Neural Network**
- **Add + Norm**

There are 6 layers stacked one above the other in the original paper.

### 2. Decoder (Stacked Layers)
Each decoder layer contains:
- **Masked Multi-Head Self Attention**  
  *(prevents seeing future words during training)*
- **Multi-Head Attention (with encoder output)**
- **Add + Norm**
- **Feed Forward Neural Network**
- **Add + Norm**

---
## Attention Mechanism

We are trying to build better relationships between the data — like how the word "My" is connected to "Name". This is done by computing **Query, Key, and Value** vectors for each word.

Each input word is transformed into:

- **Query (Q)** — What we’re looking for  
- **Key (K)** — What we have  
- **Value (V)** — What we’ll use  

And we calculate them as:
```

q₁ = x₁ · Wq

k₁ = x₁ · Wk

v₁ = x₁ · Wv

  

q₂ = x₂ · Wq

k₂ = x₂ · Wk

v₂ = x₂ · Wv

```

These weights (Wq, Wk, Wv) are trainable — meaning they learn relationships during training.

---
### Attention Score

Let’s say we are currently at the word **"My"**.

We calculate the **score** between the current word’s **query** and all the other **keys** to see how "My" relates to other words like "Name".

So:
- Score between **"My" and "My"** → q1 · k1  
- Score between **"My" and "Name"** → q1 · k2  
- Score between **"Name" and "My"** → q2 · k1  
- Score between **"Name" and "Name"** → q2 · k2

So, for "My", we’ll get two scores:  
`q1 · k1` and `q1 · k2`

These scores tell us **how much "My" is related to itself and "Name"**.

But to stabilize the training, we divide each score by √dk (dimension of the key vector).  
Then we apply **Softmax** to get probabilities (like % of attention).

Now we multiply these probabilities with the **value vectors**:

```
Softmax(q1·k1) * v1 → V11
Softmax(q1·k2) * v2 → V12
Z1 = V11 + V12 → Output vector for "My"
```

Same happens for "Name":

```
Softmax(q2·k1) * v1 → V21
Softmax(q2·k2) * v2 → V22
Z2 = V21 + V22 → Output vector for "Name"
```

These outputs (Z1, Z2, ...) are then **concatenated across multiple heads** and passed through a linear layer.

---
##  Multi-Head Attention

Instead of doing this once, we **split into multiple heads** (like multiple attentions in parallel), learn different relationships, and **concatenate** results.

```

Multi-Head Attention = [Z₁, Z₂, Z₃, ...] · Wₒ

```

  

---

## Masked Multi-Head Attention (Decoder)

During training, the decoder can’t peek into the **future words**.  
So it uses **masking** in the attention step to block future tokens.

Example:

```

Input: "The cat is _"

Mask:   ✔️   ✔️  ❌

```

---
## Final Output

After decoding, we apply a **linear + softmax layer** to predict the next word from the vocabulary.

---
## Summary – Full Flow

```
Input → Embedding + Position
→ Encoder (Multi-Head Attention + FFN)
→ Decoder (Masked Attention + Encoder Attention + FFN)
→ Linear → Softmax → Output
```

---
## BPE – Byte Pair Encoding

Before input is sent to the model, words are broken into subwords using **BPE (Byte Pair Encoding)**.  
Example:

```

"playing" → ["play", "##ing"]

"unwanted" → ["un", "##want", "##ed"]

```

  This helps handle **rare words** and **spelling variations** efficiently.
  
---
## Why Transformers?

- Parallelization (no recurrence)
- Handles long-range dependencies well
-  Faster training
-  Backbone of modern models like BERT, GPT, T5, etc.