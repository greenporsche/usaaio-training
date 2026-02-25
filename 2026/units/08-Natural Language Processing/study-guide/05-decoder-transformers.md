# Decoder Transformers: GPT

**Prerequisites**: Self-attention, causal masking, softmax, autoregressive models
**USAAIO Relevance**: GPT is the canonical decoder-only transformer. IOAI problems ask you to implement causal attention masks, trace autoregressive generation, analyze sampling strategies (temperature, top-k, top-p), and compare encoder vs. decoder architectures.

---

## Discovery

### The Generative Approach

While BERT asks "What word fits in this blank?", GPT asks "What word comes next?"

Radford et al. (2018) showed that a simple left-to-right language model, trained on enough data, could learn surprisingly powerful representations — and generate coherent text.

```
BERT (fill in the blank):
  "The [MASK] chased the mouse"  → "cat"
  (understanding task)

GPT (predict the next token):
  "The cat chased the"  → "mouse"
  (generation task)
```

> **Socratic question**: If BERT sees both directions and GPT only sees left-to-right, why would anyone choose GPT? What can GPT do that BERT cannot?

The answer: **generation**. BERT cannot generate text because it was trained to fill blanks, not to produce sequences token by token. GPT's left-to-right training naturally enables autoregressive generation.

### The GPT Family

| Model | Year | Parameters | Key Innovation |
|---|---|---|---|
| GPT-1 | 2018 | 117M | Pretrained transformer decoder + fine-tuning |
| GPT-2 | 2019 | 1.5B | Zero-shot task transfer (no fine-tuning needed) |
| GPT-3 | 2020 | 175B | Few-shot learning via in-context prompting |
| GPT-4 | 2023 | ~1.8T (estimated) | Multimodal (text + vision) |

---

## Intuition

### Architecture Overview

GPT is simpler than BERT in some ways — it is just a stack of transformer decoder blocks (without cross-attention):

```
Input:   The cat sat on the

          ↓ Token embedding + Position embedding

        ┌─────────────────────────────────────┐
        │      Transformer Decoder Block      │
        │                                     │
        │   Causal Self-Attention             │
        │   (masked so tokens only see left)  │
        │   Feed-Forward Network              │
        │   Layer Normalization               │
        │                                     │
        │       × 12 layers (GPT-1)           │
        └─────────────────────────────────────┘

          ↓ Hidden states: (B, L, D)

        h_The  h_cat  h_sat  h_on  h_the
                                     ↓
                              Linear → Softmax
                                     ↓
                              P(next token | prefix)
                              "mat" (0.15), "..." ...
```

### The Causal Attention Mask

The critical difference: GPT uses a **causal mask** (lower-triangular matrix) so each token can only attend to itself and previous tokens:

```
Tokens: [The, cat, sat, on, the]

Attention mask (causal):
       The cat sat on the
The  [  1   0   0   0   0 ]   ← "The" only sees itself
cat  [  1   1   0   0   0 ]   ← "cat" sees "The" and itself
sat  [  1   1   1   0   0 ]   ← "sat" sees "The", "cat", itself
on   [  1   1   1   1   0 ]
the  [  1   1   1   1   1 ]   ← "the" sees everything before it

Positions with 0 are set to -inf before softmax → attention weight = 0
```

This ensures the model cannot "cheat" by looking at future tokens during training.

### Autoregressive Generation

GPT generates text one token at a time, each time feeding the entire sequence so far:

```
Step 1: Input: "The"           → Model predicts next: "cat"
Step 2: Input: "The cat"       → Model predicts next: "sat"
Step 3: Input: "The cat sat"   → Model predicts next: "on"
Step 4: Input: "The cat sat on" → Model predicts next: "the"
...

Each step:
  1. Feed entire sequence into model
  2. Take the logits at the LAST position
  3. Sample or argmax to get next token
  4. Append token to sequence
  5. Repeat until [EOS] or max length
```

### Sampling Strategies

The raw model outputs **logits** $z \in \mathbb{R}^{|V|}$ for the next token. How do we pick one?

**Greedy decoding**: Always pick the highest-probability token.
```
logits: [2.1, 5.3, 0.8, 3.2, ...]
greedy: token index 1 (score 5.3)
Problem: repetitive, boring text
```

**Temperature sampling**: Scale logits by temperature $T$ before softmax.
```
T = 1.0: normal distribution (default)
T = 0.5: sharper (more confident, less diverse)
T = 2.0: flatter (less confident, more diverse)
T → 0:   approaches greedy
T → ∞:   approaches uniform random
```

```
         P(token)
T=0.2    ▓░░░░░░   (very peaked — almost greedy)
T=1.0    ▓▓▒░░░░   (balanced)
T=2.0    ▓▓▓▒▒░░   (spread out — more creative)
```

**Top-k sampling**: Only sample from the top $k$ most likely tokens.
```
All tokens:  [0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01]
Top-3:       [0.40, 0.25, 0.15, 0,    0,    0,    0,    0   ]
Renormalize: [0.50, 0.31, 0.19, 0,    0,    0,    0,    0   ]
```

**Top-p (nucleus) sampling**: Sample from the smallest set of tokens whose cumulative probability exceeds $p$.
```
Sorted probs: [0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01]
Cumulative:   [0.40, 0.65, 0.80, 0.88, 0.93, 0.97, 0.99, 1.00]
                                   ↑ p = 0.9: include first 4 tokens
```

Top-p adapts automatically — for confident predictions, fewer tokens are included. For uncertain predictions, more are included.

---

## Math

### Language Model Objective

GPT is trained to maximize the log-likelihood of the next token:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \dots, x_{t-1}; \theta)$$

where:
$$P(x_t = w \mid x_{<t}) = \text{softmax}(W_{\text{head}} \cdot h_t)_w$$

and $h_t$ is the hidden state at position $t$ from the final transformer layer.

This is equivalent to cross-entropy loss at each position, teacher-forced (the correct token is always provided as input at training time).

### Causal Mask in Attention

The standard self-attention computation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For causal attention, we add a mask $M$ **before** softmax:

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where:
$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

Since $\text{softmax}(-\infty) = 0$, future positions contribute zero attention weight.

### Temperature Scaling

Given logits $z \in \mathbb{R}^{|V|}$ and temperature $T > 0$:

$$P(w) = \frac{\exp(z_w / T)}{\sum_{w'} \exp(z_{w'} / T)}$$

Properties:
- $T = 1$: standard softmax
- $T \to 0^+$: $P$ concentrates on $\arg\max z$ (greedy)
- $T \to \infty$: $P$ approaches uniform distribution
- Temperature does not change the **ranking** of tokens, only the **spread**

### Perplexity

Perplexity measures how "surprised" the model is by the test data:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)$$

- Lower perplexity = better model
- PPL = 1 means the model perfectly predicts every token
- PPL = $|V|$ means the model is as confused as random guessing

---

## Code

### Causal Attention Mask

```python
import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower-triangular) attention mask.

    Returns: (L, L) tensor where mask[i][j] = 0 if j <= i, else -inf
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Example for seq_len=4:
# tensor([[  0., -inf, -inf, -inf],
#         [  0.,   0., -inf, -inf],
#         [  0.,   0.,   0., -inf],
#         [  0.,   0.,   0.,   0.]])
```

### GPT Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate(prompt, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # (1, V) — last position only

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits[logits < min_val] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Stop at end-of-text
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])

# Usage
print(generate("Once upon a time", temperature=0.8, top_p=0.9))
```

### Comparing BERT vs GPT Architectures

```python
from transformers import BertModel, GPT2Model

bert = BertModel.from_pretrained("bert-base-uncased")
gpt2 = GPT2Model.from_pretrained("gpt2")

# BERT: bidirectional attention (no causal mask)
# - 12 layers, 768 hidden, 12 heads
# - Input: [CLS] tokens [SEP]
# - Output: hidden states for each token
# - Use [CLS] for classification

# GPT-2: causal attention (lower-triangular mask)
# - 12 layers, 768 hidden, 12 heads
# - Input: tokens (no special start/end needed for generation)
# - Output: hidden states for each token
# - Use LAST token for next-word prediction

# Key structural differences:
print(f"BERT parameters: {sum(p.numel() for p in bert.parameters()):,}")
print(f"GPT-2 parameters: {sum(p.numel() for p in gpt2.parameters()):,}")
```

---

## Key Takeaways

1. **GPT** is a **decoder-only** transformer with **causal** (left-to-right) attention
2. The **causal mask** is a lower-triangular matrix that prevents attending to future tokens
3. GPT is trained with **next token prediction** — a standard language modeling objective
4. **Autoregressive generation**: generate one token at a time, append, repeat
5. **Temperature** controls the sharpness of the output distribution ($T < 1$ sharper, $T > 1$ flatter)
6. **Top-k** samples from the $k$ most probable tokens; **top-p** samples from the smallest set exceeding probability $p$
7. **Perplexity** measures model quality (lower = better)
8. GPT excels at **generation tasks**; BERT excels at **understanding tasks** (classification, NER, QA)
9. On USAAIO exams, be ready to implement causal masking and trace generation step by step
