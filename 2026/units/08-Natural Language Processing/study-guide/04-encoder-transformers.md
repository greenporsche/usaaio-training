# Encoder Transformers: BERT

**Prerequisites**: Self-attention mechanism, multi-head attention, layer normalization, PyTorch `nn.Module`
**USAAIO Relevance**: BERT is the canonical encoder-only transformer. IOAI problems ask you to trace attention patterns, predict masked tokens, explain why bidirectional context matters, and implement classification heads on top of pretrained encoders.

---

## Discovery

### The Bidirectional Breakthrough

Before BERT, language models read text in one direction. GPT (2018) read left-to-right. ELMo (2018) concatenated a left-to-right and right-to-left LSTM. But neither could jointly attend to context on **both sides** simultaneously.

Devlin et al. (2018) asked: **What if we trained a transformer to look in both directions at once?**

The problem: a standard language model objective (predict next token) leaks information in the bidirectional case — the model can trivially see the word it is supposed to predict.

The solution: **Masked Language Modeling (MLM)** — randomly mask some tokens and train the model to predict them from context on both sides.

```
Standard LM (left-to-right):
  "The cat sat on the ___"  → predict "mat"
  Can only see: "The cat sat on the"

BERT (bidirectional):
  "The cat [MASK] on the mat"  → predict "sat"
  Can see: "The cat __ on the mat"  (both sides!)
```

> **Socratic question**: Why can't we just train a regular left-to-right language model with bidirectional attention? What goes wrong?

### BERT: Bidirectional Encoder Representations from Transformers

| Detail | Value |
|---|---|
| Paper | Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018) |
| Architecture | Transformer encoder (no decoder) |
| Key innovation | Masked Language Modeling for bidirectional pretraining |
| Impact | State-of-the-art on 11 NLP benchmarks simultaneously |

---

## Intuition

### Architecture Overview

```
Input:   [CLS] The cat sat on the mat [SEP]

          ↓ Token embedding + Position embedding + Segment embedding

        ┌─────────────────────────────────────┐
        │        Transformer Encoder          │
        │                                     │
        │   Self-Attention (bidirectional)     │
        │   Feed-Forward Network              │
        │   Layer Normalization               │
        │                                     │
        │       × 12 layers (BERT-base)       │
        └─────────────────────────────────────┘

          ↓ Hidden states: (B, L, 768)

        h_[CLS]  h_The  h_cat  h_sat  h_on  h_the  h_mat  h_[SEP]
           ↓
     Classification    Token-level predictions
     (sequence-level)  (NER, MLM, etc.)
```

### Bidirectional vs. Causal Attention

The key difference between BERT and GPT is the **attention mask**:

```
BERT attention mask         GPT attention mask
(full / bidirectional):     (causal / lower-triangular):

  T h e c a t               T h e c a t
T[1 1 1 1 1 1]            T[1 0 0 0 0 0]
h[1 1 1 1 1 1]            h[1 1 0 0 0 0]
e[1 1 1 1 1 1]            e[1 1 1 0 0 0]
c[1 1 1 1 1 1]            c[1 1 1 1 0 0]
a[1 1 1 1 1 1]            a[1 1 1 1 1 0]
t[1 1 1 1 1 1]            t[1 1 1 1 1 1]

Every token sees every      Each token only sees
other token                  previous tokens
```

BERT can attend to the **future** because it is not trying to generate text — it is trained to fill in blanks.

### Masked Language Modeling (MLM)

During pretraining, 15% of tokens are selected for prediction:
- 80% are replaced with `[MASK]`
- 10% are replaced with a random token
- 10% are left unchanged

```
Original:    "The cat sat on the mat"
Selected:    positions 2 and 5 (0-indexed)
Masked:      "The cat [MASK] on the [MASK]"
                       ↑              ↑
Target:              "sat"          "mat"

Why not 100% [MASK]?
  - At fine-tuning time, there are no [MASK] tokens
  - The 10% random + 10% unchanged prevent the model
    from learning that [MASK] = "predict this"
```

### Special Tokens

```
Single sentence:     [CLS] sentence [SEP]
Sentence pair:       [CLS] sentence_A [SEP] sentence_B [SEP]

[CLS] — Classification token. Its hidden state summarizes the sequence.
[SEP] — Separator token. Marks sentence boundaries.
[MASK] — Mask token. Used during MLM pretraining.
[PAD] — Padding token. Fills sequences to equal length.
[UNK] — Unknown token. Replaces out-of-vocabulary tokens.
```

### Input Representation

BERT's input embedding is the **sum** of three embeddings:

```
Token embeddings:     [CLS]  The   cat   sat   [SEP]
                       ↓      ↓     ↓     ↓     ↓
Segment embeddings:    E_A    E_A   E_A   E_A   E_A   (all sentence A)
                       ↓      ↓     ↓     ↓     ↓
Position embeddings:   P_0    P_1   P_2   P_3   P_4
                       ↓      ↓     ↓     ↓     ↓
Input = Token + Segment + Position
```

---

## Math

### Pretraining Objectives

**1. Masked Language Modeling (MLM)**

Given input sequence $x = (x_1, \dots, x_L)$ with masked positions $\mathcal{M} \subset \{1, \dots, L\}$:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\setminus \mathcal{M}})$$

where $x_{\setminus \mathcal{M}}$ is the sequence with masked positions replaced. The prediction is:

$$P(x_i = w \mid x_{\setminus \mathcal{M}}) = \text{softmax}(W h_i + b)_w$$

where $h_i \in \mathbb{R}^D$ is the hidden state at position $i$ from the final transformer layer, $W \in \mathbb{R}^{|V| \times D}$, and $b \in \mathbb{R}^{|V|}$.

**2. Next Sentence Prediction (NSP)**

Given sentence pair (A, B), predict whether B follows A:

$$P(\text{IsNext} \mid A, B) = \sigma(W_{\text{NSP}} h_{[\text{CLS}]} + b_{\text{NSP}})$$

where $h_{[\text{CLS}]}$ is the hidden state of the `[CLS]` token.

50% of training pairs are actual consecutive sentences (IsNext), 50% are random (NotNext).

*Note*: Later work (RoBERTa) showed NSP provides minimal benefit. MLM is the key objective.

### BERT Model Sizes

| | Layers ($L$) | Hidden ($D$) | Heads ($H$) | Parameters |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

Head dimension: $d_k = D / H$ (64 for BERT-base).

### Shape Trace Through BERT

```
Input token IDs:          (B, L)              int64
Token embeddings:         (B, L, 768)         float32   via nn.Embedding
+ Position embeddings:    (B, L, 768)         float32   added
+ Segment embeddings:     (B, L, 768)         float32   added
= BERT input:             (B, L, 768)         float32

Per layer (×12):
  Q, K, V projection:    (B, L, 768) → (B, 12, L, 64)   reshape for multi-head
  Attention scores:       (B, 12, L, L)                    QK^T / sqrt(64)
  Attention weights:      (B, 12, L, L)                    after softmax
  Attention output:       (B, 12, L, 64) → (B, L, 768)    concat + project
  FFN:                    (B, L, 768) → (B, L, 3072) → (B, L, 768)

Final hidden states:      (B, L, 768)
[CLS] representation:     (B, 768)            first token
MLM logits:               (B, L, |V|)         vocabulary prediction
```

---

## Code

### Using Pretrained BERT with HuggingFace

```python
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Masked prediction
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, L, |V|)

# Find the [MASK] position
mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Get top-5 predictions
top5 = torch.topk(logits[0, mask_idx], 5, dim=-1)
for score, idx in zip(top5.values[0], top5.indices[0]):
    print(f"  {tokenizer.decode([idx]):>10s}  (score: {score:.2f})")
# Expected: paris (top prediction)
```

### BERT for Classification

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn as nn

# Option 1: HuggingFace model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Option 2: Manual classification head
class BertClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token: (B, 768)
        logits = self.classifier(cls_hidden)              # (B, num_classes)
        return logits
```

### Analyzing Attention Patterns

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple of (B, H, L, L) tensors, one per layer
# Layer 0, Head 0 attention weights:
attn = outputs.attentions[0][0, 0]  # (L, L)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Print attention from "sat" to all other tokens
sat_idx = tokens.index("sat")
for i, token in enumerate(tokens):
    print(f"  sat → {token:>8s}: {attn[sat_idx, i]:.3f}")
```

---

## Key Takeaways

1. **BERT** is an **encoder-only** transformer with **bidirectional** (full) attention
2. **Masked Language Modeling** is the key pretraining objective — predict randomly masked tokens from surrounding context
3. The `[CLS]` token's hidden state serves as a **sequence-level representation** for classification
4. BERT uses **three embeddings summed**: token + position + segment
5. The attention mask is **all ones** (every token attends to every other token)
6. For classification: take `[CLS]` hidden state → add a linear head → fine-tune
7. For token-level tasks (NER): take each token's hidden state → per-token classification
8. On USAAIO exams, be ready to trace attention patterns and explain why bidirectional attention helps for understanding tasks
