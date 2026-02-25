# Natural Language Processing — Cheat Sheet

> Quick reference for USAAIO 2026 | AI 510-NLP

---

## Tokenization

| Method | Approach | Pros | Cons |
|---|---|---|---|
| **Character-level** | Each character is a token | Small vocab, no OOV | Very long sequences, loses word meaning |
| **BPE** | Iteratively merge most frequent byte pairs | Balances vocab/seq length | Greedy, order-dependent |
| **WordPiece** | Merge by likelihood increase | Used in BERT | Slightly more complex training |
| **Unigram** | Start large, prune by likelihood | Probabilistic | Used in SentencePiece |

**BPE Algorithm**:
```
1. Start with character-level vocabulary
2. Count all adjacent pairs in corpus
3. Merge the most frequent pair → new token
4. Repeat steps 2-3 for N merges
```

**Special Tokens**: `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`, `<|endoftext|>`

---

## Word Embeddings

| Method | Objective | Key Idea |
|---|---|---|
| **Skip-gram** | $\max \sum_{(w_c, w_o)} \log P(w_o \mid w_c)$ | Predict context from center |
| **CBOW** | $\max \sum \log P(w_c \mid w_{o_1}, \dots, w_{o_k})$ | Predict center from context |
| **GloVe** | $J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$ | Factorize co-occurrence matrix |

**Skip-gram probability (softmax)**:
$$P(w_o \mid w_c) = \frac{\exp(u_{w_o}^T v_{w_c})}{\sum_{w \in V} \exp(u_w^T v_{w_c})}$$

**Negative sampling approximation**:
$$\log \sigma(u_{w_o}^T v_{w_c}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n} [\log \sigma(-u_{w_k}^T v_{w_c})]$$

**Embedding arithmetic**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$

**Cosine similarity**: $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$

---

## GloVe Details

| Item | Detail |
|---|---|
| Co-occurrence matrix | $X_{ij}$ = count of word $j$ in context of word $i$ |
| Weighting function | $f(x) = \min\left(\left(\frac{x}{x_{\max}}\right)^{0.75}, 1\right)$ |
| Objective | Weighted least squares on log co-occurrence |
| Embedding | Final embedding = $w_i + \tilde{w}_i$ (sum of both vectors) |

---

## Transformer Building Blocks (NLP Focus)

**Self-Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Component | Shape | Notes |
|---|---|---|
| Token IDs | $(B, L)$ | Integer indices into vocabulary |
| Token embeddings | $(B, L, D)$ | Looked up from embedding table |
| Position embeddings | $(B, L, D)$ or $(L, D)$ | Added to token embeddings |
| Attention scores | $(B, H, L, L)$ | Before/after softmax |
| Attention mask (BERT) | $(L, L)$ all ones | Full bidirectional attention |
| Attention mask (GPT) | $(L, L)$ lower triangular | Causal (left-to-right only) |

---

## BERT (Encoder-Only)

| Item | Detail |
|---|---|
| Architecture | Transformer encoder stack |
| Attention | Bidirectional (full attention mask) |
| Pretraining | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |
| MLM | Mask 15% of tokens: 80% `[MASK]`, 10% random, 10% unchanged |
| Input format | `[CLS] sentence_A [SEP] sentence_B [SEP]` |
| Classification | Use `[CLS]` token's hidden state → linear head |
| Token classification | Use each token's hidden state → per-token linear head |
| BERT-base | 12 layers, 768 hidden, 12 heads, 110M params |
| BERT-large | 24 layers, 1024 hidden, 16 heads, 340M params |

---

## GPT (Decoder-Only)

| Item | Detail |
|---|---|
| Architecture | Transformer decoder stack |
| Attention | Causal (lower-triangular mask) |
| Pretraining | Next token prediction (autoregressive) |
| Objective | $\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t})$ |
| Generation | Sample or greedy decode token by token |
| Temperature | $P(w) \propto \exp(z_w / T)$; $T < 1$ sharper, $T > 1$ flatter |
| Top-k | Sample from top $k$ most probable tokens |
| Top-p (nucleus) | Sample from smallest set with cumulative prob $\geq p$ |

---

## Fine-Tuning

| Step | Detail |
|---|---|
| 1. Load pretrained model | BERT/GPT with pretrained weights |
| 2. Add task head | Linear layer(s) on top of hidden states |
| 3. Freeze or unfreeze | Optionally freeze backbone early layers |
| 4. Train on task data | Lower learning rate ($2 \times 10^{-5}$ typical) |
| 5. Evaluate | Task-specific metrics (F1, accuracy, BLEU, etc.) |

**Common tasks**:
- Sequence classification: `[CLS]` → linear → class
- Token classification (NER): each token → linear → label
- Question answering: predict start/end span positions

---

## Common NLP Shapes

```
Vocabulary size:     V (e.g., 30522 for BERT)
Sequence length:     L (e.g., 512 max for BERT)
Hidden dimension:    D (e.g., 768 for BERT-base)
Batch size:          B
Number of heads:     H

Token IDs:           (B, L)         int64
Attention mask:      (B, L)         float32, 0/1
Token embeddings:    (B, L, D)      float32
Attention weights:   (B, H, L, L)   float32
Logits:              (B, L, V)      float32 (language model)
Classification:      (B, C)         float32 (after pooling)
```

---

## Quick Recipes

```
BPE:         count pairs → merge most frequent → repeat N times
Skip-gram:   center word → predict context → negative sampling → SGD
GloVe:       build X_ij → weight by f(X_ij) → least squares on log X_ij
BERT:        tokenize → [CLS] sent [SEP] → encode → [CLS] for classification
GPT:         tokenize → feed prefix → sample next token → append → repeat
Fine-tune:   load pretrained → add head → small LR → train on task data
Cosine sim:  dot(u,v) / (||u|| * ||v||) → [-1, 1]
```
