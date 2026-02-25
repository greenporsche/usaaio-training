# Word Embeddings

**Prerequisites**: Matrix multiplication, gradient descent, softmax, cross-entropy loss
**USAAIO Relevance**: Word embeddings are a foundational concept — they show how neural networks learn distributed representations. Skip-gram and CBOW appear in IOAI problems as both theory (derive the gradient) and implementation (train Word2Vec from scratch in PyTorch).

---

## Discovery

### The Distributional Hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

Before neural embeddings, NLP used **one-hot vectors**: each word is a basis vector in $\mathbb{R}^{|V|}$. The problem:
- "cat" and "dog" are **equidistant** from each other as "cat" and "democracy"
- $\cos(\text{one-hot}(\text{cat}), \text{one-hot}(\text{dog})) = 0$
- The representation encodes **no semantic information**

Mikolov et al. (2013) changed everything with **Word2Vec**: learn a dense vector for each word such that words with similar contexts have similar vectors.

> **Socratic question**: If two words appear in similar contexts ("The ___ chased the mouse" → cat/dog), should their vectors be similar? What does "similar context" actually mean?

### The Word2Vec Revolution

| Approach | Representation | Dimension | Semantic info |
|---|---|---|---|
| One-hot | Sparse | $|V|$ (e.g., 50,000) | None |
| Word2Vec | Dense | $d$ (e.g., 300) | Rich |

The key insight: by training a simple neural network to predict context words, the **hidden layer weights** become meaningful word representations.

---

## Intuition

### Skip-gram: Predict Context from Center

Given a center word, predict which words appear nearby:

```
Sentence:  "the cat sat on the mat"
Window = 2

Center: "sat"
Context: {"cat", "on"}  (within 2 positions)

Training pairs:
  (sat, cat), (sat, on)

The model learns: P(cat | sat) should be high
                  P(democracy | sat) should be low
```

Visually:

```
         ┌───────── context window ──────────┐
         │                                    │
    the  cat  sat  on  the  mat
              ↑ center
         ←2→      ←2→
```

### CBOW: Predict Center from Context

The reverse: given surrounding words, predict the center:

```
Context: {"the", "cat", "on", "the"}  →  predict "sat"

CBOW averages context vectors and predicts the center word.
```

```
Skip-gram:                    CBOW:

   center → context           context → center
   "sat"  → "cat"            {"cat","on"} → "sat"
   "sat"  → "on"

Better for rare words         Better for frequent words
Slower to train               Faster to train
```

### The Neural Network View

Skip-gram is a simple two-layer network:

```
Input          Hidden           Output
(one-hot)      (embedding)      (softmax)

  w_c          v_c              P(w_1|w_c)
  |V|    →      d        →     P(w_2|w_c)
[0,0,1,..]   [0.2,0.8,..]      ...
                                P(w_|V||w_c)

  W_in          W_out
  (V × d)      (d × V)
```

After training, $W_{\text{in}}$ **is** the embedding matrix. Row $i$ of $W_{\text{in}}$ is the embedding of word $i$.

---

## Math

### Skip-gram Objective

Given a corpus of words $w_1, w_2, \dots, w_T$ and a context window of size $c$, maximize:

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

where:

$$P(w_o \mid w_c) = \frac{\exp(u_{w_o}^T v_{w_c})}{\sum_{w=1}^{|V|} \exp(u_w^T v_{w_c})}$$

- $v_{w_c} \in \mathbb{R}^d$ = center word embedding (row of $W_{\text{in}}$)
- $u_{w_o} \in \mathbb{R}^d$ = context word embedding (row of $W_{\text{out}}$)

### The Computational Problem

Computing the full softmax requires summing over the **entire vocabulary** for each training example:

$$\nabla_{v_{w_c}} \log P(w_o \mid w_c) = u_{w_o} - \sum_{w=1}^{|V|} P(w \mid w_c) \cdot u_w$$

With $|V| = 50{,}000+$, this is extremely expensive.

### Negative Sampling

**Solution**: Instead of computing the full softmax, approximate it with a binary classification task.

For each positive pair $(w_c, w_o)$, sample $K$ negative words $w_1^-, \dots, w_K^-$ from a noise distribution $P_n(w)$:

$$J_{\text{neg}} = \log \sigma(u_{w_o}^T v_{w_c}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n} \left[\log \sigma(-u_{w_k}^T v_{w_c})\right]$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

**Noise distribution**: $P_n(w) = \frac{\text{count}(w)^{3/4}}{Z}$ — the 3/4 power smooths the distribution, giving rare words slightly higher probability of being sampled.

**Gradients**:

For the center word embedding:
$$\nabla_{v_{w_c}} J_{\text{neg}} = (1 - \sigma(u_{w_o}^T v_{w_c})) \cdot u_{w_o} - \sum_{k=1}^{K} \sigma(u_{w_k}^T v_{w_c}) \cdot u_{w_k}$$

For the positive context word:
$$\nabla_{u_{w_o}} J_{\text{neg}} = (1 - \sigma(u_{w_o}^T v_{w_c})) \cdot v_{w_c}$$

For each negative sample:
$$\nabla_{u_{w_k}} J_{\text{neg}} = -\sigma(u_{w_k}^T v_{w_c}) \cdot v_{w_c}$$

### CBOW Objective

Given context words $w_{c-m}, \dots, w_{c-1}, w_{c+1}, \dots, w_{c+m}$, predict center word $w_c$:

$$P(w_c \mid \text{context}) = \frac{\exp(u_{w_c}^T \bar{v})}{\sum_{w=1}^{|V|} \exp(u_w^T \bar{v})}$$

where $\bar{v} = \frac{1}{2m} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} v_{w_{c+j}}$ is the average of context word embeddings.

---

## Code

### Skip-gram with Negative Sampling in PyTorch

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Initialize with small random values
        nn.init.uniform_(self.center_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center_ids, context_ids, negative_ids):
        """
        center_ids:   (B,) center word indices
        context_ids:  (B,) positive context word indices
        negative_ids: (B, K) negative sample indices

        Returns: scalar loss
        """
        # Shape: (B, D)
        v_center = self.center_embeddings(center_ids)
        u_context = self.context_embeddings(context_ids)
        u_negative = self.context_embeddings(negative_ids)  # (B, K, D)

        # Positive score: (B,)
        pos_score = torch.sum(v_center * u_context, dim=1)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score) + 1e-10))

        # Negative scores: (B, K)
        neg_score = torch.bmm(u_negative, v_center.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.mean(torch.sum(
            torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1
        ))

        return pos_loss + neg_loss
```

### Training Loop Sketch

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SkipGramDataset(Dataset):
    def __init__(self, corpus_ids, window_size=5, num_negatives=5, vocab_size=10000):
        self.pairs = []
        self.num_negatives = num_negatives
        self.vocab_size = vocab_size

        # Generate (center, context) pairs
        for i, center in enumerate(corpus_ids):
            for j in range(max(0, i - window_size), min(len(corpus_ids), i + window_size + 1)):
                if j != i:
                    self.pairs.append((center, corpus_ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negatives = torch.randint(0, self.vocab_size, (self.num_negatives,))
        return (
            torch.tensor(center),
            torch.tensor(context),
            negatives
        )

# Training
model = SkipGram(vocab_size=10000, embed_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for center, context, negatives in dataloader:
        loss = model(center, context, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Using Pretrained Embeddings (Gensim)

```python
import gensim.downloader as api

# Load pretrained Word2Vec
model = api.load("word2vec-google-news-300")

# Find similar words
model.most_similar("king")
# [('kings', 0.71), ('queen', 0.65), ('monarch', 0.64), ...]

# Analogy: king - man + woman = ?
model.most_similar(positive=["king", "woman"], negative=["man"])
# [('queen', 0.71), ...]

# Similarity
model.similarity("cat", "dog")  # ~0.76
model.similarity("cat", "car")  # ~0.20
```

---

## Key Takeaways

1. **One-hot vectors** encode no semantic similarity; all words are equidistant
2. **Skip-gram** learns word embeddings by predicting context from center words
3. **CBOW** does the reverse: predicts center from context (faster, worse on rare words)
4. **Negative sampling** makes training tractable by replacing the full softmax with binary classification against $K$ random negative words
5. The noise distribution $P_n(w) \propto \text{count}(w)^{3/4}$ smooths sampling of rare words
6. After training, **similar words have similar vectors** (high cosine similarity)
7. Embeddings capture **linear relationships**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$
