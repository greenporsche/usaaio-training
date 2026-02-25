# GloVe: Global Vectors for Word Representation

**Prerequisites**: Matrix factorization, least squares optimization, cosine similarity
**USAAIO Relevance**: GloVe provides a complementary view to Word2Vec — instead of predicting context words, it factorizes a global co-occurrence matrix. IOAI problems may ask you to compute co-occurrence statistics, derive the GloVe objective, or perform embedding arithmetic (analogies).

---

## Discovery

### From Local to Global

Word2Vec (Skip-gram, CBOW) learns from **local context windows** — it processes one center-context pair at a time. But there is global information in the corpus that local methods miss.

Consider the words "ice" and "steam". If you look at their co-occurrence patterns with other words:

| | "solid" | "gas" | "water" | "fashion" |
|---|---|---|---|---|
| **ice** | high | low | high | low |
| **steam** | low | high | high | low |

The **ratio** of co-occurrences is more informative than raw counts:

$$\frac{P(\text{solid} \mid \text{ice})}{P(\text{solid} \mid \text{steam})} \gg 1 \quad \text{(ice is solid, steam is not)}$$

$$\frac{P(\text{water} \mid \text{ice})}{P(\text{water} \mid \text{steam})} \approx 1 \quad \text{(both relate to water)}$$

Pennington, Socher, and Manning (2014) designed **GloVe** to capture exactly these ratio patterns.

> **Socratic question**: Why might ratios of co-occurrence probabilities be more meaningful than the raw probabilities themselves?

### Word2Vec vs. GloVe

| | Word2Vec | GloVe |
|---|---|---|
| Input | Local context windows | Global co-occurrence matrix |
| Training | Online (SGD on pairs) | Batch (weighted least squares) |
| Objective | Predict context words | Factorize log co-occurrence |
| Theoretical basis | Neural language model | Matrix factorization |

In practice, both produce similar quality embeddings. GloVe is often preferred when you have the memory to build the co-occurrence matrix.

---

## Intuition

### Building the Co-occurrence Matrix

Given a corpus with vocabulary of size $|V|$, the co-occurrence matrix $X$ is $|V| \times |V|$:

```
Corpus: "the cat sat on the mat the cat ate"
Window size: 1

       the  cat  sat  on  mat  ate
the  [  0    2    0    1    1    0 ]
cat  [  2    0    1    0    0    1 ]
sat  [  0    1    0    1    0    0 ]
on   [  1    0    1    0    0    0 ]
mat  [  1    0    0    0    0    0 ]
ate  [  0    1    0    0    0    0 ]
```

$X_{ij}$ = number of times word $j$ appears in the context window of word $i$.

Note: $X$ is **symmetric** (if $j$ appears near $i$, then $i$ appears near $j$).

### The GloVe Insight

GloVe wants the dot product of two word vectors to approximate the log of their co-occurrence:

$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}$$

Why **log** co-occurrence? Because:
1. Co-occurrence counts span many orders of magnitude
2. Log compresses the scale, making optimization easier
3. The resulting vectors capture **ratios** (log turns ratios into differences)

### Weighting: Not All Co-occurrences Are Equal

Very frequent co-occurrences (like "the" with almost everything) should not dominate. Very rare co-occurrences might be noise. GloVe uses a weighting function:

```
f(x)
 1 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     ╱
    ╱
   ╱  f(x) = (x/x_max)^0.75  for x < x_max
  ╱   f(x) = 1                for x >= x_max
 ╱
╱____________________________________ x
0                x_max (=100)
```

---

## Math

### Co-occurrence Probability

Define:
- $X_{ij}$ = number of times word $j$ appears in context of word $i$
- $X_i = \sum_k X_{ik}$ = total count for word $i$
- $P_{ij} = P(j \mid i) = X_{ij} / X_i$ = probability that word $j$ appears in context of word $i$

### Deriving the Objective

GloVe starts from the observation that the **ratio** $P_{ik} / P_{jk}$ encodes semantic relationships. We want:

$$F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}$$

Through a series of constraints (linearity, symmetry, homomorphism), Pennington et al. show that the natural solution is:

$$w_i^T \tilde{w}_k = \log P_{ik} = \log X_{ik} - \log X_i$$

Absorbing $\log X_i$ into bias terms $b_i$ and $\tilde{b}_k$:

$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij}$$

### GloVe Objective Function

$$J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

where the weighting function is:

$$f(x) = \begin{cases} (x / x_{\max})^{\alpha} & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

Typical values: $x_{\max} = 100$, $\alpha = 0.75$.

### Gradients

For word vector $w_i$:
$$\frac{\partial J}{\partial w_i} = \sum_{j=1}^{|V|} f(X_{ij}) \cdot 2\left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right) \cdot \tilde{w}_j$$

For context vector $\tilde{w}_j$:
$$\frac{\partial J}{\partial \tilde{w}_j} = \sum_{i=1}^{|V|} f(X_{ij}) \cdot 2\left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right) \cdot w_i$$

For biases:
$$\frac{\partial J}{\partial b_i} = \sum_{j=1}^{|V|} f(X_{ij}) \cdot 2\left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)$$

### Final Embeddings

After training, the final word embedding is the **sum** of both vectors:

$$e_i = w_i + \tilde{w}_i$$

This works because $w$ and $\tilde{w}$ are symmetric in the objective (the co-occurrence matrix is symmetric), so averaging/summing them produces a better embedding than using either alone.

---

## Code

### Building a Co-occurrence Matrix

```python
import numpy as np
from collections import Counter, defaultdict

def build_cooccurrence_matrix(corpus: list[str], vocab: dict[str, int],
                               window_size: int = 5) -> np.ndarray:
    """
    Build a co-occurrence matrix from a tokenized corpus.

    Args:
        corpus: List of tokens (words)
        vocab: Word to index mapping
        window_size: Context window size on each side

    Returns:
        X: (V, V) co-occurrence count matrix
    """
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.float64)

    for i, word in enumerate(corpus):
        if word not in vocab:
            continue
        w_idx = vocab[word]

        # Look at context window
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)

        for j in range(start, end):
            if j == i or corpus[j] not in vocab:
                continue
            c_idx = vocab[corpus[j]]
            # Weight by inverse distance (closer words count more)
            distance = abs(i - j)
            X[w_idx, c_idx] += 1.0 / distance

    return X
```

### GloVe Training in PyTorch

```python
import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, x_max: float = 100.0, alpha: float = 0.75):
        super().__init__()
        self.x_max = x_max
        self.alpha = alpha

        self.w = nn.Embedding(vocab_size, embed_dim)
        self.w_tilde = nn.Embedding(vocab_size, embed_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)

        # Initialize
        nn.init.uniform_(self.w.weight, -1.0, 1.0)
        nn.init.uniform_(self.w_tilde.weight, -1.0, 1.0)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b_tilde.weight)

    def _weight_fn(self, x):
        """Weighting function f(x)."""
        return torch.clamp((x / self.x_max) ** self.alpha, max=1.0)

    def forward(self, i_indices, j_indices, x_ij):
        """
        i_indices: (B,) word indices
        j_indices: (B,) context word indices
        x_ij:      (B,) co-occurrence counts

        Returns: scalar loss
        """
        w_i = self.w(i_indices)           # (B, D)
        w_j = self.w_tilde(j_indices)     # (B, D)
        b_i = self.b(i_indices).squeeze()  # (B,)
        b_j = self.b_tilde(j_indices).squeeze()  # (B,)

        # Dot product + biases
        dot = torch.sum(w_i * w_j, dim=1)  # (B,)
        prediction = dot + b_i + b_j

        # Target: log(x_ij)
        target = torch.log(x_ij + 1e-10)

        # Weighted squared error
        weights = self._weight_fn(x_ij)
        loss = torch.mean(weights * (prediction - target) ** 2)

        return loss

    def get_embeddings(self):
        """Return final embeddings (sum of w and w_tilde)."""
        return self.w.weight.data + self.w_tilde.weight.data
```

### Embedding Arithmetic (Analogy Tasks)

```python
def analogy(embeddings, vocab, idx_to_word, a, b, c, top_k=5):
    """
    Solve: a is to b as c is to ?
    Vector arithmetic: result = b - a + c
    """
    vec = embeddings[vocab[b]] - embeddings[vocab[a]] + embeddings[vocab[c]]

    # Compute cosine similarity with all words
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)

    similarities = normalized @ vec_norm

    # Exclude input words
    exclude = {vocab[a], vocab[b], vocab[c]}
    for idx in exclude:
        similarities[idx] = -np.inf

    # Top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(idx_to_word[i], similarities[i]) for i in top_indices]

# Example: king - man + woman = ?
# analogy(emb, vocab, idx2word, "man", "king", "woman")
# → [("queen", 0.89), ...]
```

---

## Key Takeaways

1. **GloVe** learns embeddings by factorizing the log co-occurrence matrix — a global method
2. The **ratio** of co-occurrence probabilities captures semantic relationships better than raw counts
3. The weighting function $f(X_{ij})$ prevents frequent pairs from dominating the objective
4. Final embeddings are the **sum** $w_i + \tilde{w}_i$ of both learned vector sets
5. **Embedding arithmetic** works: vector differences encode analogical relationships
6. GloVe and Word2Vec produce similar quality embeddings but from different perspectives (global vs. local)
7. On USAAIO exams, practice computing co-occurrence matrices by hand and tracing the analogy arithmetic
