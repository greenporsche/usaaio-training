# GloVe Exercises

**Topic**: Co-occurrence matrices, GloVe objective, embedding arithmetic, analogy tasks
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: Building a Co-occurrence Matrix

Given the corpus: `"I like deep learning I like NLP"` with window size 1.

Vocabulary: `{I: 0, like: 1, deep: 2, learning: 3, NLP: 4}`

**Tasks**:
1. Build the co-occurrence matrix $X$ (5x5). Count each (word, context) pair within the window.
2. Is the matrix symmetric? Why or why not?
3. Compute $X_0 = \sum_k X_{0k}$ (total count for word "I").
4. Compute $P(j \mid i=\text{like})$ for all words $j$.
5. If the window size were 2 instead of 1, which entries would change?

<details>
<summary>Solution</summary>

**1.** Corpus: `I(0) like(1) deep(2) learning(3) I(4) like(5) NLP(6)`

With window 1, adjacent pairs (bidirectional):
- (I, like): positions (0,1) and (4,5) → count 2
- (like, deep): position (1,2) → count 1
- (deep, learning): position (2,3) → count 1
- (learning, I): position (3,4) → count 1
- (like, NLP): position (5,6) → count 1

Co-occurrence matrix:
```
       I  like  deep  learn  NLP
I    [ 0    2     0     1     0 ]
like [ 2    0     1     0     1 ]
deep [ 0    1     0     1     0 ]
learn[ 1    0     1     0     0 ]
NLP  [ 0    1     0     0     0 ]
```

**2.** Yes, the matrix is symmetric: $X_{ij} = X_{ji}$. If word $j$ is in the context of word $i$, then word $i$ is also in the context of word $j$ (for symmetric windows).

**3.** $X_0 = 0 + 2 + 0 + 1 + 0 = 3$

**4.** $P(j \mid \text{like}) = X_{1j} / X_1$ where $X_1 = 2 + 0 + 1 + 0 + 1 = 4$
- $P(\text{I} \mid \text{like}) = 2/4 = 0.5$
- $P(\text{like} \mid \text{like}) = 0/4 = 0$
- $P(\text{deep} \mid \text{like}) = 1/4 = 0.25$
- $P(\text{learning} \mid \text{like}) = 0/4 = 0$
- $P(\text{NLP} \mid \text{like}) = 1/4 = 0.25$

**5.** With window 2, additional pairs would include:
- (I, deep): position 0 to 2, and position 4 to 6 (but 6 is NLP)
- (like, learning): position 1 to 3
- (deep, I): position 2 to 4
- (learning, like): position 3 to 5
- (I, NLP): position 4 to 6

Many more entries would become nonzero.

</details>

---

## Exercise 2: GloVe Weighting Function

The GloVe weighting function is:
$$f(x) = \begin{cases} (x / x_{\max})^{0.75} & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

with $x_{\max} = 100$.

**Tasks**:
1. Compute $f(x)$ for $x = 1, 10, 50, 100, 200$.
2. Plot (or sketch) the function. What does it look like?
3. Why use $\alpha = 0.75$ instead of $\alpha = 1$ (linear)?
4. What would happen if we did not use a weighting function at all (i.e., $f(x) = 1$ for all $x$)?

<details>
<summary>Solution</summary>

**1.**
- $f(1) = (1/100)^{0.75} = 0.01^{0.75} = 0.0178$
- $f(10) = (10/100)^{0.75} = 0.1^{0.75} = 0.1778$
- $f(50) = (50/100)^{0.75} = 0.5^{0.75} = 0.5946$
- $f(100) = 1$ (capped)
- $f(200) = 1$ (capped)

**2.** The function is a concave curve from 0 to 1, rising quickly at first and then leveling off, capped at 1 for $x \geq 100$:
```
f(x)
1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                        ╱───────────────────
                      ╱
                   ╱
                 ╱
              ╱
           ╱
        ╱
     ╱
   ╱
0 ────────────────────────────────────────── x
  0    20    40    60    80   100  120  140
```

**3.** With $\alpha = 1$ (linear), $f(x) = x/x_{\max}$. This gives too much weight to very frequent pairs. The 0.75 exponent creates a concave (sublinear) curve that still differentiates between rare and common pairs but does not let the most frequent pairs dominate. The pair ("the", "is") with count 10,000 would get weight 1.0 either way (capped), but a pair with count 50 gets weight 0.59 instead of 0.50 — the sublinear scaling is more generous to moderately frequent pairs.

**4.** Without weighting ($f(x) = 1$):
- Very frequent pairs (function words like "the" + "is") would dominate the loss
- Rare but informative pairs (content words) would be overshadowed
- The model would spend most of its capacity fitting the most common (and least informative) co-occurrences
- Result: embeddings biased toward function word relationships rather than semantic content

</details>

---

## Exercise 3: GloVe Objective Computation

Using the co-occurrence matrix from Exercise 1 and $x_{\max} = 10$, $\alpha = 0.75$.

Suppose after some training, the parameters are:
```
w_I       = [0.5, 0.3]     w̃_I       = [0.4, 0.2]     b_I = 0.1     b̃_I = 0.1
w_like    = [0.8, -0.1]    w̃_like    = [0.7, 0.0]     b_like = 0.2  b̃_like = 0.1
w_deep    = [0.3, 0.6]     w̃_deep    = [0.2, 0.5]     b_deep = 0.0  b̃_deep = 0.1
```

**Tasks**:
1. Compute the contribution to the GloVe loss from the pair (I, like) where $X_{01} = 2$.
2. Compute the contribution from the pair (like, deep) where $X_{12} = 1$.
3. Which pair contributes more to the total loss?
4. Compute $\frac{\partial J_{\text{pair}}}{\partial w_I}$ for the (I, like) pair.

<details>
<summary>Solution</summary>

**1.** For pair (I, like), $X_{01} = 2$:

$f(2) = (2/10)^{0.75} = 0.2^{0.75} = 0.2990$

$w_I^T \tilde{w}_{\text{like}} = 0.5 \times 0.7 + 0.3 \times 0.0 = 0.35$

Prediction: $w_I^T \tilde{w}_{\text{like}} + b_I + \tilde{b}_{\text{like}} = 0.35 + 0.1 + 0.1 = 0.55$

Target: $\log(X_{01}) = \log(2) = 0.6931$

Loss contribution: $f(2) \times (0.55 - 0.6931)^2 = 0.2990 \times (-0.1431)^2 = 0.2990 \times 0.0205 = 0.00613$

**2.** For pair (like, deep), $X_{12} = 1$:

$f(1) = (1/10)^{0.75} = 0.1^{0.75} = 0.1778$

$w_{\text{like}}^T \tilde{w}_{\text{deep}} = 0.8 \times 0.2 + (-0.1) \times 0.5 = 0.16 - 0.05 = 0.11$

Prediction: $0.11 + 0.2 + 0.1 = 0.41$

Target: $\log(1) = 0$

Loss contribution: $0.1778 \times (0.41 - 0)^2 = 0.1778 \times 0.1681 = 0.02989$

**3.** The (like, deep) pair contributes more (0.02989 vs 0.00613) despite having a smaller co-occurrence count. This is because the prediction error is larger (0.41 vs 0.14), and the squared error grows quadratically.

**4.** Gradient for (I, like):
$$\frac{\partial J_{\text{pair}}}{\partial w_I} = f(X_{01}) \cdot 2(w_I^T \tilde{w}_{\text{like}} + b_I + \tilde{b}_{\text{like}} - \log X_{01}) \cdot \tilde{w}_{\text{like}}$$

$= 0.2990 \times 2 \times (-0.1431) \times [0.7, 0.0]$

$= 0.2990 \times (-0.2862) \times [0.7, 0.0]$

$= -0.0856 \times [0.7, 0.0]$

$= [-0.0599, 0]$

The gradient points in the negative direction of $\tilde{w}_{\text{like}}$, scaled by the error. Since the prediction undershoots the target ($0.55 < 0.69$), the gradient will push $w_I$ to increase its dot product with $\tilde{w}_{\text{like}}$.

</details>

---

## Exercise 4: Embedding Arithmetic and Analogies

Given pretrained GloVe embeddings (3-dimensional for simplicity):
```
king   = [1.0, 0.8, 0.3]
queen  = [0.9, 0.2, 0.7]
man    = [0.7, 0.9, 0.1]
woman  = [0.6, 0.3, 0.6]
prince = [0.8, 0.7, 0.2]
```

**Tasks**:
1. Compute $v = \text{king} - \text{man} + \text{woman}$. Which word is it closest to (by cosine similarity)?
2. Compute $v = \text{queen} - \text{woman} + \text{man}$. What should this return?
3. What does the vector $\text{king} - \text{man}$ represent conceptually? Compute it.
4. What does the vector $\text{queen} - \text{king}$ represent? Compute it.
5. Compare $\text{king} - \text{man}$ with $\text{queen} - \text{woman}$. Are they similar?

<details>
<summary>Solution</summary>

**1.** $v = [1.0, 0.8, 0.3] - [0.7, 0.9, 0.1] + [0.6, 0.3, 0.6] = [0.9, 0.2, 0.8]$

Cosine similarities with candidates:
- $\cos(v, \text{queen}) = \cos([0.9, 0.2, 0.8], [0.9, 0.2, 0.7])$

$= \frac{0.81 + 0.04 + 0.56}{\sqrt{0.81+0.04+0.64} \times \sqrt{0.81+0.04+0.49}} = \frac{1.41}{\sqrt{1.49} \times \sqrt{1.34}} = \frac{1.41}{1.221 \times 1.158} = \frac{1.41}{1.414} = 0.997$

- $\cos(v, \text{prince}) = \cos([0.9, 0.2, 0.8], [0.8, 0.7, 0.2])$

$= \frac{0.72+0.14+0.16}{\sqrt{1.49} \times \sqrt{0.64+0.49+0.04}} = \frac{1.02}{1.221 \times 1.082} = \frac{1.02}{1.321} = 0.772$

Closest to **queen** (0.997). The analogy works: king - man + woman = queen.

**2.** $v = [0.9, 0.2, 0.7] - [0.6, 0.3, 0.6] + [0.7, 0.9, 0.1] = [1.0, 0.8, 0.2]$

This should return **king** (the reverse analogy). Compare with king = [1.0, 0.8, 0.3] — very close.

**3.** $\text{king} - \text{man} = [0.3, -0.1, 0.2]$ — represents the "royalty" or "status/title" direction, abstracting away gender.

**4.** $\text{queen} - \text{king} = [-0.1, -0.6, 0.4]$ — represents the "female-male" or gender direction within the royalty context.

**5.** $\text{king} - \text{man} = [0.3, -0.1, 0.2]$ and $\text{queen} - \text{woman} = [0.3, -0.1, 0.1]$.

These are very similar, showing that the "royalty" direction is approximately the same regardless of gender. This is exactly the linear structure that GloVe captures.

</details>

---

## Exercise 5: Comparing Word2Vec and GloVe

**Tasks** (analytical, no computation):
1. Word2Vec Skip-gram processes one (center, context) pair at a time. GloVe processes co-occurrence counts. Explain why GloVe might converge faster on a large corpus.
2. Both methods produce two sets of vectors (center/context for Word2Vec, $w/\tilde{w}$ for GloVe). Why does GloVe recommend summing $w + \tilde{w}$ as the final embedding?
3. A corpus has 1 billion tokens and a vocabulary of 400,000 words. What is the shape of the GloVe co-occurrence matrix? Estimate its memory requirements for float32. Is this practical?
4. Levy and Goldberg (2014) showed that Word2Vec's Skip-gram with negative sampling implicitly factorizes a shifted PMI (pointwise mutual information) matrix. Explain intuitively why this connects Word2Vec to GloVe.
5. In what practical scenario would you prefer GloVe over Word2Vec, and vice versa?

<details>
<summary>Solution</summary>

**1.** GloVe pre-computes co-occurrence statistics, converting the corpus into a matrix of counts. Training then iterates over non-zero entries of this matrix (which is much smaller than the corpus). Word2Vec must process every token occurrence separately. For a corpus of 1B tokens with a co-occurrence matrix having ~100M non-zero entries, GloVe sees each "data point" once per epoch over the matrix, while Word2Vec processes 1B training samples per epoch.

**2.** The GloVe objective treats $w_i$ and $\tilde{w}_j$ symmetrically — the co-occurrence matrix $X$ is symmetric, so $w$ and $\tilde{w}$ are interchangeable. Using only one set wastes information. Summing $w + \tilde{w}$ averages the two perspectives (word-as-focus and word-as-context), reducing noise and producing a better embedding.

**3.** Shape: $(400{,}000 \times 400{,}000)$. Memory: $400{,}000^2 \times 4$ bytes = $6.4 \times 10^{11}$ bytes = 640 GB. This is **not practical** as a dense matrix. In practice, the matrix is extremely sparse (most word pairs never co-occur), and GloVe only stores and trains on non-zero entries. The sparse representation typically requires ~1-10 GB.

**4.** PMI measures how much more often two words co-occur than expected: $\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)}$. GloVe's objective makes $w_i^T \tilde{w}_j \approx \log X_{ij}$, which is closely related to $\log P(i,j)$ up to normalization constants (absorbed by biases). So both methods are implicitly factorizing matrices based on co-occurrence statistics — just with different formulations. They are two views of the same underlying mathematical structure.

**5.**
- **Prefer GloVe** when: you have a fixed corpus and want reproducible results; you have enough memory for the co-occurrence matrix; you want fast training (matrix factorization converges quickly).
- **Prefer Word2Vec** when: new data arrives continuously (online/streaming learning); memory is limited; you want simplicity of implementation; you are working with a specialized domain corpus.
- In practice, pretrained GloVe and Word2Vec embeddings perform similarly. The choice often comes down to implementation convenience.

</details>
