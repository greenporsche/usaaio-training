# USAAIO 2026 Round 2 — Preparation Analysis

> What a perfect score demands, based on analysis of 2025 Round 2 problems and the full USAAIO syllabus.

---

## Context

| Fact | Detail |
|------|--------|
| **Round 2 date** | April 4–5, 2026, MIT |
| **Format** | ~3 problems, 4 hours, Jupyter notebooks via Google Colab |
| **Scope** | Units 01–12 (full syllabus) |
| **2025 result** | Gold medal |
| **2026 status** | Requalified for Round 2 via January 30 Round 1 |

### Round 1 vs Round 2: The Structural Difference

Round 1 tests **breadth** — 9 problems in 3 hours across Units 01–07 (300 pts). Round 2 tests **depth** — last year it was 3 problems in 4 hours, each worth 100 points, each with 5–14 parts that build on each other like a guided research paper walkthrough. A single mistake in an early part can cascade through the rest of the problem.

---

## 2025 Round 2 — What Was Actually Tested

The three problems from 2025 Round 2 serve as the best available blueprint for what to expect.

### Problem 1: Physics-Informed Neural Networks (PINNs) — 100 pts

| Aspect | Details |
|--------|---------|
| **Units** | 07 (Deep Learning) + 02 (Math — PDEs/calculus) |
| **Difficulty** | Very Hard |
| **Core idea** | Train a neural network to satisfy a PDE (1D heat equation) instead of fitting data |

**Skills required:**

- Partial differential equations (heat equation: $u_t = \alpha u_{xx}$)
- Boundary conditions and initial conditions as loss terms
- `torch.autograd.grad` with `create_graph=True` for higher-order derivatives
- Custom compound loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{BC}} + \lambda_3 \mathcal{L}_{\text{IC}}$
- Training a neural network to satisfy physics constraints, not data

**What makes it hard:** You need to understand both the math (PDEs, boundary value problems) and the PyTorch mechanics (automatic differentiation for arbitrary-order derivatives). The problem teaches you PINNs during the exam — the skill is absorbing and implementing fast.

---

### Problem 2: Multi-Head Attention & Variants — 100 pts, 14 parts

| Aspect | Details |
|--------|---------|
| **Units** | 09 (Transformers) + 02 (Linear Algebra — SVD, rank) |
| **Difficulty** | Very Hard |
| **Core idea** | Build MHA from scratch, then understand GQA and MLA as variants with mathematical proofs |

**Part-by-part progression:**

| Parts | Task | Type | Points |
|-------|------|------|--------|
| 1–2 | Q/K/V projection matrix shapes | Non-coding | 10 |
| 3 | Scaled dot-product attention formula | Non-coding | 10 |
| 4 | Pre/post out-projection shapes | Non-coding | 5 |
| 5 | **Build MHA from scratch in PyTorch** (no loops) | Coding | 10 |
| 6 | Rank of repeated GQA weight matrix | Non-coding (proof) | 5 |
| 7 | **Build GQA from scratch in PyTorch** (no loops) | Coding | 10 |
| 8 | Prove MHA is special case of GQA | Non-coding | 5 |
| 9 | Prove GQA can be represented as MLA (via SVD) | Non-coding (proof) | 10 |
| 10 | Implement GQA-to-MLA conversion with NumPy | Coding | 5 |
| 11 | Prove GQA ⊊ MLA (counterexample) | Non-coding (proof) | 10 |
| 12 | Derive reduced query/key/value matrices for efficient MLA | Non-coding | 10 |
| 13 | Implement reduced MLA and verify equivalence | Coding | 5 |
| 14 | KV-cache analysis: MHA vs MLA | Non-coding | 5 |

**Skills required:**

- MHA from scratch: `nn.Linear` for projections, tensor reshaping `(B,L,H*D) → (B,H,L,D)`, scaled dot-product attention
- GQA: broadcast K/V heads across query groups using `reshape` + dimension insertion
- MLA: low-rank decomposition of weight matrices via SVD
- Mathematical proofs: rank invariance, GQA⊂MLA, counterexample construction
- KV-cache memory analysis for autoregressive inference

**Key tensor reshaping pattern (must be muscle memory):**

```python
# Project then split into heads
Q = self.W_Q(x)                                              # (B, L, H*D)
Q = Q.reshape(-1, L, self.H, self.D_qk).permute(0, 2, 1, 3) # (B, H, L, D)

# Attention
logits = Q @ K.mT / self.D_qk ** 0.5   # (B, H, L1, L2)
Alpha = F.softmax(logits, dim=-1)        # (B, H, L1, L2)
O = Alpha @ V                            # (B, H, L1, Dv)

# Merge heads
O = O.permute(0, 2, 1, 3).reshape(-1, L, self.H * self.D_v)  # (B, L, H*Dv)
```

**What makes it hard:** 14 parts in ~80 minutes (~6 min/part). The mathematical proofs (rank, SVD, counterexample) require both linear algebra fluency and the ability to construct clean arguments quickly. The coding parts forbid loops, requiring mastery of tensor broadcasting.

---

### Problem 3: CLIP (Contrastive Language-Image Pre-Training) — 100 pts

| Aspect | Details |
|--------|---------|
| **Units** | 10 (CV & GenAI) + 09 (Transformers) |
| **Difficulty** | Hard |
| **Core idea** | Understand and implement multimodal contrastive learning |

**Skills required:**

- Dual-encoder architecture (vision encoder + text encoder)
- Contrastive loss / InfoNCE: $\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$
- Temperature scaling in contrastive objectives
- Zero-shot classification via cosine similarity between image and text embeddings
- Vision Transformers (ViT) as image encoder

**What makes it hard:** You need to understand how two different modalities (images and text) are projected into a shared embedding space and trained with contrastive objectives. The implementation requires careful handling of the similarity matrix and symmetric loss.

---

## Topic Weight Analysis

Based on 2025 Round 2, the topic distribution for Round 2-specific material:

| Category | Weight | Primary Units |
|----------|--------|---------------|
| Attention mechanisms & Transformers | ~35% | Unit 09 |
| Multimodal AI / Contrastive learning | ~30% | Unit 10 |
| Advanced deep learning applications | ~20% | Unit 07 (advanced) |
| Mathematical proofs (SVD, rank, low-rank) | ~15% | Unit 02 (advanced) |

Note: Units 01–07 are still tested implicitly — you need strong PyTorch skills (Unit 06), deep learning fundamentals (Unit 07), and linear algebra fluency (Unit 02) to solve Round 2 problems. But the *new* content is from Units 08–12.

---

## Gap Analysis: Current State vs. Perfect Score

| Area | Current State | Gap | Priority |
|------|---------------|-----|----------|
| Units 01–07 (Round 1) | Strong — Gold medal, 14 practice sets | Maintain, don't grind | — |
| Unit 09: Transformers | Scaffolded, not drilled | MHA/GQA/MLA from scratch | **Critical** |
| Unit 10: CV & GenAI | Scaffolded | CLIP, diffusion models, contrastive learning | **Critical** |
| Unit 08: NLP | Scaffolded | Tokenization, embeddings, BERT/GPT | High |
| Unit 11: GNNs | Scaffolded | Message-passing, GCN, GAT | Medium |
| Unit 07 advanced | Good foundations | PINNs, advanced autograd | High |
| Round 2 practice | `round-2-sample-exams/` is empty | No timed practice at Round 2 difficulty | **Critical** |
| Mathematical proofs | Strong from Round 1 | Extend to SVD/rank/low-rank arguments | High |

---

## Preparation Plan

### Tier 1: Non-Negotiable (~200+ points at stake)

#### 1. Master attention from the ground up (Unit 09)

- [ ] Implement MHA from scratch in PyTorch — no `nn.MultiheadAttention`, no loops
- [ ] Implement GQA — broadcast K/V heads, understand memory savings
- [ ] Understand MLA (DeepSeek) — low-rank decomposition, SVD-based proofs
- [ ] Derive reduced query/key/value matrices for efficient MLA inference
- [ ] Practice the tensor reshaping choreography: `(B,L,H*D) ↔ (B,H,L,D)`
- [ ] Study 2025 Round 2 Problem 2 end-to-end — it's the Rosetta Stone

#### 2. Master contrastive learning & multimodal (Unit 10)

- [ ] Implement CLIP-style contrastive training from scratch
- [ ] InfoNCE / NT-Xent loss derivation and implementation
- [ ] Vision encoder (ViT or ResNet) + text encoder dual architecture
- [ ] Zero-shot classification pipeline via cosine similarity
- [ ] Understand autoencoders, VAEs, GANs, and diffusion model basics
- [ ] Study 2025 Round 2 Problem 3 end-to-end

#### 3. Build Round 2 timed practice

- [ ] Create 2–3 full Round 2 mock exams (3 problems, 4 hours each)
- [ ] Theme each mock around: (a) a transformer-variant problem, (b) a generative/multimodal problem, (c) an advanced application problem
- [ ] Practice under exam conditions: no AI tools, no search engines, screen + face recording

### Tier 2: High-Value (~50–100 points at stake)

#### 4. Advanced autograd & PINNs (Unit 07 extension)

- [ ] `torch.autograd.grad` with `create_graph=True` for higher-order derivatives
- [ ] Implement a PINN for a simple PDE (heat equation, wave equation)
- [ ] Custom compound losses with physics constraints
- [ ] Study 2025 Round 2 Problem 1 end-to-end

#### 5. NLP fundamentals (Unit 08)

- [ ] Tokenization: BPE, WordPiece — implement from scratch
- [ ] Word embeddings: Skip-gram, CBOW — derive objective and implement
- [ ] BERT (encoder-only, masked LM) vs GPT (decoder-only, causal LM)
- [ ] Fine-tuning vs pre-training workflows

#### 6. Graph Neural Networks (Unit 11)

- [ ] Message-passing framework
- [ ] GCN: $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$
- [ ] Graph attention networks (GAT)
- [ ] On syllabus — hasn't appeared in Round 2 yet, but could this year

### Tier 3: Edge Mastery (the difference between 290 and 300)

#### 7. Mathematical proof fluency

- [ ] SVD-based arguments (rank, low-rank approximation)
- [ ] Counterexample construction (e.g., MLA⊋GQA)
- [ ] Matrix rank under concatenation, repetition, multiplication
- [ ] Convex optimization duality (on syllabus, lightly tested so far)

#### 8. Speed and composure under exam pressure

- [ ] Practice 14-part problems — ~6 min per part
- [ ] If stuck on a proof, use the *stated result* to continue coding later parts
- [ ] LaTeX typesetting speed for non-coding tasks
- [ ] Rapid tensor shape annotation after every operation

---

## The Meta-Skill: Paper-to-Implementation Speed

Every Round 2 problem is a guided walkthrough of a published technique:

| 2025 Problem | Based On |
|--------------|----------|
| Problem 1 | Raissi et al., *Physics-Informed Neural Networks* (2019) |
| Problem 2 | Vaswani et al. (2017), Ainslie et al. GQA (2023), DeepSeek-V2 MLA (2024) |
| Problem 3 | Radford et al., *CLIP* (OpenAI, 2021) |

The exam *teaches* you the method in the problem statement. The skill isn't memorizing architectures — it's being fast enough to:

1. **Absorb** a new architecture's math from the problem description
2. **Reason** about its properties (proofs, shape analysis, edge cases)
3. **Implement** it in PyTorch with correct tensor shapes — all within 60–90 minutes

Drilling this meta-skill is arguably more valuable than memorizing any specific architecture.

---

## Key Formulas for Round 2

### Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Contrastive Loss (InfoNCE)

$$\mathcal{L}_i = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k)/\tau)}$$

### PINN Loss

$$\mathcal{L} = \lambda_{\text{PDE}} \|\mathcal{N}[u] - f\|^2 + \lambda_{\text{BC}} \|\mathcal{B}[u] - g\|^2 + \lambda_{\text{IC}} \|u(x,0) - h(x)\|^2$$

### GCN Layer

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

### Low-Rank Decomposition (MLA)

$$\mathbf{W}^{\mathbf{K}} = \mathbf{W}^{\mathbf{UK}} \mathbf{W}^{\mathbf{DKV}}, \quad \mathbf{W}^{\mathbf{V}} = \mathbf{W}^{\mathbf{UV}} \mathbf{W}^{\mathbf{DKV}}$$

KV-cache per position: $r$ (MLA) vs $2D$ (MHA), where $r \ll D$.

---

## Resources

| Topic | Resource |
|-------|----------|
| Attention & Transformers | [UvA DL Tutorial 6](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html) |
| CLIP | [OpenAI CLIP paper](https://arxiv.org/abs/2103.00020), [annotated implementation](https://github.com/openai/CLIP) |
| PINNs | [Raissi et al. 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125) |
| GQA | [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) |
| MLA / DeepSeek-V2 | [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) |
| Diffusion Models | [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| GNNs | Stanford CS224W, [PyG tutorials](https://pytorch-geometric.readthedocs.io/) |
| Computer Vision | [Stanford CS231n](http://cs231n.stanford.edu/) |

---

*Analysis based on 2025 USAAIO Round 2 problems and official syllabus. Last updated: 2026-02-24.*
