# Unit 12: AI Grandmaster (AI 900) — Overview

## Course Position

This is the capstone unit of the USAAIO 2026 training curriculum. It sits at the 900 level, beyond all 100–500 courses. Completing this unit means you are prepared for:

- **USAAIO Round 2** — the implementation-heavy final round
- **National training camp** selection
- **IOAI / IAIO** international competition

## Prerequisites

You must have mastered:

- **Unit 1–4:** Python, NumPy, data processing, classical ML
- **Unit 5–7:** Deep learning fundamentals, CNNs, sequence models
- **Unit 8–9:** Transformers, attention mechanisms, modern architectures
- **Unit 10–11:** Generative models, advanced training techniques

If any of these areas feel shaky, revisit them before proceeding. This unit assumes fluency.

## What Makes This Unit Different

Previous units taught you **specific methods**. This unit teaches you **how to learn and implement new methods on the fly** — which is exactly what Round 2 tests.

In Round 2, you will encounter methods you have **never seen before**. The exam will teach you the method through a paper-style description, then ask you to:

1. Prove mathematical properties
2. Implement the method in PyTorch
3. Train and evaluate it
4. Analyze its behavior

Success requires **speed of absorption**, not just depth of knowledge.

## Unit Structure

### Study Guide Topics

| # | Topic | Why It Matters |
|---|-------|----------------|
| 01 | Physics-Informed Neural Networks | Direct Round 2 topic (2025 P1 was PINNs) |
| 02 | Advanced Optimization | L-BFGS, meta-learning, curriculum learning |
| 03 | Paper-to-Implementation | THE meta-skill for Round 2 |
| 04 | Novel Architectures | Implement unfamiliar architectures from descriptions |
| 05 | Competition Strategy | Time management, partial credit, exam technique |

### Assignments

10 Jupyter notebook assignments at actual Round 2 difficulty:

| # | Topic | Focus |
|---|-------|-------|
| 01 | PINN Heat Equation | Mirrors 2025 Round 2 P1 |
| 02 | PINN Wave Equation | New PDE, same PINN framework |
| 03 | Novel Attention Variant | Paper-to-implementation |
| 04 | Contrastive Method | Paper-to-implementation |
| 05 | Higher-Order Autograd | Advanced derivative computation |
| 06 | Mixed-Precision & Efficiency | FP16, gradient accumulation |
| 07 | Transformer Variant (Round 2-style) | Full exam simulation |
| 08 | Generative Model (Round 2-style) | Full exam simulation |
| 09 | Advanced Application (Round 2-style) | Multi-technique combination |
| 10 | Timed Challenge | Full Round 2 mock exam |

## Study Approach

### Recommended Order

1. Read `01-pinns.md` thoroughly — this is the most directly relevant topic
2. Work through `assignment-01` (heat equation PINN) — build muscle memory
3. Read `03-paper-to-implementation.md` — develop the meta-skill
4. Work through assignments 03–04 — practice paper-to-implementation
5. Read remaining study guides
6. Work through assignments 05–09
7. Take `assignment-10` as a timed mock exam under real conditions

### Time Commitment

- Study guides: ~8 hours
- Exercises: ~4 hours
- Assignments 01–09: ~12 hours
- Assignment 10 (timed): 4 hours
- **Total: ~28 hours** (exceeds the 20-hour course estimate — that is intentional for a capstone)

## Key Resources

- Raissi, Perdikaris, Karniadakis (2019). "Physics-informed neural networks." *Journal of Computational Physics*.
- PyTorch `torch.autograd.grad` documentation
- 2025 USAAIO Round 2 problems (especially Problem 1)
