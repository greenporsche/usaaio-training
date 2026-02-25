# USAAIO 2026 Training Units

A structured 12-unit curriculum for preparing for the [USA AI Olympiad (USAAIO)](https://www.usaaio.org/) and the [International Olympiad in AI (IOAI)](https://ioai-official.org/). Units are sequenced so that each builds on the knowledge from previous ones, progressing from foundational skills through advanced deep learning and generative AI.

> Disclaimer: This repository contains self-authored self-study material and is neither officially affiliated with, endorsed by, nor sourced from official resources of **IOAI**, **USAAIO**, or **BeaverEdge**.

## Unit Overview

| # | Unit | BeaverEdge Course | Key Topics |
|---|------|-------------------|------------|
| 01 | [Markdown Programming for AI](01-Markdown%20programming%20for%20AI/) | AI 100 | Text formatting, writing code & math in Google Colab markdown cells |
| 02 | [Mathematical Foundations for AI](02-Mathematical%20Foundations%20for%20AI/) | AI 200 | Linear algebra, calculus, probability & statistics, convex optimization |
| 03 | [AI Programming in Python](03-AI%20Programming%20in%20Python/) | AI 210 | Advanced Python, NumPy, Pandas, Matplotlib, Seaborn |
| 04 | [ML1: Supervised Learning](04-ML1%20Supervised%20Learning/) | AI 300 | Linear/logistic regression, kernel methods, kNN, cross-validation, bias-variance tradeoff |
| 05 | [ML2: Unsupervised Learning](05-ML2%20Unsupervised%20Learning/) | AI 400 | SVMs, decision trees, random forests, boosting, PCA, t-SNE, UMAP, k-means clustering |
| 06 | [Programming PyTorch](06-Programming%20PyTorch/) | AI 310 | Tensors, autograd, modules, datasets, dataloaders, losses, optimizers |
| 07 | [Deep Learning](07-Deep%20Learning/) | AI 410 | MLPs, backpropagation, CNNs, batch normalization, dropout, ResNet, transfer learning |
| 08 | [Natural Language Processing](08-Natural%20Language%20Processing/) | AI 510 | Tokenization, word embeddings (Skip-gram, CBOW, GloVe), BERT, GPT |
| 09 | [Transformers](09-Transformers/) | AI 500 | Self/cross/masked attention, positional encoding, pre-training, fine-tuning, vision transformers |
| 10 | [Computer Vision & Generative AI](10-Computer%20Vision%20%26%20Generative%20AI/) | AI 520 | Object detection, UNet, autoencoders, GANs, diffusion models, stable diffusion, CLIP |
| 11 | [Graph Neural Networks](11-Graph%20Neural%20Networks/) | AI 510 | Graph convolutional networks, graph attention networks |
| 12 | [AI Grandmaster](12-AI%20Grandmaster/) | AI 900 | Advanced topics beyond all 100-500 level courses |

## Progression Path

```
AI 100  Markdown Programming
  │
  ▼
AI 200  Mathematical Foundations ──────────────────┐
  │                                                │
  ▼                                                │
AI 210  Python Programming ──────────┐             │
  │                                  │             │
  ▼                                  ▼             │
AI 300  Supervised Learning      AI 310  PyTorch ──┤
  │                                  │             │
  ▼                                  │             │
AI 400  Unsupervised Learning        │             │
  │                                  │             │
  └──────────────┬───────────────────┘             │
                 ▼                                 │
           AI 410  Deep Learning ◄─────────────────┘
                 │
                 ▼
           AI 500  Transformers
                 │
                 ▼
           AI 510  NLP & Graph Neural Networks
                 │
                 ▼
           AI 520  Computer Vision & Generative AI
                 │
                 ▼
           AI 900  Grandmaster
```

## Competition Milestones

| Milestone | After Completing |
|-----------|-----------------|
| USAAIO Round 1 — Honor Roll | AI 200 + AI 210 |
| USAAIO Round 1 — High Honor Roll | AI 300 + AI 310 |
| USAAIO Round 1 — Distinguished Honor Roll | AI 400 + AI 410 |
| USAAIO Round 2 — Medal contention | AI 500 + AI 510 + AI 520 |
| National team / IOAI qualification | AI 900 |

## Curriculum Materials

A comprehensive, self-contained curriculum was generated on **2026-02-24** covering all 12 units. The materials use the **Sokratik Discovery-First pedagogy** — each concept is encountered through four cognitive lenses: Discovery, Intuition, Math, and Code.

### Per-Unit Structure

```
units/XX-Name/
├── XX-Name.md                  # Original BeaverEdge course overview (reference)
├── cheat-sheet.md              # Dense 1–2 page exam-day quick reference
├── study-guide/                # Core learning content
│   ├── 00-overview.md          # Unit intro, prereqs, roadmap
│   ├── 01-topic.md             # D→I→M→C for each topic
│   ├── 02-topic.md
│   └── ...
├── exercises/                  # Pencil-and-paper speed drills
│   ├── 01-topic-exercises.md   # 5 exercises per topic, solutions in <details>
│   └── ...
└── assignments/                # Contest-format Jupyter notebooks
    ├── assignment-01-topic.ipynb
    ├── assignment-02-topic.ipynb
    └── ...                     # 3–12 per unit, USAAIO Round 2 format
```

### Study Guide Format (D→I→M→C)

Each study guide file follows four phases:

1. **Discovery** — Historical framing and Socratic questions guiding toward the key insight
2. **Intuition** — Visual/geometric explanations, failure case analysis
3. **Math** — Rigorous formalization with complete derivations
4. **Code** — From-scratch NumPy implementation + PyTorch equivalent with shape annotations

### Assignment Notebook Format

All `.ipynb` notebooks mirror the **official USAAIO Round 2 exam format**:
- Title cell with total points (100 per notebook)
- Import cell with `DO NOT MAKE ANY CHANGE` restriction
- Warning cell about import restrictions
- Multi-part problems: `## Part N (X points, coding/non-coding task)`
- `Reasoning required` / `Reasoning not required` labels
- `### WRITE YOUR SOLUTION HERE ###` and `""" END OF THIS PART """` markers
- Shape annotations required: `# (B, H, L, D_qk)` after tensor operations
- No solutions provided — blank solution cells for students

### Inventory

| # | Unit | Cheat | Guides | Exercises | Notebooks | Total |
|---|------|-------|--------|-----------|-----------|-------|
| 01 | Markdown Programming | 1 | 4 | 3 | 3 | **11** |
| 02 | Mathematical Foundations | 1 | 9 | 8 | 10 | **28** |
| 03 | AI Programming in Python | 1 | 6 | 5 | 10 | **22** |
| 04 | ML1: Supervised Learning | 1 | 8 | 7 | 10 | **26** |
| 05 | ML2: Unsupervised Learning | 1 | 8 | 7 | 10 | **26** |
| 06 | Programming PyTorch | 1 | 8 | 7 | 10 | **26** |
| 07 | Deep Learning | 1 | 9 | 8 | 12 | **30** |
| 08 | Natural Language Processing | 1 | 7 | 6 | 10 | **24** |
| 09 | Transformers | 1 | 9 | 8 | 12 | **30** |
| 10 | CV & Generative AI | 1 | 10 | 9 | 12 | **32** |
| 11 | Graph Neural Networks | 1 | 6 | 5 | 10 | **22** |
| 12 | AI Grandmaster | 1 | 6 | 5 | 10 | **22** |
| | **Totals** | **12** | **90** | **78** | **119** | **299** |

### Round 2 Critical Assignments

The most important notebooks for USAAIO Round 2 preparation:

- **`09-Transformers/assignments/assignment-12-comprehensive.ipynb`** — 14-part, 100-point problem mirroring the 2025 Round 2 Problem 2 (MHA → GQA → MLA progression)
- **`10-CV & Generative AI/assignments/assignment-12-comprehensive.ipynb`** — Full CLIP/diffusion problem in 2025 Round 2 Problem 3 style
- **`07-Deep Learning/assignments/assignment-11-pinn-basics.ipynb`** — PINNs with `autograd.grad(create_graph=True)` matching 2025 Round 2 Problem 1

---

## Original Unit Structure

Each unit folder also contains the original BeaverEdge course overview markdown file with:

- **Unit overview** — what the unit covers and why
- **Topics** — detailed topic list aligned with the [USAAIO syllabus](https://www.usaaio.org/syllabus)
- **Course contents** — breakdown of the corresponding BeaverEdge course
- **FAQs** — common questions about prerequisites and scope
- **Takeaways** — expected outcomes after completing the unit
- **Resources** — curated links to textbooks, courses, and tutorials

---

