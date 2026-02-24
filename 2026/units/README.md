# USAAIO 2026 Training Units

A structured 12-unit curriculum for preparing for the [USA AI Olympiad (USAAIO)](https://www.usaaio.org/) and the [International Olympiad in AI (IOAI)](https://ioai-official.org/). Units are sequenced so that each builds on the knowledge from previous ones, progressing from foundational skills through advanced deep learning and generative AI.

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

## Structure

Each unit folder contains a markdown file with:

- **Unit overview** — what the unit covers and why
- **Topics** — detailed topic list aligned with the [USAAIO syllabus](https://www.usaaio.org/syllabus)
- **Course contents** — breakdown of the corresponding BeaverEdge course
- **FAQs** — common questions about prerequisites and scope
- **Takeaways** — expected outcomes after completing the unit
- **Resources** — curated links to textbooks, courses, and tutorials
