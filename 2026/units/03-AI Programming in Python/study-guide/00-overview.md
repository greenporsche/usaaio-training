# Unit 03: AI Programming in Python -- Study Guide Overview

**Unit**: AI 210 -- Coding for AI 1
**USAAIO Relevance**: Every coding question on the USAAIO exam requires these libraries. This unit is the foundation for all subsequent AI implementation work.

---

## What This Unit Covers

This unit transforms you from a Python programmer into an AI programmer. The key shift: **stop thinking in loops, start thinking in arrays**.

Every AI algorithm -- from linear regression to neural networks -- is implemented as matrix operations. NumPy is the engine. Pandas is how you load and prepare data. Matplotlib/Seaborn is how you communicate results. Together, they are the language of AI implementation.

## Study Guide Structure

Each topic follows the **D-I-M-C** pattern:

| Phase | Purpose |
|-------|---------|
| **Discovery** | Why does this exist? What problem does it solve? |
| **Intuition** | Visual understanding, mental models, what can go wrong |
| **Math** | Formal definitions, shape analysis, mathematical foundations |
| **Code** | From-scratch implementations with shape annotations |

## Topics

| # | Topic | Key Skills | USAAIO Weight |
|---|-------|-----------|---------------|
| 01 | Advanced Python | Comprehensions, generators, decorators, lambdas | Medium |
| 02 | NumPy Basics | Array creation, indexing, slicing, shapes | High |
| 03 | NumPy Vectorization | Broadcasting, no-loop patterns, einsum | **Critical** |
| 04 | Pandas | DataFrames, groupby, merge, data manipulation | High |
| 05 | Plotting | Matplotlib, Seaborn, visual communication | Medium |

## How to Use This Guide

1. **Read sequentially** -- each topic builds on the previous
2. **Run every code block** -- type it yourself, do not copy-paste
3. **Do the exercises** -- they mimic USAAIO question formats
4. **Complete assignments** -- timed practice under exam conditions
5. **Use the cheat sheet** -- during practice, not during study

## Prerequisites

- Python fundamentals (variables, lists, dicts, loops, functions)
- Basic math (algebra, simple statistics)
- A working Python 3.11+ environment with NumPy, Pandas, Matplotlib, Seaborn installed

## Estimated Study Time

| Component | Time |
|-----------|------|
| Study guides (5 topics) | 10-15 hours |
| Exercises (30 problems) | 3-5 hours |
| Assignments (10 notebooks) | 10-15 hours |
| **Total** | **23-35 hours** |

---

## Quick Environment Setup

```bash
uv init usaaio-practice && cd usaaio-practice
uv add numpy pandas matplotlib seaborn jupyter
uv run jupyter notebook
```

Or use Google Colab (all libraries pre-installed).
