# Text Formatting in Markdown

**Prerequisites**: None
**USAAIO Relevance**: Every Round 1 answer must be clearly formatted. Headings organize multi-part solutions. Tables present confusion matrices and comparison results. Lists structure step-by-step derivations.

---

## Discovery

It is 1987. You are John Gruber (well, he would create Markdown later in 2004, but bear with us). You are reading a plain-text email -- the kind people write on Usenet. You notice something: even without any formatting tools, people naturally use conventions to add structure:

```
*this feels emphasized*
THIS FEELS LOUD
- these feel like
- bullet points
```

You wonder: what if we formalized these conventions? What if a simple text file could be automatically converted to a beautifully formatted document, just by following a few rules?

**Think about this**: When you see `**important**` in a text file, your brain already reads it as bold. Why? Because the asterisks visually "wrap" and "squeeze" the word, drawing attention to it.

**Question**: Why do you think `#` was chosen for headings? Think about what `#` means in other contexts (think: numbering, ordering, hierarchy).

**Misconception trap**: Markdown is NOT a programming language. It is a *markup* language -- it describes how text should look, not what the computer should do. You cannot write loops or conditionals in markdown.

---

## Intuition

What you just experienced is exactly the design philosophy behind Markdown: **readable as plain text, beautiful when rendered.**

John Gruber created Markdown in 2004 with one guiding principle: the source should be readable even without rendering. This is why:

- `**bold**` uses double asterisks (they visually "shout")
- `*italic*` uses single asterisks (they visually "lean")
- `# Heading` uses a hash (larger = more important = fewer hashes)
- `- item` uses a dash (it already looks like a bullet)

### The Heading Hierarchy

```
# Heading 1       (Title level -- use once per document)
## Heading 2      (Section level)
### Heading 3     (Subsection)
#### Heading 4    (Sub-subsection)
##### Heading 5   (Rarely needed)
###### Heading 6  (Almost never needed)
```

Think of it like an outline:

```
I.   Heading 1
  A. Heading 2
    1. Heading 3
      a. Heading 4
```

**For USAAIO**: Use `##` for each problem part (8.1, 8.2, ...) and `###` for sub-steps within a part.

### What Goes Wrong Without Good Formatting?

Consider two ways to present the same USAAIO answer:

**Bad formatting:**
```
accuracy equals TP plus TN divided by TP plus TN plus FP plus FN so thats 45 plus 40 divided by 100 which gives 0.85 or 85 percent
```

**Good formatting:**
```markdown
## Part 8.1: Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{45 + 40}{100} = 0.85 = 85\%
$$
```

The second version is instantly readable. The first requires the grader to parse your stream of consciousness. On a timed exam, the grader's patience is your enemy.

---

## Math

This formalizes your understanding of Markdown text formatting syntax.

### Headings

| Syntax | Level | Typical USAAIO Use |
|---|---|---|
| `# Title` | H1 | Document title (one per notebook) |
| `## Section` | H2 | Problem number (`## Problem 8`) |
| `### Subsection` | H3 | Part number (`### Part 8.1`) |
| `#### Sub-subsection` | H4 | Step within a part |

*Reasoning required*: Always use heading levels in order. Never skip from `#` to `###`.

### Emphasis

| Syntax | Result | When to Use |
|---|---|---|
| `**bold**` | **bold** | Key terms, final answers |
| `*italic*` | *italic* | Variable names in prose, emphasis |
| `***bold italic***` | ***bold italic*** | Critical warnings (rare) |
| `~~strikethrough~~` | ~~strikethrough~~ | Corrections, showing wrong approaches |
| `` `code` `` | `code` | Function names, variable names, file names |

### Lists

**Unordered lists** (use `-`, `*`, or `+`):
```markdown
- Step one
- Step two
  - Sub-step 2a
  - Sub-step 2b
    - Deep nesting
```

**Ordered lists**:
```markdown
1. First step
2. Second step
   1. Sub-step
   2. Sub-step
3. Third step
```

*Reasoning required*: Indentation matters. Use 2 or 4 spaces for nesting. Be consistent.

### Links and Images

```markdown
[Link text](https://example.com)
[Link with title](https://example.com "Hover text")

![Alt text](path/to/image.png)
![Alt text](https://url.com/image.png "Caption")
```

### Blockquotes

```markdown
> This is a blockquote.
> It can span multiple lines.
>
> > Nested blockquotes work too.
```

Use blockquotes for problem statements or to highlight important notes.

### Tables

```markdown
| Column A | Column B | Column C |
|:---------|:--------:|---------:|
| Left     | Center   |    Right |
| aligned  | aligned  |  aligned |
```

The colons control alignment:
- `:---` = left (default)
- `:---:` = center
- `---:` = right

**For USAAIO**: Tables are essential for confusion matrices, hyperparameter comparisons, and dataset summaries.

### Horizontal Rules

```markdown
---
```
or
```markdown
***
```

Use to separate major sections or between problem parts.

### Line Breaks

- **New paragraph**: Leave a blank line between text blocks
- **Line break within paragraph**: End a line with two spaces, then enter
- **Forced break**: Use `<br>` (HTML tag, works in most renderers)

---

## Code

Here are practical examples of formatting USAAIO solutions.

### Example 1: Simple Problem Answer

```markdown
## Problem 8.1: Compute Accuracy

Given the confusion matrix:

| | Predicted + | Predicted - |
|---|---|---|
| **Actual +** | 45 (TP) | 5 (FN) |
| **Actual -** | 10 (FP) | 40 (TN) |

**Solution:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{45 + 40}{100} = \mathbf{0.85}
$$
```

### Example 2: Multi-Step Derivation

```markdown
### Part 8.4: F1-Score

1. **Recall** the formula:
   $$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

2. **Substitute** values for the positive class:
   - Precision: $P = 9/11 \approx 0.818$
   - Recall: $R = 0.90$

3. **Compute**:
   $$F_1 = 2 \cdot \frac{0.818 \times 0.90}{0.818 + 0.90} = \frac{1.473}{1.718} \approx \mathbf{0.857}$$
```

### Example 3: Nested List for Algorithm Steps

```markdown
### K-Nearest Neighbors Algorithm

1. **Choose** the number of neighbors $k$
2. **For each** test point $\mathbf{x}_{\text{test}}$:
   - Compute distance to all training points
   - Sort distances in ascending order
   - Select the $k$ closest training points
   - **Classification**: majority vote among $k$ neighbors
   - **Regression**: average of $k$ neighbors' values
3. **Return** predictions
```

---

## Resources

- [Markdown Guide -- Basic Syntax](https://www.markdownguide.org/basic-syntax/)
- [GitHub Flavored Markdown Spec](https://github.github.com/gfm/)
- [Google Colab Markdown Guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
- [Daring Fireball: Markdown Syntax](https://daringfireball.net/projects/markdown/syntax) (the original spec)
