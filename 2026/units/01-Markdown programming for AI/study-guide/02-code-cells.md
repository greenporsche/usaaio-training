# Code Cells and Code Formatting

**Prerequisites**: `01-text-formatting.md`
**USAAIO Relevance**: Round 1 requires mixing markdown explanations with code snippets. Round 2 is entirely code-based. You must know when to use markdown cells vs code cells, and how to present code clearly in both.

---

## Discovery

It is 2011. You are a researcher at IPython (later Project Jupyter). You have a problem: your research involves writing Python code *and* explaining the mathematics behind it. Your options are:

- Write code in a `.py` file and explanations in a separate Word document
- Write everything in code comments (ugly, hard to read)
- Write a paper and paste screenshots of code output (fragile, not reproducible)

None of these work well. You want a single document that interleaves explanation, code, and output -- a computational *notebook*.

**Think about this**: What if every paragraph in a textbook could also be *executed*? What if the equation $y = mx + b$ could be followed immediately by code that plots the line, and the plot appeared right there in the document?

**Question**: When you read a USAAIO solution, which parts are "explanation" and which parts are "computation"? How would you separate them?

**Misconception trap**: Markdown cells and code cells are fundamentally different. A code cell runs Python. A markdown cell renders formatted text. You cannot run Python in a markdown cell, and you cannot render LaTeX in a code cell (unless you use specific libraries like IPython.display).

---

## Intuition

What you just discovered is the core idea behind Jupyter notebooks (and Google Colab, which is Google's hosted version of Jupyter).

A notebook is a sequence of **cells**. Each cell is one of two types:

```
+---------------------------+
| ## Problem 8.1            |  <-- Markdown cell
| The accuracy formula:     |      (rendered text)
| $$\frac{TP+TN}{Total}$$  |
+---------------------------+
| accuracy = (45+40) / 100  |  <-- Code cell
| print(f"Accuracy: {acc}") |      (executable Python)
+---------------------------+
| Accuracy: 0.85            |  <-- Output (auto-generated)
+---------------------------+
| **Answer**: 0.85 = 85%    |  <-- Markdown cell
+---------------------------+
```

### When to Use Each Cell Type

| Use Markdown When... | Use Code When... |
|---|---|
| Explaining your approach | Computing a numerical answer |
| Writing mathematical derivations | Loading/processing data |
| Presenting a confusion matrix | Training a model |
| Stating your final answer | Generating a plot |
| Annotating code logic | Running inference |

### What Goes Wrong Without Proper Cell Organization?

**Bad notebook**: Everything in one giant code cell with `# comments`

```python
# Problem 8.1
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
tp, tn, fp, fn = 45, 40, 10, 5
acc = (tp + tn) / (tp + tn + fp + fn)
# acc = 0.85
# So the accuracy is 85%
print(acc)
```

**Good notebook**: Interleaved markdown and code cells

*Markdown cell:*
> ## Problem 8.1: Accuracy
> $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

*Code cell:*
```python
tp, tn, fp, fn = 45, 40, 10, 5
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy:.2%}")
```

*Output:*
> `Accuracy: 85.00%`

*Markdown cell:*
> **Answer**: The accuracy is **0.85 (85%)**.

The second version is what USAAIO graders expect.

---

## Math

This section formalizes the syntax for presenting code in markdown.

### Fenced Code Blocks

Triple backticks create a code block. Add the language name for syntax highlighting:

````markdown
```python
import numpy as np
x = np.array([1, 2, 3])
print(x.mean())
```
````

Supported languages in Google Colab markdown cells:

| Language Tag | Use Case |
|---|---|
| `python` | All USAAIO code |
| `bash` | Shell commands (`!pip install`) |
| `json` | Data formats, config files |
| `latex` | LaTeX source (for showing raw LaTeX) |
| (none) | Generic preformatted text |

*Reasoning required*: Code blocks inside markdown cells are **not executable**. They are for display only. To run code, use a code cell.

### Inline Code

Use single backticks for inline code references:

```markdown
The function `softmax()` takes a vector `z` and returns probabilities.
The variable `learning_rate` is set to `0.001`.
```

Use inline code for:
- Function names: `model.fit()`
- Variable names: `X_train`
- File names: `model.pth`
- Short expressions: `n_samples = 100`

### Showing Code Output in Markdown

When you need to show expected output in a markdown cell (not a code cell), use a code block without a language tag or prefix with `Output:`:

```markdown
**Expected output:**
```
Accuracy: 0.85
Precision: 0.818
Recall: 0.900
```
```

### Mixing Math and Code

A common USAAIO pattern: derive a formula in math, then implement it in code.

*Markdown cell:*
```markdown
### Softmax Function

The softmax function converts a vector of raw scores (logits) into probabilities:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

For numerical stability, we subtract $\max(z)$ before exponentiating.
```

*Code cell:*
```python
import numpy as np

def softmax(z):
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum():.4f}")
```

---

## Code

### Example 1: Proper USAAIO Notebook Structure

Here is how a complete USAAIO Round 1 solution should be organized:

**Cell 1 (Markdown):**
```markdown
# USAAIO 2026 Round 1 -- Student Name

## Problem 8: Classification Metrics
```

**Cell 2 (Markdown):**
```markdown
### Part 8.1: Accuracy

Given the confusion matrix:

| | Predicted + | Predicted - |
|---|---|---|
| **Actual +** | 45 (TP) | 5 (FN) |
| **Actual -** | 10 (FP) | 40 (TN) |

$$
\text{Accuracy} = \frac{TP + TN}{N} = \frac{45 + 40}{100} = \mathbf{0.85}
$$
```

**Cell 3 (Code):**
```python
# Verification
tp, tn, fp, fn = 45, 40, 10, 5
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy}")
```

**Cell 4 (Markdown):**
```markdown
### Part 8.2: Precision

$$
\text{Precision}_+ = \frac{TP}{TP + FP} = \frac{45}{55} = \frac{9}{11} \approx 0.818
$$

$$
\text{Precision}_- = \frac{TN}{TN + FN} = \frac{40}{45} = \frac{8}{9} \approx 0.889
$$
```

### Example 2: Annotated Code with Markdown

When USAAIO asks you to explain code, use this pattern:

**Cell 1 (Markdown):**
```markdown
## Explaining the K-NN Pipeline

The code below implements k-Nearest Neighbors with preprocessing:
1. **StandardScaler** normalizes features to zero mean and unit variance
2. **GridSearchCV** finds the optimal $k$ value
3. **F1-macro** evaluates performance equally across all classes
```

**Cell 2 (Code):**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11]}
search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=5)
search.fit(X_train, y_train)
```

**Cell 3 (Markdown):**
```markdown
The optimal value of $k$ is found by cross-validation.
Standardization is critical because kNN uses Euclidean distance,
which is sensitive to feature scale.
```

### Example 3: Google Colab Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+M M` | Convert cell to markdown |
| `Ctrl+M Y` | Convert cell to code |
| `Ctrl+M A` | Insert cell above |
| `Ctrl+M B` | Insert cell below |
| `Ctrl+M D` | Delete cell |
| `Shift+Enter` | Run cell, move to next |
| `Ctrl+Enter` | Run cell, stay in place |

---

## Resources

- [Google Colab Overview](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/)
- [Google Colab Markdown Guide](https://colab.research.google.com/notebooks/markdown_guide.ipynb)
- [IPython Display Module](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html) -- for rendering math/HTML from code cells
