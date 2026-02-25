# Text Formatting Exercises

**5 exercises** | Covers: headings, lists, tables, emphasis, blockquotes

---

## Exercise 1: Structure a Problem Solution

**Target time**: 2 minutes

You received full marks on USAAIO Problem 8 (Classification Metrics). Your raw answer is:

```
the accuracy is 0.85 which is 85 percent. i computed it by adding TP and TN and dividing by the total. TP is 45, TN is 40, FP is 10, FN is 5. total is 100. so 85 divided by 100 is 0.85.
```

Rewrite this as a properly formatted USAAIO solution using:
- A heading for the problem part
- Bold for the final answer
- The formula in LaTeX (display mode)
- Clear step-by-step structure

<details>
<summary>Solution</summary>

```markdown
## Part 8.1: Accuracy

Given: $TP = 45$, $TN = 40$, $FP = 10$, $FN = 5$, $N = 100$.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{45 + 40}{100} = \mathbf{0.85 = 85\%}
$$
```

Key improvements:
- Heading identifies which part this answers
- Given values are stated explicitly
- Formula is in display mode with LaTeX
- Final answer is bolded

</details>

---

## Exercise 2: Build a Confusion Matrix Table

**Target time**: 3 minutes

Create a markdown table representing this confusion matrix for a 3-class classifier (Cat, Dog, Bird):

- Cat predicted as Cat: 30, as Dog: 5, as Bird: 2
- Dog predicted as Cat: 3, as Dog: 40, as Bird: 7
- Bird predicted as Cat: 1, as Dog: 4, as Bird: 35

Include row/column headers and make the actual class labels bold.

<details>
<summary>Solution</summary>

```markdown
| | Predicted Cat | Predicted Dog | Predicted Bird |
|---|:---:|:---:|:---:|
| **Actual Cat** | 30 | 5 | 2 |
| **Actual Dog** | 3 | 40 | 7 |
| **Actual Bird** | 1 | 4 | 35 |
```

Renders as:

| | Predicted Cat | Predicted Dog | Predicted Bird |
|---|:---:|:---:|:---:|
| **Actual Cat** | 30 | 5 | 2 |
| **Actual Dog** | 3 | 40 | 7 |
| **Actual Bird** | 1 | 4 | 35 |

</details>

---

## Exercise 3: Fix the Markdown

**Target time**: 2 minutes

The following markdown has 6 errors. Find and fix all of them.

```markdown
#Problem 8

##Part 8.1

The accuracy is **0.85.

We compute it using the formula:
- Step 1: Add TP + TN = 85
- Step 2; Divide by total = 100
  -Step 3: Result = 0.85

The answer is [85%].
```

<details>
<summary>Solution</summary>

There are 6 errors:

1. `#Problem 8` -- missing space after `#`. Should be `# Problem 8`
2. `##Part 8.1` -- missing space after `##`. Should be `## Part 8.1`
3. `**0.85.` -- unclosed bold. Should be `**0.85**`
4. `- Step 2;` -- semicolon instead of colon (stylistic, but inconsistent). Should be `- Step 2:`
5. `  -Step 3:` -- missing space after `-` in list item. Should be `  - Step 3:`
6. `[85%]` -- square brackets create a broken link reference. Should be `85%` or `**85%**`

Corrected:

```markdown
# Problem 8

## Part 8.1

The accuracy is **0.85**.

We compute it using the formula:
- Step 1: Add TP + TN = 85
- Step 2: Divide by total = 100
  - Step 3: Result = 0.85

The answer is **85%**.
```

</details>

---

## Exercise 4: Create a Hyperparameter Comparison Table

**Target time**: 3 minutes

Format the following information as a right-aligned numeric table with a left-aligned text column:

```
k=1: accuracy 0.72, f1 0.68, training time 0.1s
k=3: accuracy 0.81, f1 0.79, training time 0.2s
k=5: accuracy 0.85, f1 0.83, training time 0.3s
k=7: accuracy 0.83, f1 0.81, training time 0.3s
k=11: accuracy 0.80, f1 0.78, training time 0.4s
```

Bold the row with the best accuracy. Include a caption as a heading.

<details>
<summary>Solution</summary>

```markdown
### Hyperparameter Search Results: k-NN

| k | Accuracy | F1-Score | Training Time |
|:--|--------:|---------:|--------------:|
| 1 | 0.72 | 0.68 | 0.1s |
| 3 | 0.81 | 0.79 | 0.2s |
| **5** | **0.85** | **0.83** | **0.3s** |
| 7 | 0.83 | 0.81 | 0.3s |
| 11 | 0.80 | 0.78 | 0.4s |
```

Key points:
- `:--` for left alignment (k column)
- `--:` for right alignment (numeric columns)
- Bold the entire best row
- Heading serves as table caption

</details>

---

## Exercise 5: Format a Multi-Part USAAIO Answer

**Target time**: 5 minutes

You are answering USAAIO Problem 12 about a pretrained ResNet34 model. Format a complete solution for these parts using proper markdown structure:

- Part 12.1: Total learnable parameters = 21,797,672
- Part 12.2: Output shape of layer4 for input (B, 3, 224, 224) is (B, 512, 7, 7)
- Part 12.3: 33 conv layers, 1 max pool, 1 average pool, 1 fully connected
- Part 12.4: Transfer learning -- freeze backbone, replace final FC layer with nn.Linear(512, 5)

Use headings, code blocks (for the PyTorch code in 12.4), bold for answers, and inline code for tensor shapes.

<details>
<summary>Solution</summary>

```markdown
## Problem 12: Pretrained Model Analysis (ResNet34)

### Part 12.1: Total Learnable Parameters

```python
total_params = sum(p.numel() for p in model.parameters())
# Total: 21,797,672
```

**Answer**: The model has **21,797,672** learnable parameters.

### Part 12.2: Output Shape of `layer4`

For an input tensor of shape `(B, 3, 224, 224)`:

$$
224 \xrightarrow{\text{conv1+pool}} 56 \xrightarrow{\text{layer1}} 56 \xrightarrow{\text{layer2}} 28 \xrightarrow{\text{layer3}} 14 \xrightarrow{\text{layer4}} 7
$$

**Answer**: The output shape is `(B, 512, 7, 7)`.

### Part 12.3: Layer Count

| Layer Type | Count |
|:---|---:|
| Convolutional (`Conv2d`) | 33 |
| Max Pooling (`MaxPool2d`) | 1 |
| Average Pooling (`AdaptiveAvgPool2d`) | 1 |
| Fully Connected (`Linear`) | 1 |

### Part 12.4: Transfer Learning

```python
import torch.nn as nn
from torchvision.models import resnet34

model = resnet34(weights='DEFAULT')

# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for 5-class classification
model.fc = nn.Linear(512, 5)
```

The new `model.fc` layer has `requires_grad=True` by default, so only it will be updated during training.
```

</details>

---
