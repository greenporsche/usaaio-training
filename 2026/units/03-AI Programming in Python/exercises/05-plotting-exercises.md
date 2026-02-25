# Data Visualization Exercises

**Time target**: 2-5 minutes each | **Total**: 6 exercises

---

## Exercise 1: Figure and Axes

What is wrong with this code? Fix it.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.subplots(1, 2, figsize=(10, 4))
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
```

<details>
<summary>Solution</summary>

The problem: `plt.subplots(1, 2)` creates two axes, but the code ignores them. Both `plt.plot()` calls go to the same (last) axes.

**Fixed code**:
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x, np.sin(x))
ax1.set_title('sin(x)')
ax2.plot(x, np.cos(x))
ax2.set_title('cos(x)')
plt.tight_layout()
plt.show()
```

**Key lesson**: Always unpack the return value of `plt.subplots()` into `fig` and axes variables. Use the OO API (`ax.plot()`) instead of the pyplot API (`plt.plot()`) when working with subplots.
</details>

---

## Exercise 2: What Does This Plot Look Like?

Without running the code, describe what this plot shows:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.randn(3, 100)

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(3):
    ax.hist(data[i], bins=20, alpha=0.5, label=f'Group {i}')
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
plt.show()
```

<details>
<summary>Solution</summary>

The plot shows **three overlapping histograms** on the same axes:
- 3 groups of 100 samples each, all drawn from a standard normal distribution
- Each histogram has 20 bins and 50% transparency (`alpha=0.5`)
- The histograms overlap, and because they all come from the same distribution (standard normal), they look similar but with random variation
- A legend identifies each group (Group 0, Group 1, Group 2)
- X-axis: "Value" (roughly -3 to 3 for standard normal)
- Y-axis: "Frequency" (raw counts, not density)

The overlapping transparency lets you see where distributions overlap and differ. This is a common pattern for comparing distributions across groups.
</details>

---

## Exercise 3: Scatter Plot with Color Encoding

Write code to create a scatter plot of 200 random 2D points where:
- X and Y coordinates are from a standard normal distribution
- Point **color** represents the distance from the origin
- Point **size** is proportional to the distance from the origin
- Include a colorbar with label "Distance from origin"
- Title: "2D Random Points"

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(200)
y = np.random.randn(200)
distance = np.sqrt(x**2 + y**2)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y,
                     c=distance,
                     s=distance * 30,
                     cmap='viridis',
                     alpha=0.7,
                     edgecolors='white',
                     linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='Distance from origin')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Random Points')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```

**Key details**:
- `c=distance` maps values to colors via the colormap
- `s=distance * 30` makes farther points larger (scale factor is arbitrary)
- `cmap='viridis'` is the default colormap (perceptually uniform)
- `set_aspect('equal')` ensures circles look circular
</details>

---

## Exercise 4: Grouped Bar Chart

Given this data, create a grouped bar chart comparing math and science scores across students:

```python
students = ['Alice', 'Bob', 'Carol', 'Dave']
math_scores = [92, 78, 85, 95]
science_scores = [88, 82, 90, 75]
```

Requirements:
- Bars should be side-by-side (not stacked)
- Different colors for math and science
- Include legend, title, and y-axis label
- Y-axis should start at 0

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

students = ['Alice', 'Bob', 'Carol', 'Dave']
math_scores = [92, 78, 85, 95]
science_scores = [88, 82, 90, 75]

x = np.arange(len(students))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, math_scores, width, label='Math', color='steelblue')
ax.bar(x + width/2, science_scores, width, label='Science', color='salmon')

ax.set_xticks(x)
ax.set_xticklabels(students)
ax.set_ylabel('Score')
ax.set_title('Math vs Science Scores')
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.show()
```

**Pattern**: For grouped bars, create `x = np.arange(n)` positions, then offset each group by `+/- width/2`. For 3 groups, use `-width`, `0`, `+width`.
</details>

---

## Exercise 5: Seaborn Heatmap

Given a 4x4 correlation matrix, create a heatmap with:
- Values annotated to 2 decimal places
- "coolwarm" colormap centered at 0
- Square cells
- No duplicate labels (show both axes)

```python
import numpy as np

np.random.seed(42)
data = np.random.randn(100, 4)
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(100) * 0.3  # correlated
features = ['Height', 'Weight', 'Age', 'Score']
```

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)
data = np.random.randn(100, 4)
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(100) * 0.3
features = ['Height', 'Weight', 'Age', 'Score']

df = pd.DataFrame(data, columns=features)
corr = df.corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            xticklabels=features,
            yticklabels=features,
            ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

**Key points**:
- `center=0` ensures white is at correlation 0
- `vmin=-1, vmax=1` fixes the color scale to the full correlation range
- `annot=True, fmt='.2f'` displays values in each cell
- Height and Weight should show high positive correlation (~0.9) since we constructed them that way
</details>

---

## Exercise 6: Multi-Panel Figure

Create a figure with 4 subplots in a 2x2 grid showing different views of the same dataset:

```python
np.random.seed(42)
x = np.random.randn(500)
y = 0.5 * x + np.random.randn(500) * 0.3
```

- Top-left: scatter plot of x vs y
- Top-right: histogram of x
- Bottom-left: histogram of y
- Bottom-right: 2D hexbin plot of x vs y

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(500)
y = 0.5 * x + np.random.randn(500) * 0.3

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: scatter
axes[0, 0].scatter(x, y, alpha=0.5, s=10)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Scatter Plot')

# Top-right: histogram of x
axes[0, 1].hist(x, bins=30, color='steelblue', edgecolor='white')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of x')

# Bottom-left: histogram of y
axes[1, 0].hist(y, bins=30, color='salmon', edgecolor='white')
axes[1, 0].set_xlabel('y')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of y')

# Bottom-right: hexbin
hb = axes[1, 1].hexbin(x, y, gridsize=20, cmap='YlOrRd')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('2D Density (Hexbin)')
plt.colorbar(hb, ax=axes[1, 1])

plt.suptitle('Dataset Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Hexbin plots** show 2D density by binning points into hexagonal cells and coloring by count. They are better than scatter plots when you have many overlapping points, because scatter plots hide density (all overlapping points look the same).
</details>
