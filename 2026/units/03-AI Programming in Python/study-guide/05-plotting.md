# Data Visualization with Matplotlib & Seaborn

**Prerequisites**: NumPy basics, Pandas
**USAAIO Relevance**: Visualization questions appear on the exam -- you may need to create specific plot types, customize plots, or interpret visualizations. More importantly, plotting is essential for understanding your data and debugging your AI models.

---

## Discovery

### Why Visualization Matters in AI

A table of 10,000 numbers is meaningless to a human. A plot tells a story instantly:

- **Is this data normally distributed?** -> histogram
- **Are these two features correlated?** -> scatter plot
- **Which model performs best across categories?** -> bar chart
- **Where are the outliers?** -> box plot
- **What's the confusion matrix pattern?** -> heatmap

The famous Anscombe's quartet shows four datasets with identical statistical properties (mean, variance, correlation, regression line) that look completely different when plotted. Statistics can deceive; plots reveal truth.

> **Socratic question**: If two datasets have the same mean and standard deviation, are they the same? What information does a histogram reveal that mean/std do not?

### The Matplotlib Architecture

Matplotlib has a layered architecture:

```
┌─────────────────────────────────┐
│         Your Code               │
│  fig, ax = plt.subplots()       │
│  ax.plot(x, y)                  │
├─────────────────────────────────┤
│     Artist Layer                │  <- Figure, Axes, Line2D, Text...
│     (objects you interact with) │
├─────────────────────────────────┤
│     Backend Layer               │  <- renders to screen/file
│     (PNG, SVG, PDF, screen)     │
└─────────────────────────────────┘
```

Key objects:
- **Figure**: the entire window/page (contains one or more Axes)
- **Axes**: a single plot area (contains the actual plot)
- **Axis**: the x-axis or y-axis (ticks, labels, limits)

**Common confusion**: `Axes` (a plot area) vs `Axis` (x or y axis). One Axes has two Axis objects.

---

## Intuition

### The Two Matplotlib APIs

Matplotlib has two ways to create plots:

```
PYPLOT API (quick and dirty)          OBJECT-ORIENTED API (recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

plt.plot(x, y)                        fig, ax = plt.subplots()
plt.xlabel('X')                       ax.plot(x, y)
plt.ylabel('Y')                       ax.set_xlabel('X')
plt.title('Title')                    ax.set_ylabel('Y')
plt.show()                            ax.set_title('Title')
                                      plt.show()

Good for: quick exploration           Good for: publication, subplots,
                                      customization, reproducibility
```

**Always use the OO API for USAAIO** -- it gives you explicit control and works naturally with subplots.

### Choosing the Right Plot

```
What do you want to show?            -> Plot type
━━━━━━━━━━━━━━━━━━━━━━━━━            ━━━━━━━━━━
Distribution of one variable         -> histogram, KDE
Relationship between two variables   -> scatter plot
Trend over time                      -> line plot
Comparison across categories         -> bar chart
Distribution across categories       -> box plot, violin plot
Correlation between many variables   -> heatmap
Part-to-whole relationship           -> pie chart (use sparingly)
```

---

## Math

### Histograms and Density

A histogram bins continuous data into intervals and counts frequencies:

For $n$ data points and $k$ bins, each bin $i$ has:
- **Count**: number of points in the bin
- **Density**: count / (n * bin_width), so total area = 1

**KDE (Kernel Density Estimation)**: a smooth version of the histogram, placing a Gaussian kernel at each data point and summing:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where $K$ is a kernel function (usually Gaussian) and $h$ is the bandwidth.

### Heatmaps and Correlation

The Pearson correlation between columns $X$ and $Y$:

$$r_{XY} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i - \bar{X})^2 \sum(Y_i - \bar{Y})^2}}$$

Range: $[-1, 1]$ where $1$ = perfect positive linear, $-1$ = perfect negative linear, $0$ = no linear relationship.

---

## Code

### Matplotlib Basics

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic line plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color='blue', linewidth=2, label='sin(x)')
ax.plot(x, np.cos(x), color='red', linestyle='--', label='cos(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trigonometric Functions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Line Styles, Markers, Colors

```python
fig, ax = plt.subplots(figsize=(8, 4))

# Format string: 'color marker linestyle'
ax.plot(x, np.sin(x), 'b-', label='solid blue')
ax.plot(x, np.sin(x + 1), 'r--', label='dashed red')
ax.plot(x, np.sin(x + 2), 'g:', label='dotted green')
ax.plot(x, np.sin(x + 3), 'k-.', label='dash-dot black')

# Explicit parameters
ax.plot(x, np.sin(x + 4),
        color='purple',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=4,
        markevery=10,    # show marker every 10 points
        alpha=0.7,
        label='custom')

ax.legend()
plt.tight_layout()
plt.show()

# Common colors: 'b' (blue), 'r' (red), 'g' (green), 'k' (black),
#                'c' (cyan), 'm' (magenta), 'y' (yellow), 'w' (white)
# Or hex: '#FF5733', or named: 'steelblue', 'salmon'

# Common markers: 'o' (circle), 's' (square), '^' (triangle up),
#                 'v' (triangle down), 'D' (diamond), 'x', '+', '.'

# Common linestyles: '-' (solid), '--' (dashed), ':' (dotted), '-.' (dash-dot)
```

### Scatter Plots

```python
np.random.seed(42)
N = 200
x = np.random.randn(N)
y = 0.5 * x + np.random.randn(N) * 0.5
colors = np.random.rand(N)
sizes = np.abs(np.random.randn(N)) * 100

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y,
                     c=colors,      # color by value
                     s=sizes,       # size by value
                     cmap='viridis',# colormap
                     alpha=0.6,
                     edgecolors='white',
                     linewidth=0.5)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Scatter Plot with Color and Size')
plt.colorbar(scatter, ax=ax, label='Color Value')
plt.tight_layout()
plt.show()
```

### Histograms

```python
data = np.random.randn(1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Basic histogram
axes[0].hist(data, bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Basic Histogram')

# Normalized (density)
axes[1].hist(data, bins=30, density=True, alpha=0.7, color='salmon')
axes[1].set_title('Density Histogram')

# Overlapping histograms
data2 = np.random.randn(1000) + 2
axes[2].hist(data, bins=30, alpha=0.5, label='Group A')
axes[2].hist(data2, bins=30, alpha=0.5, label='Group B')
axes[2].legend()
axes[2].set_title('Overlapping Histograms')

plt.tight_layout()
plt.show()
```

### Bar Charts

```python
categories = ['Math', 'Science', 'English', 'History', 'Art']
scores_A = [85, 92, 78, 88, 95]
scores_B = [90, 85, 82, 75, 88]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grouped bar chart
x_pos = np.arange(len(categories))
width = 0.35
axes[0].bar(x_pos - width/2, scores_A, width, label='Class A', color='steelblue')
axes[0].bar(x_pos + width/2, scores_B, width, label='Class B', color='salmon')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel('Score')
axes[0].set_title('Grouped Bar Chart')
axes[0].legend()

# Horizontal bar chart
axes[1].barh(categories, scores_A, color='steelblue')
axes[1].set_xlabel('Score')
axes[1].set_title('Horizontal Bar Chart')

plt.tight_layout()
plt.show()
```

### Subplots

```python
# Grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# axes is a 2D array: axes[row, col]
for i in range(2):
    for j in range(3):
        ax = axes[i, j]
        data = np.random.randn(100)
        ax.hist(data, bins=20)
        ax.set_title(f'Plot ({i},{j})')

plt.tight_layout()  # prevents overlap
plt.show()

# Unequal subplot sizes with gridspec
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3)

ax_big = fig.add_subplot(gs[:, 0:2])   # spans all rows, first 2 cols
ax_top = fig.add_subplot(gs[0, 2])     # top right
ax_bot = fig.add_subplot(gs[1, 2])     # bottom right

ax_big.plot(np.random.randn(100).cumsum())
ax_big.set_title('Main Plot')
ax_top.hist(np.random.randn(100))
ax_top.set_title('Distribution')
ax_bot.scatter(np.random.rand(50), np.random.rand(50))
ax_bot.set_title('Scatter')

plt.tight_layout()
plt.show()
```

### Customization

```python
fig, ax = plt.subplots(figsize=(8, 5))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), 'b-', linewidth=2)

# Title and labels
ax.set_title('Customized Plot', fontsize=16, fontweight='bold')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)

# Axis limits and ticks
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks([-1, 0, 1])

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Annotations
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(3, 1.3),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

# Horizontal/vertical lines
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=np.pi, color='green', linestyle='--', alpha=0.5, label='x=pi')

# Legend
ax.legend(fontsize=10, loc='upper right')

# Spine customization
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('custom_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Seaborn: Statistical Visualization

Seaborn builds on Matplotlib and provides higher-level statistical plots with better defaults.

```python
import seaborn as sns
import pandas as pd

# Set style
sns.set_style('whitegrid')   # 'darkgrid', 'white', 'dark', 'ticks'
sns.set_palette('husl')      # color palette

# Sample data
np.random.seed(42)
df = pd.DataFrame({
    'feature_1': np.random.randn(200),
    'feature_2': np.random.randn(200) * 2 + 1,
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'value': np.random.randn(200) * 10 + 50
})
```

### Seaborn Distribution Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram with KDE
sns.histplot(df['feature_1'], kde=True, ax=axes[0])
axes[0].set_title('Histogram + KDE')

# KDE plot
sns.kdeplot(data=df, x='feature_1', hue='category', ax=axes[1])
axes[1].set_title('KDE by Category')

# Box plot
sns.boxplot(data=df, x='category', y='value', ax=axes[2])
axes[2].set_title('Box Plot')

plt.tight_layout()
plt.show()
```

### Seaborn Relationship Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter with regression line
sns.regplot(data=df, x='feature_1', y='feature_2', ax=axes[0],
            scatter_kws={'alpha': 0.5})
axes[0].set_title('Scatter + Regression')

# Scatter colored by category
sns.scatterplot(data=df, x='feature_1', y='feature_2',
                hue='category', ax=axes[1])
axes[1].set_title('Scatter by Category')

# Violin plot (distribution + box plot)
sns.violinplot(data=df, x='category', y='value', ax=axes[2])
axes[2].set_title('Violin Plot')

plt.tight_layout()
plt.show()
```

### Heatmaps

```python
# Correlation heatmap
numeric_df = df[['feature_1', 'feature_2', 'value']]
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr,
            annot=True,           # show values
            fmt='.2f',            # format to 2 decimals
            cmap='coolwarm',      # blue-white-red colormap
            center=0,             # center colormap at 0
            vmin=-1, vmax=1,      # fix scale
            square=True,          # square cells
            ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Confusion matrix pattern
from sklearn.metrics import confusion_matrix  # or compute manually
# cm = confusion_matrix(y_true, y_pred)
cm = np.array([[50, 5, 2], [3, 45, 8], [1, 4, 52]])
labels = ['Cat', 'Dog', 'Bird']

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

### Pair Plot (Multi-Feature Overview)

```python
# Pair plot -- shows all pairwise relationships
# Diagonal: distribution of each feature
# Off-diagonal: scatter plot of each pair
sns.pairplot(df, hue='category', diag_kind='kde')
plt.suptitle('Pair Plot', y=1.02)
plt.show()
```

### Common Colormaps

```python
# Sequential (for ordered data):
# 'viridis' (default), 'plasma', 'inferno', 'magma', 'Blues', 'Reds'

# Diverging (for data with a center point):
# 'coolwarm', 'RdBu', 'seismic' (use center=0 for positive/negative)

# Qualitative (for categories):
# 'Set1', 'Set2', 'tab10', 'Pastel1'

# Display colormap
fig, axes = plt.subplots(3, 1, figsize=(10, 3))
for ax, cmap_name in zip(axes, ['viridis', 'coolwarm', 'Set1']):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap_name)
    ax.set_title(cmap_name, fontsize=10)
    ax.set_yticks([])
plt.tight_layout()
plt.show()
```

### Saving Figures

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Save as PNG (raster)
fig.savefig('plot.png', dpi=150, bbox_inches='tight')

# Save as PDF (vector -- best for papers)
fig.savefig('plot.pdf', bbox_inches='tight')

# Save as SVG (vector -- best for web)
fig.savefig('plot.svg', bbox_inches='tight')

# bbox_inches='tight' crops whitespace around the plot
```

---

## Resources

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) -- find any plot type with code
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Python Graph Gallery](https://www.python-graph-gallery.com/) -- searchable by plot type
- [Color Brewer](https://colorbrewer2.org/) -- choosing color palettes
