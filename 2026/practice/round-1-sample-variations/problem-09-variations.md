# Problem 9 Variations: Data Preprocessing (EXHAUSTIVE)

> Original: Bank Marketing dataset — DataFrame exploration, normalization, one-hot encoding, binning, train/test split
> Core Skills: pandas operations, min-max normalization, one-hot encoding from scratch, conditional filtering, data visualization

---

## CATEGORY A: Different Dataset Scenarios

### Variation A1: Titanic Dataset Style
Consider a passenger dataset with the following columns:
- `PassengerId`: Integer (unique identifier)
- `Survived`: Integer (0 or 1)
- `Pclass`: Integer (1, 2, or 3)
- `Name`: String
- `Sex`: String ("male" or "female")
- `Age`: Float (with some NaN values)
- `SibSp`: Integer (siblings/spouses aboard)
- `Parch`: Integer (parents/children aboard)
- `Fare`: Float
- `Embarked`: String ("C", "Q", or "S", with some NaN)

```python
import pandas as pd
import numpy as np

# Simulated Titanic-style data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': np.random.choice([0, 1], n),
    'Pclass': np.random.choice([1, 2, 3], n),
    'Name': [f'Passenger_{i}' for i in range(n)],
    'Sex': np.random.choice(['male', 'female'], n),
    'Age': np.where(np.random.random(n) < 0.2, np.nan,
                   np.random.randint(1, 80, n).astype(float)),
    'SibSp': np.random.randint(0, 5, n),
    'Parch': np.random.randint(0, 4, n),
    'Fare': np.random.exponential(30, n),
    'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n, p=[0.2, 0.1, 0.6, 0.1])
})
```

**Part 9.1**: How many rows and columns does the dataset have?

**Part 9.2**: How many missing values are there in the `Age` column? In `Embarked`?

**Part 9.3**: Normalize the `Fare` column to [0, 1] from scratch.

**Part 9.4**: One-hot encode the `Sex` column from scratch.

**Part 9.5**: What is the survival rate for each passenger class?

<details>
<summary>Solution A1</summary>

**9.1**:
```python
print(df.shape)  # (200, 10)
```

**9.2**:
```python
print(df['Age'].isna().sum())      # ~40 (20% of 200)
print(df['Embarked'].isna().sum()) # ~20 (10% of 200)
```

**9.3**:
```python
fare = df['Fare']
fare_min = fare.min()
fare_max = fare.max()
fare_normalized = (fare - fare_min) / (fare_max - fare_min)
```

**9.4**:
```python
# One-hot encode Sex from scratch
for value in df['Sex'].unique():
    df[f'Sex_{value}'] = (df['Sex'] == value).astype(int)
```

**9.5**:
```python
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_by_class)
```
</details>

### Variation A2: Housing Dataset Style
Consider a housing dataset with columns:
- `price`: Float (target variable)
- `area`: Integer (square feet)
- `bedrooms`: Integer
- `bathrooms`: Float (e.g., 2.5)
- `stories`: Integer
- `mainroad`: String ("yes" or "no")
- `guestroom`: String ("yes" or "no")
- `basement`: String ("yes" or "no")
- `hotwaterheating`: String ("yes" or "no")
- `airconditioning`: String ("yes" or "no")
- `parking`: Integer
- `prefarea`: String ("yes" or "no")
- `furnishingstatus`: String ("furnished", "semi-furnished", "unfurnished")

**Part 9.1**: Identify all categorical columns.

**Part 9.2**: Convert all binary "yes"/"no" columns to 0/1 integers.

**Part 9.3**: Apply standardization (z-score) to numeric columns: (x - mean) / std

**Part 9.4**: One-hot encode `furnishingstatus` from scratch.

**Part 9.5**: Split into 70% train, 15% validation, 15% test with seed=42.

<details>
<summary>Solution A2</summary>

**9.1**:
```python
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
#  'airconditioning', 'prefarea', 'furnishingstatus']
```

**9.2**:
```python
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = (df[col] == 'yes').astype(int)
```

**9.3**:
```python
numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
```

**9.4**:
```python
for status in df['furnishingstatus'].unique():
    df[f'furnishingstatus_{status}'] = (df['furnishingstatus'] == status).astype(int)
```

**9.5**:
```python
np.random.seed(42)
n = len(df)
indices = np.random.permutation(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

df_train = df.iloc[train_idx]
df_val = df.iloc[val_idx]
df_test = df.iloc[test_idx]
```
</details>

### Variation A3: Customer Churn Dataset
Consider a telecom customer churn dataset with:
- `customerID`: String
- `gender`: String ("Male"/"Female")
- `SeniorCitizen`: Integer (0/1)
- `tenure`: Integer (months)
- `MonthlyCharges`: Float
- `TotalCharges`: String (with some empty strings)
- `Churn`: String ("Yes"/"No")

**Part 9.1**: Convert `TotalCharges` from string to float, handling empty strings as NaN.

**Part 9.2**: Fill missing `TotalCharges` with the median value.

**Part 9.3**: Create a new feature `AvgMonthlyCharge = TotalCharges / tenure` (handle tenure=0).

**Part 9.4**: Bin `tenure` into categories: "0-12", "13-24", "25-48", "49-72".

**Part 9.5**: Calculate churn rate by tenure bin.

<details>
<summary>Solution A3</summary>

**9.1**:
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Empty strings become NaN
```

**9.2**:
```python
median_total = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total, inplace=True)
```

**9.3**:
```python
# Handle division by zero
df['AvgMonthlyCharge'] = np.where(
    df['tenure'] > 0,
    df['TotalCharges'] / df['tenure'],
    df['MonthlyCharges']  # Use MonthlyCharges for tenure=0
)
```

**9.4**:
```python
bins = [0, 12, 24, 48, 72]
labels = ['0-12', '13-24', '25-48', '49-72']
df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
```

**9.5**:
```python
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
churn_by_tenure = df.groupby('tenure_bin')['Churn_binary'].mean()
print(churn_by_tenure)
```
</details>

---

## CATEGORY B: Different Normalization Techniques

### Variation B1: Min-Max Normalization Variants
Given a column with values: [10, 20, 30, 40, 50, 100]

**Part 9.1**: Normalize to range [0, 1].

**Part 9.2**: Normalize to range [-1, 1].

**Part 9.3**: Normalize to range [0, 100].

**Part 9.4**: What happens if all values are the same (e.g., [5, 5, 5, 5])?

<details>
<summary>Solution B1</summary>

**9.1**: Standard min-max: x' = (x - min) / (max - min)
```python
x = np.array([10, 20, 30, 40, 50, 100])
x_norm = (x - x.min()) / (x.max() - x.min())
# [0.0, 0.111, 0.222, 0.333, 0.444, 1.0]
```

**9.2**: Range [-1, 1]: x' = 2 * (x - min) / (max - min) - 1
```python
x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
# [-1.0, -0.778, -0.556, -0.333, -0.111, 1.0]
```

**9.3**: Range [0, 100]: x' = 100 * (x - min) / (max - min)
```python
x_norm = 100 * (x - x.min()) / (x.max() - x.min())
# [0.0, 11.1, 22.2, 33.3, 44.4, 100.0]
```

**9.4**: If all values are the same:
```python
x = np.array([5, 5, 5, 5])
# max - min = 0, division by zero!
# Solution: return 0 (or 0.5) for all values
x_norm = np.where(x.max() - x.min() == 0, 0, (x - x.min()) / (x.max() - x.min()))
```
</details>

### Variation B2: Z-Score Standardization
Given a column with values: [2, 4, 6, 8, 10]

**Part 9.1**: Compute the z-score standardization from scratch.

**Part 9.2**: After standardization, what is the mean? Standard deviation?

**Part 9.3**: If an outlier (100) is added, how does standardization change?

**Part 9.4**: Compare z-score results with and without the outlier.

<details>
<summary>Solution B2</summary>

**9.1**:
```python
x = np.array([2, 4, 6, 8, 10])
mean = x.mean()  # 6
std = x.std()    # √8 = 2.83 (population std)
z = (x - mean) / std
# [-1.41, -0.71, 0, 0.71, 1.41]
```

**9.2**: After standardization:
- Mean = 0 (by construction)
- Std = 1 (by construction)

**9.3**: With outlier:
```python
x_outlier = np.array([2, 4, 6, 8, 10, 100])
mean_new = x_outlier.mean()  # 21.67
std_new = x_outlier.std()    # 35.4
z_new = (x_outlier - mean_new) / std_new
# [-0.56, -0.50, -0.44, -0.39, -0.33, 2.21]
```

**9.4**: Comparison:
- Without outlier: values spread nicely in [-1.41, 1.41]
- With outlier: original values compressed to [-0.56, -0.33], outlier at 2.21
- **Insight**: Z-score is sensitive to outliers!
</details>

### Variation B3: Robust Scaling (Median/IQR)
Given values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

**Part 9.1**: Compute robust scaling: x' = (x - median) / IQR

**Part 9.2**: Compare with z-score standardization.

**Part 9.3**: When is robust scaling preferred?

<details>
<summary>Solution B3</summary>

**9.1**:
```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
median = np.median(x)  # 5.5
q1 = np.percentile(x, 25)  # 3.25
q3 = np.percentile(x, 75)  # 8.75
iqr = q3 - q1  # 5.5
x_robust = (x - median) / iqr
# [-0.82, -0.64, -0.45, -0.27, -0.09, 0.09, 0.27, 0.45, 0.64, 17.18]
```

**9.2**: Z-score:
```python
mean = x.mean()  # 14.5
std = x.std()    # 28.7
x_zscore = (x - mean) / std
# [-0.47, -0.44, -0.40, -0.37, -0.33, -0.30, -0.26, -0.23, -0.19, 2.98]
```

**9.3**: Robust scaling is preferred when:
- Data has outliers
- Distribution is skewed
- You want scaling invariant to extreme values

The outlier (100) has less influence on robust scaling's parameters.
</details>

### Variation B4: Log Transformation
Given highly skewed values: [1, 10, 100, 1000, 10000]

**Part 9.1**: Apply log transformation: x' = log(x)

**Part 9.2**: Apply log1p transformation: x' = log(1 + x). When is this needed?

**Part 9.3**: Normalize the log-transformed values to [0, 1].

<details>
<summary>Solution B4</summary>

**9.1**:
```python
x = np.array([1, 10, 100, 1000, 10000])
x_log = np.log10(x)  # [0, 1, 2, 3, 4]
# Or natural log: np.log(x) = [0, 2.3, 4.6, 6.9, 9.2]
```

**9.2**: log1p is needed when x can be 0:
```python
x_with_zero = np.array([0, 1, 10, 100])
# np.log(x_with_zero) → -inf for x=0!
x_log1p = np.log1p(x_with_zero)  # [0, 0.69, 2.40, 4.62]
```

**9.3**:
```python
x_log = np.log10(x)  # [0, 1, 2, 3, 4]
x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min())
# [0, 0.25, 0.5, 0.75, 1.0]
```

*Key Insight*: Log transformation compresses large values and spreads small values, making skewed distributions more normal.
</details>

### Variation B5: Power Transformation (Box-Cox)
Given values: [1, 4, 9, 16, 25] (squares of 1-5)

**Part 9.1**: Apply square root transformation.

**Part 9.2**: Apply Box-Cox with λ=0.5 (equivalent to sqrt).

**Part 9.3**: What is Box-Cox with λ=0? λ=1?

<details>
<summary>Solution B5</summary>

**9.1**:
```python
x = np.array([1, 4, 9, 16, 25])
x_sqrt = np.sqrt(x)  # [1, 2, 3, 4, 5]
```

**9.2**: Box-Cox formula: x' = (x^λ - 1) / λ for λ ≠ 0
```python
lambda_val = 0.5
x_boxcox = (x**lambda_val - 1) / lambda_val
# [(1-1)/0.5, (2-1)/0.5, ...] = [0, 2, 4, 6, 8]
```

**9.3**:
- λ = 0: Box-Cox reduces to log(x) (by L'Hôpital's rule as λ→0)
- λ = 1: x' = (x - 1) / 1 = x - 1 (linear shift)
- λ = -1: x' = (x^(-1) - 1) / (-1) = 1 - 1/x (reciprocal transform)

*Key Insight*: Box-Cox finds the optimal λ to make data most normal-like.
</details>

---

## CATEGORY C: Different Encoding Techniques

### Variation C1: One-Hot Encoding Variations
Given a column with values: ["A", "B", "C", "A", "B", "A"]

**Part 9.1**: Create one-hot encoding from scratch.

**Part 9.2**: Create one-hot encoding with drop_first=True (avoid multicollinearity).

**Part 9.3**: What if a new category "D" appears in test data?

<details>
<summary>Solution C1</summary>

**9.1**:
```python
col = pd.Series(["A", "B", "C", "A", "B", "A"])
for value in col.unique():
    print(f"{value}: {(col == value).astype(int).tolist()}")
# A: [1, 0, 0, 1, 0, 1]
# B: [0, 1, 0, 0, 1, 0]
# C: [0, 0, 1, 0, 0, 0]
```

**9.2**: Drop first category:
```python
categories = col.unique()[1:]  # Skip first
for value in categories:
    print(f"{value}: {(col == value).astype(int).tolist()}")
# B: [0, 1, 0, 0, 1, 0]
# C: [0, 0, 1, 0, 0, 0]
# A is implicit when B=0 and C=0
```

**9.3**: Handling unknown categories:
```python
known_categories = ['A', 'B', 'C']
test_col = pd.Series(['A', 'D', 'B'])

# Option 1: Ignore unknown (all zeros)
for cat in known_categories:
    result = (test_col == cat).astype(int)
# D gets [0, 0, 0] representation

# Option 2: Add "unknown" column
df_encoded['unknown'] = (~test_col.isin(known_categories)).astype(int)
```
</details>

### Variation C2: Label Encoding
Given ordinal categories: ["low", "medium", "high", "medium", "low"]

**Part 9.1**: Apply label encoding from scratch (preserving order).

**Part 9.2**: What's the difference between label encoding and one-hot encoding?

**Part 9.3**: When should you use label encoding vs. one-hot encoding?

<details>
<summary>Solution C2</summary>

**9.1**:
```python
col = pd.Series(["low", "medium", "high", "medium", "low"])
order_map = {"low": 0, "medium": 1, "high": 2}
col_encoded = col.map(order_map)
# [0, 1, 2, 1, 0]
```

**9.2**: Differences:
| Aspect | Label Encoding | One-Hot Encoding |
|--------|---------------|------------------|
| Output | Single column | Multiple columns |
| Values | Integers 0, 1, 2, ... | Binary 0/1 |
| Implies order | Yes | No |
| Dimensionality | Low (1) | High (k categories) |

**9.3**: Use label encoding when:
- Categories have natural order (ordinal data)
- Using tree-based models (can handle label encoding)
- Memory/dimensionality is a concern

Use one-hot encoding when:
- Categories have no order (nominal data)
- Using linear models, neural networks
- Number of categories is small
</details>

### Variation C3: Target Encoding
Given categories and target:
- categories: ["A", "A", "B", "B", "C"]
- target: [1, 0, 1, 1, 0]

**Part 9.1**: Compute target encoding (mean of target for each category).

**Part 9.2**: What is the problem with target encoding? How to address it?

<details>
<summary>Solution C3</summary>

**9.1**:
```python
df = pd.DataFrame({
    'cat': ['A', 'A', 'B', 'B', 'C'],
    'target': [1, 0, 1, 1, 0]
})

# Target encoding = mean target per category
target_means = df.groupby('cat')['target'].mean()
# A: 0.5, B: 1.0, C: 0.0

df['cat_encoded'] = df['cat'].map(target_means)
# [0.5, 0.5, 1.0, 1.0, 0.0]
```

**9.2**: Problems and solutions:

**Problem**: Target leakage (using target to create features can cause overfitting).

**Solutions**:
1. **Leave-one-out encoding**: For each row, use mean of OTHER rows with same category
2. **K-fold target encoding**: Use cross-validation to compute means
3. **Smoothing**: Blend category mean with global mean:
   ```python
   m = 10  # smoothing factor
   global_mean = df['target'].mean()
   n_cat = df.groupby('cat')['target'].count()
   smoothed = (n_cat * target_means + m * global_mean) / (n_cat + m)
   ```
</details>

### Variation C4: Frequency Encoding
Given categories: ["A", "A", "A", "B", "B", "C"]

**Part 9.1**: Compute frequency encoding (count of each category).

**Part 9.2**: Compute normalized frequency encoding (proportion).

<details>
<summary>Solution C4</summary>

**9.1**:
```python
col = pd.Series(["A", "A", "A", "B", "B", "C"])
freq = col.value_counts()
# A: 3, B: 2, C: 1

col_freq_encoded = col.map(freq)
# [3, 3, 3, 2, 2, 1]
```

**9.2**:
```python
n = len(col)
freq_norm = col.value_counts(normalize=True)
# A: 0.5, B: 0.333, C: 0.167

col_freq_norm = col.map(freq_norm)
# [0.5, 0.5, 0.5, 0.333, 0.333, 0.167]
```

*Key Insight*: Frequency encoding captures how common a category is without creating many columns like one-hot encoding.
</details>

### Variation C5: Binary Encoding
Given categories: ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8"]

**Part 9.1**: Why is binary encoding useful for high-cardinality features?

**Part 9.2**: Encode the categories using binary representation.

<details>
<summary>Solution C5</summary>

**9.1**: For k categories:
- One-hot: k columns
- Binary: ⌈log₂(k)⌉ columns

For 8 categories: one-hot needs 8 columns, binary needs only 3!

**9.2**:
```python
categories = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8"]
# Assign integer labels 0-7
label_map = {cat: i for i, cat in enumerate(categories)}

# Binary representation (3 bits for 8 categories)
# cat1 (0): 000 → [0, 0, 0]
# cat2 (1): 001 → [0, 0, 1]
# cat3 (2): 010 → [0, 1, 0]
# cat4 (3): 011 → [0, 1, 1]
# cat5 (4): 100 → [1, 0, 0]
# cat6 (5): 101 → [1, 0, 1]
# cat7 (6): 110 → [1, 1, 0]
# cat8 (7): 111 → [1, 1, 1]

def binary_encode(cat, n_bits=3):
    label = label_map[cat]
    return [(label >> i) & 1 for i in range(n_bits-1, -1, -1)]

for cat in categories:
    print(f"{cat}: {binary_encode(cat)}")
```
</details>

---

## CATEGORY D: Missing Value Handling

### Variation D1: Different Imputation Strategies
Given a column with values: [1, 2, NaN, 4, 5, NaN, 7]

**Part 9.1**: Impute with mean.

**Part 9.2**: Impute with median.

**Part 9.3**: Impute with mode (most frequent).

**Part 9.4**: Impute with forward fill (ffill).

**Part 9.5**: Impute with backward fill (bfill).

<details>
<summary>Solution D1</summary>

```python
import numpy as np
import pandas as pd

col = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7])
```

**9.1**: Mean imputation:
```python
mean_val = col.mean()  # (1+2+4+5+7)/5 = 3.8
col_mean = col.fillna(mean_val)
# [1, 2, 3.8, 4, 5, 3.8, 7]
```

**9.2**: Median imputation:
```python
median_val = col.median()  # 4
col_median = col.fillna(median_val)
# [1, 2, 4, 4, 5, 4, 7]
```

**9.3**: Mode imputation:
```python
# All values appear once, so mode is first value
mode_val = col.mode()[0]  # 1 (or could be any)
col_mode = col.fillna(mode_val)
```

**9.4**: Forward fill:
```python
col_ffill = col.fillna(method='ffill')
# [1, 2, 2, 4, 5, 5, 7]  # NaN takes previous value
```

**9.5**: Backward fill:
```python
col_bfill = col.fillna(method='bfill')
# [1, 2, 4, 4, 5, 7, 7]  # NaN takes next value
```
</details>

### Variation D2: Missing Value Indicators
Given a DataFrame with missing values:

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, 2, 3, 4]
})
```

**Part 9.1**: Create a missing indicator column for each column.

**Part 9.2**: When is adding a missing indicator useful?

<details>
<summary>Solution D2</summary>

**9.1**:
```python
for col in df.columns:
    df[f'{col}_missing'] = df[col].isna().astype(int)

# Result:
#      A    B  C  A_missing  B_missing  C_missing
# 0  1.0  NaN  1          0          1          0
# 1  2.0  2.0  2          0          0          0
# 2  NaN  3.0  3          1          0          0
# 3  4.0  4.0  4          0          0          0
```

**9.2**: Missing indicators are useful when:
- Missingness itself is informative (e.g., people skip income questions)
- You want to preserve information while also imputing
- The pattern of missingness correlates with the target
</details>

### Variation D3: Conditional Imputation
Given a DataFrame:

```python
df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'height': [175, np.nan, 180, np.nan, 170]
})
```

**Part 9.1**: Impute missing heights with the mean height of the same gender.

<details>
<summary>Solution D3</summary>

**9.1**: Group-wise imputation:
```python
# Calculate mean height per gender
mean_by_gender = df.groupby('gender')['height'].mean()
# M: (175+180+170)/3 = 175
# F: NaN (no valid values initially)

# If F has no values, use overall mean as fallback
overall_mean = df['height'].mean()  # 175

# Impute with group mean
def impute_by_group(row):
    if pd.isna(row['height']):
        group_mean = mean_by_gender.get(row['gender'], overall_mean)
        return group_mean if not pd.isna(group_mean) else overall_mean
    return row['height']

df['height_imputed'] = df.apply(impute_by_group, axis=1)
```

Alternative using `transform`:
```python
df['height_imputed'] = df.groupby('gender')['height'].transform(
    lambda x: x.fillna(x.mean() if x.notna().any() else df['height'].mean())
)
```
</details>

### Variation D4: Detecting Missing Value Patterns
Given a DataFrame with multiple columns having missing values:

**Part 9.1**: Count missing values per column.

**Part 9.2**: Count missing values per row.

**Part 9.3**: Find rows with more than 50% missing values.

**Part 9.4**: Find columns with more than 30% missing values.

<details>
<summary>Solution D4</summary>

```python
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan],
    'B': [np.nan, np.nan, 3, 4],
    'C': [1, 2, np.nan, np.nan],
    'D': [1, 2, 3, 4]
})
```

**9.1**:
```python
missing_per_col = df.isna().sum()
# A: 2, B: 2, C: 2, D: 0
```

**9.2**:
```python
missing_per_row = df.isna().sum(axis=1)
# Row 0: 1, Row 1: 2, Row 2: 1, Row 3: 2
```

**9.3**:
```python
n_cols = len(df.columns)
threshold = 0.5 * n_cols  # 2 columns
rows_high_missing = df[df.isna().sum(axis=1) > threshold]
# Rows 1 and 3 (each has 2 missing > 50% of 4 columns)
```

**9.4**:
```python
n_rows = len(df)
threshold = 0.3 * n_rows  # 1.2 rows
cols_high_missing = df.columns[df.isna().sum() > threshold]
# A, B, C (each has 2 > 1.2)
```
</details>

---

## CATEGORY E: Filtering and Conditional Operations

### Variation E1: Boolean Indexing
Given DataFrame:

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 70000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
})
```

**Part 9.1**: Filter rows where age > 30.

**Part 9.2**: Filter rows where department is 'IT' AND salary > 60000.

**Part 9.3**: Filter rows where department is 'IT' OR 'HR'.

**Part 9.4**: Filter rows where name starts with 'A' or 'B'.

<details>
<summary>Solution E1</summary>

**9.1**:
```python
df_filtered = df[df['age'] > 30]
# Charlie (35), Eve (32)
```

**9.2**:
```python
df_filtered = df[(df['department'] == 'IT') & (df['salary'] > 60000)]
# Charlie only
```

**9.3**:
```python
df_filtered = df[df['department'].isin(['IT', 'HR'])]
# Alice, Bob, Charlie, Eve
```

**9.4**:
```python
df_filtered = df[df['name'].str.startswith(('A', 'B'))]
# Alice, Bob
```
</details>

### Variation E2: Query Method
Using the same DataFrame:

**Part 9.1**: Rewrite the filter "age > 30" using `.query()`.

**Part 9.2**: Query with variable: filter where salary > threshold (threshold=55000).

**Part 9.3**: Query with string operations.

<details>
<summary>Solution E2</summary>

**9.1**:
```python
df_filtered = df.query('age > 30')
```

**9.2**:
```python
threshold = 55000
df_filtered = df.query('salary > @threshold')
# The @ symbol references external variables
```

**9.3**:
```python
# Query doesn't directly support string methods, use boolean indexing
df_filtered = df[df['name'].str.len() > 4]
# Or filter department:
df_filtered = df.query('department == "IT"')
```
</details>

### Variation E3: GroupBy and Aggregate
Given DataFrame:

```python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'A'],
    'subcategory': ['x', 'y', 'x', 'y', 'x'],
    'value': [10, 20, 30, 40, 50],
    'count': [1, 2, 3, 4, 5]
})
```

**Part 9.1**: Calculate mean value per category.

**Part 9.2**: Calculate sum of count and mean of value per category.

**Part 9.3**: Calculate multiple statistics per category.

**Part 9.4**: Group by multiple columns.

<details>
<summary>Solution E3</summary>

**9.1**:
```python
mean_by_cat = df.groupby('category')['value'].mean()
# A: (10+20+50)/3 = 26.67
# B: (30+40)/2 = 35
```

**9.2**:
```python
agg_result = df.groupby('category').agg({
    'count': 'sum',
    'value': 'mean'
})
# A: count=8, value=26.67
# B: count=7, value=35
```

**9.3**:
```python
stats = df.groupby('category')['value'].agg(['mean', 'std', 'min', 'max', 'count'])
```

**9.4**:
```python
multi_group = df.groupby(['category', 'subcategory'])['value'].mean()
# A, x: (10+50)/2 = 30
# A, y: 20
# B, x: 30
# B, y: 40
```
</details>

### Variation E4: Complex Conditional Logic
Given DataFrame:

```python
df = pd.DataFrame({
    'score': [85, 72, 91, 65, 78],
    'attendance': [0.95, 0.80, 0.88, 0.70, 0.92]
})
```

**Part 9.1**: Create a grade column: A (≥90), B (≥80), C (≥70), D (≥60), F (<60).

**Part 9.2**: Create a pass/fail column: pass if score ≥ 70 AND attendance ≥ 0.75.

<details>
<summary>Solution E4</summary>

**9.1**: Using `np.select`:
```python
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70,
    df['score'] >= 60
]
choices = ['A', 'B', 'C', 'D']
df['grade'] = np.select(conditions, choices, default='F')
# [B, C, A, D, C]
```

Or using `pd.cut`:
```python
bins = [0, 60, 70, 80, 90, 100]
labels = ['F', 'D', 'C', 'B', 'A']
df['grade'] = pd.cut(df['score'], bins=bins, labels=labels, right=False)
```

**9.2**:
```python
df['pass'] = ((df['score'] >= 70) & (df['attendance'] >= 0.75)).astype(int)
# [1, 1, 1, 0, 1]  (row 3 fails attendance requirement)
```
</details>

---

## CATEGORY F: Train/Test Split Variations

### Variation F1: Basic Splits
Given 100 samples:

**Part 9.1**: Split 80/20 for train/test with seed=42.

**Part 9.2**: Split 60/20/20 for train/val/test.

**Part 9.3**: What happens if you don't set a seed?

<details>
<summary>Solution F1</summary>

**9.1**:
```python
np.random.seed(42)
n = 100
indices = np.random.permutation(n)
train_end = int(0.8 * n)

train_idx = indices[:train_end]  # 80 samples
test_idx = indices[train_end:]   # 20 samples
```

**9.2**:
```python
np.random.seed(42)
n = 100
indices = np.random.permutation(n)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_idx = indices[:train_end]       # 60 samples
val_idx = indices[train_end:val_end]  # 20 samples
test_idx = indices[val_end:]          # 20 samples
```

**9.3**: Without setting seed:
- Different splits each run
- Results not reproducible
- Hard to compare experiments
</details>

### Variation F2: Stratified Split
Given imbalanced data:

```python
y = np.array([0]*90 + [1]*10)  # 90% class 0, 10% class 1
```

**Part 9.1**: Why is stratified splitting important for imbalanced data?

**Part 9.2**: Implement stratified split from scratch.

<details>
<summary>Solution F2</summary>

**9.1**: Without stratification:
- Test set might have 0 positive samples by chance
- Train set class distribution might differ from original
- Model evaluation becomes unreliable

With stratification:
- Preserves class proportions in both train and test
- More representative evaluation

**9.2**:
```python
def stratified_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)

    # Find indices for each class
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    # Shuffle each class separately
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)

    # Calculate split points
    n0_test = int(len(class_0_idx) * test_size)  # 18 from 90
    n1_test = int(len(class_1_idx) * test_size)  # 2 from 10

    # Split each class
    test_idx = np.concatenate([class_0_idx[:n0_test], class_1_idx[:n1_test]])
    train_idx = np.concatenate([class_0_idx[n0_test:], class_1_idx[n1_test:]])

    # Shuffle final indices
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)

    return train_idx, test_idx

# Verify proportions
train_idx, test_idx = stratified_split(None, y)
print(f"Train class 1 ratio: {y[train_idx].mean():.2f}")  # ~0.10
print(f"Test class 1 ratio: {y[test_idx].mean():.2f}")   # ~0.10
```
</details>

### Variation F3: Time Series Split
Given time-ordered data:

```python
dates = pd.date_range('2020-01-01', periods=100, freq='D')
values = np.random.randn(100)
df = pd.DataFrame({'date': dates, 'value': values})
```

**Part 9.1**: Why can't we use random split for time series?

**Part 9.2**: Implement chronological train/test split.

**Part 9.3**: Implement rolling window validation.

<details>
<summary>Solution F3</summary>

**9.1**: Random split causes **data leakage**:
- Future data in training, past data in testing
- Model "cheats" by seeing the future
- Overly optimistic performance estimates

**9.2**: Chronological split:
```python
n = len(df)
split_date = df['date'].iloc[int(0.8 * n)]

train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
# First 80 days for training, last 20 for testing
```

**9.3**: Rolling window:
```python
def rolling_window_split(df, train_size=60, test_size=10, step=10):
    """
    Yields (train_idx, test_idx) for rolling validation
    """
    n = len(df)
    splits = []

    start = 0
    while start + train_size + test_size <= n:
        train_idx = range(start, start + train_size)
        test_idx = range(start + train_size, start + train_size + test_size)
        splits.append((list(train_idx), list(test_idx)))
        start += step

    return splits

# Example: train on days 0-59, test on 60-69
#          train on days 10-69, test on 70-79
#          train on days 20-79, test on 80-89
```
</details>

### Variation F4: K-Fold Cross-Validation
**Part 9.1**: Implement k-fold CV from scratch.

**Part 9.2**: What is the relationship between k and bias/variance of the CV estimate?

<details>
<summary>Solution F4</summary>

**9.1**:
```python
def kfold_split(n, k=5, seed=42):
    """
    Returns list of (train_idx, val_idx) tuples
    """
    np.random.seed(seed)
    indices = np.random.permutation(n)
    fold_size = n // k

    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else n

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        folds.append((train_idx, val_idx))

    return folds

# Usage
n = 100
folds = kfold_split(n, k=5)
for i, (train_idx, val_idx) in enumerate(folds):
    print(f"Fold {i}: train={len(train_idx)}, val={len(val_idx)}")
```

**9.2**: Bias-variance trade-off:
- **Small k (e.g., 2)**:
  - High bias (each fold uses only 50% of data for training)
  - Low variance (only 2 estimates to average)

- **Large k (e.g., n, Leave-One-Out)**:
  - Low bias (uses n-1 samples for training)
  - High variance (estimates are highly correlated)

- **k=5 or k=10**: Good balance, commonly used
</details>

---

## CATEGORY G: Coding Implementations

### Variation G1: Complete Preprocessing Pipeline
Implement a preprocessing class from scratch:

```python
class SimplePreprocessor:
    def __init__(self):
        self.numeric_stats = {}  # Store min, max for each numeric column
        self.category_values = {}  # Store unique values for each categorical column

    def fit(self, df, numeric_cols, categorical_cols):
        """Learn parameters from training data"""
        pass

    def transform(self, df):
        """Apply learned transformations"""
        pass

    def fit_transform(self, df, numeric_cols, categorical_cols):
        """Fit and transform in one step"""
        pass
```

<details>
<summary>Solution G1</summary>

```python
import numpy as np
import pandas as pd

class SimplePreprocessor:
    def __init__(self):
        self.numeric_stats = {}
        self.category_values = {}
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, df, numeric_cols, categorical_cols):
        """Learn parameters from training data"""
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # Learn min/max for normalization
        for col in numeric_cols:
            self.numeric_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max()
            }

        # Learn unique values for one-hot encoding
        for col in categorical_cols:
            self.category_values[col] = df[col].unique().tolist()

        return self

    def transform(self, df):
        """Apply learned transformations"""
        df_transformed = df.copy()

        # Normalize numeric columns
        for col in self.numeric_cols:
            stats = self.numeric_stats[col]
            range_val = stats['max'] - stats['min']
            if range_val > 0:
                df_transformed[col] = (df[col] - stats['min']) / range_val
            else:
                df_transformed[col] = 0

        # One-hot encode categorical columns
        for col in self.categorical_cols:
            for val in self.category_values[col]:
                df_transformed[f'{col}_{val}'] = (df[col] == val).astype(int)
            df_transformed.drop(col, axis=1, inplace=True)

        return df_transformed

    def fit_transform(self, df, numeric_cols, categorical_cols):
        """Fit and transform in one step"""
        self.fit(df, numeric_cols, categorical_cols)
        return self.transform(df)

# Test
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 80000],
    'gender': ['M', 'F', 'M', 'F'],
    'city': ['NYC', 'LA', 'NYC', 'SF']
})

preprocessor = SimplePreprocessor()
df_train = preprocessor.fit_transform(df,
                                       numeric_cols=['age', 'income'],
                                       categorical_cols=['gender', 'city'])
print(df_train)
```
</details>

### Variation G2: Missing Value Handler
Implement missing value handling from scratch:

```python
class MissingValueHandler:
    def __init__(self, strategy='mean'):
        """
        strategy: 'mean', 'median', 'mode', or 'constant'
        """
        pass

    def fit(self, df, columns):
        """Learn imputation values"""
        pass

    def transform(self, df):
        """Apply imputation"""
        pass
```

<details>
<summary>Solution G2</summary>

```python
import numpy as np
import pandas as pd

class MissingValueHandler:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.impute_values = {}
        self.columns = []

    def fit(self, df, columns):
        """Learn imputation values from data"""
        self.columns = columns

        for col in columns:
            if self.strategy == 'mean':
                self.impute_values[col] = df[col].mean()
            elif self.strategy == 'median':
                self.impute_values[col] = df[col].median()
            elif self.strategy == 'mode':
                self.impute_values[col] = df[col].mode()[0]
            elif self.strategy == 'constant':
                self.impute_values[col] = self.fill_value
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, df):
        """Apply imputation using learned values"""
        df_transformed = df.copy()

        for col in self.columns:
            df_transformed[col] = df_transformed[col].fillna(self.impute_values[col])

        return df_transformed

    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df)

# Test
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
})

handler = MissingValueHandler(strategy='mean')
df_imputed = handler.fit_transform(df, columns=['A', 'B'])
print(df_imputed)
print(f"Imputation values: {handler.impute_values}")
```
</details>

### Variation G3: Custom Train/Test Splitter
Implement a comprehensive splitter:

```python
class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.0, stratify=None, seed=42):
        pass

    def split(self, X, y=None):
        """Returns (X_train, X_val, X_test, y_train, y_val, y_test)"""
        pass
```

<details>
<summary>Solution G3</summary>

```python
import numpy as np
import pandas as pd

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.0, stratify=False, seed=42):
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.seed = seed

    def _stratified_indices(self, y, sizes):
        """Get stratified indices for given split sizes"""
        np.random.seed(self.seed)

        classes = np.unique(y)
        indices = {c: np.where(y == c)[0] for c in classes}

        # Shuffle each class
        for c in classes:
            np.random.shuffle(indices[c])

        splits = [[] for _ in range(len(sizes) + 1)]

        for c in classes:
            class_idx = indices[c]
            n_class = len(class_idx)

            prev_end = 0
            for i, size in enumerate(sizes):
                end = int(prev_end + size * n_class)
                splits[i].extend(class_idx[prev_end:end])
                prev_end = end

            # Remaining goes to last split (train)
            splits[-1].extend(class_idx[prev_end:])

        return [np.array(s) for s in splits]

    def _random_indices(self, n, sizes):
        """Get random indices for given split sizes"""
        np.random.seed(self.seed)
        indices = np.random.permutation(n)

        splits = []
        prev_end = 0
        for size in sizes:
            end = int(prev_end + size * n)
            splits.append(indices[prev_end:end])
            prev_end = end

        splits.append(indices[prev_end:])  # Remaining for train
        return splits

    def split(self, X, y=None):
        """
        Returns train/val/test splits
        If val_size=0, returns (X_train, X_test, y_train, y_test)
        Else returns (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        sizes = [self.test_size]
        if self.val_size > 0:
            sizes = [self.test_size, self.val_size]

        if self.stratify and y is not None:
            split_indices = self._stratified_indices(np.array(y), sizes)
        else:
            split_indices = self._random_indices(n, sizes)

        # Unpack based on number of splits
        if self.val_size > 0:
            test_idx, val_idx, train_idx = split_indices
        else:
            test_idx, train_idx = split_indices
            val_idx = None

        # Handle both DataFrame and numpy array
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            X_val = X.iloc[val_idx] if val_idx is not None else None
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
            X_val = X[val_idx] if val_idx is not None else None

        if y is not None:
            y = np.array(y)
            y_train = y[train_idx]
            y_test = y[test_idx]
            y_val = y[val_idx] if val_idx is not None else None

            if self.val_size > 0:
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                return X_train, X_test, y_train, y_test
        else:
            if self.val_size > 0:
                return X_train, X_val, X_test
            else:
                return X_train, X_test

# Test
X = np.arange(100).reshape(-1, 1)
y = np.array([0]*80 + [1]*20)

splitter = DataSplitter(test_size=0.2, val_size=0.1, stratify=True, seed=42)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

print(f"Train: {len(X_train)}, class 1 ratio: {y_train.mean():.2f}")
print(f"Val: {len(X_val)}, class 1 ratio: {y_val.mean():.2f}")
print(f"Test: {len(X_test)}, class 1 ratio: {y_test.mean():.2f}")
```
</details>

### Variation G4: Data Quality Report
Implement a data quality checker:

```python
def data_quality_report(df):
    """
    Generate a comprehensive data quality report including:
    - Column types
    - Missing values
    - Unique values
    - Basic statistics
    """
    pass
```

<details>
<summary>Solution G4</summary>

```python
import numpy as np
import pandas as pd

def data_quality_report(df):
    """Generate comprehensive data quality report"""

    report = []

    for col in df.columns:
        col_data = df[col]
        col_info = {
            'column': col,
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.notna().sum(),
            'null_count': col_data.isna().sum(),
            'null_pct': f"{100 * col_data.isna().mean():.1f}%",
            'unique_count': col_data.nunique(),
            'unique_pct': f"{100 * col_data.nunique() / len(col_data):.1f}%"
        }

        # Add statistics based on type
        if np.issubdtype(col_data.dtype, np.number):
            col_info['min'] = col_data.min()
            col_info['max'] = col_data.max()
            col_info['mean'] = f"{col_data.mean():.2f}"
            col_info['std'] = f"{col_data.std():.2f}"
            col_info['median'] = col_data.median()
        else:
            col_info['min'] = '-'
            col_info['max'] = '-'
            col_info['mean'] = '-'
            col_info['std'] = '-'
            col_info['median'] = '-'
            # Top values for categorical
            if col_data.nunique() < 10:
                col_info['top_values'] = dict(col_data.value_counts().head(5))

        report.append(col_info)

    report_df = pd.DataFrame(report)

    # Print summary
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print()
    print("Column Details:")
    print("-" * 80)
    print(report_df.to_string(index=False))

    return report_df

# Test
df = pd.DataFrame({
    'id': range(100),
    'age': np.random.randint(18, 80, 100),
    'income': np.random.exponential(50000, 100),
    'category': np.random.choice(['A', 'B', 'C', np.nan], 100),
    'target': np.random.choice([0, 1], 100)
})

report = data_quality_report(df)
```
</details>

### Variation G5: Binning Implementation
Implement various binning strategies:

```python
def create_bins(values, strategy='equal_width', n_bins=5):
    """
    strategy: 'equal_width', 'equal_freq', 'custom'
    Returns: bin_edges, bin_labels
    """
    pass
```

<details>
<summary>Solution G5</summary>

```python
import numpy as np
import pandas as pd

def create_bins(values, strategy='equal_width', n_bins=5, custom_edges=None):
    """
    Create bins using different strategies.

    Parameters:
    - values: array-like
    - strategy: 'equal_width', 'equal_freq', 'custom'
    - n_bins: number of bins (ignored for 'custom')
    - custom_edges: list of bin edges (only for 'custom')

    Returns:
    - binned_values: Series with bin labels
    - bin_edges: array of bin edges
    """
    values = np.array(values)

    if strategy == 'equal_width':
        # Equal width bins
        bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)

    elif strategy == 'equal_freq':
        # Equal frequency (quantile) bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(values, percentiles)
        # Remove duplicate edges
        bin_edges = np.unique(bin_edges)

    elif strategy == 'custom':
        if custom_edges is None:
            raise ValueError("custom_edges required for 'custom' strategy")
        bin_edges = np.array(custom_edges)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create labels
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
              for i in range(len(bin_edges)-1)]

    # Apply binning
    binned = pd.cut(values, bins=bin_edges, labels=labels, include_lowest=True)

    return binned, bin_edges

# Test with different strategies
np.random.seed(42)
values = np.random.exponential(50, 1000)  # Skewed distribution

print("Equal Width Bins:")
binned_ew, edges_ew = create_bins(values, strategy='equal_width', n_bins=5)
print(pd.Series(binned_ew).value_counts().sort_index())
print()

print("Equal Frequency Bins:")
binned_ef, edges_ef = create_bins(values, strategy='equal_freq', n_bins=5)
print(pd.Series(binned_ef).value_counts().sort_index())
print()

print("Custom Bins:")
binned_c, edges_c = create_bins(values, strategy='custom',
                                 custom_edges=[0, 25, 50, 100, 200, 500])
print(pd.Series(binned_c).value_counts().sort_index())
```
</details>

---

## CATEGORY H: Edge Cases & Tricky Situations

### Variation H1: Handling Extreme Values
Given data with outliers: [1, 2, 3, 4, 5, 1000]

**Part 9.1**: What happens with min-max normalization?

**Part 9.2**: How to handle outliers before normalization?

**Part 9.3**: Implement clipping before normalization.

<details>
<summary>Solution H1</summary>

**9.1**: Min-max normalization:
```python
x = np.array([1, 2, 3, 4, 5, 1000])
x_norm = (x - x.min()) / (x.max() - x.min())
# [0.0, 0.001, 0.002, 0.003, 0.004, 1.0]
# All normal values compressed to near 0!
```

**9.2**: Handling outliers:
1. **Clipping**: Cap values at percentiles
2. **Winsorization**: Replace outliers with boundary values
3. **Remove**: Filter out extreme values
4. **Log transform**: Compress large values

**9.3**: Clipping implementation:
```python
def normalize_with_clipping(x, lower_pct=1, upper_pct=99):
    """Clip outliers before normalization"""
    x = np.array(x)
    lower = np.percentile(x, lower_pct)
    upper = np.percentile(x, upper_pct)

    x_clipped = np.clip(x, lower, upper)
    x_norm = (x_clipped - lower) / (upper - lower)

    return x_norm

x = np.array([1, 2, 3, 4, 5, 1000])
x_norm = normalize_with_clipping(x)
# Values spread more evenly, outlier capped
```
</details>

### Variation H2: Empty Categories
Given categorical column in train: ["A", "B", "C"]
Test data has new category: ["A", "D", "B"]

**Part 9.1**: What happens with standard one-hot encoding?

**Part 9.2**: How to handle unknown categories?

<details>
<summary>Solution H2</summary>

**9.1**: Standard one-hot encoding fails:
```python
# Training: fit on ["A", "B", "C"]
# Test: ["A", "D", "B"] - "D" is unknown!
# Options:
# 1. Crash (if using sklearn without handle_unknown)
# 2. All zeros (if handled)
# 3. Map to special "unknown" category
```

**9.2**: Handling unknown categories:
```python
class SafeOneHotEncoder:
    def __init__(self, handle_unknown='zeros'):
        """
        handle_unknown: 'zeros' (all zeros) or 'indicator' (add unknown column)
        """
        self.handle_unknown = handle_unknown
        self.categories = {}

    def fit(self, df, columns):
        for col in columns:
            self.categories[col] = df[col].unique().tolist()
        return self

    def transform(self, df):
        df_out = df.copy()

        for col, cats in self.categories.items():
            # Create columns for known categories
            for cat in cats:
                df_out[f'{col}_{cat}'] = (df[col] == cat).astype(int)

            # Handle unknown
            if self.handle_unknown == 'indicator':
                df_out[f'{col}_unknown'] = (~df[col].isin(cats)).astype(int)

            df_out.drop(col, axis=1, inplace=True)

        return df_out

# Test
train = pd.DataFrame({'cat': ['A', 'B', 'C', 'A']})
test = pd.DataFrame({'cat': ['A', 'D', 'B']})

encoder = SafeOneHotEncoder(handle_unknown='indicator')
encoder.fit(train, ['cat'])
test_encoded = encoder.transform(test)
print(test_encoded)
# cat_A  cat_B  cat_C  cat_unknown
#     1      0      0            0   (A - known)
#     0      0      0            1   (D - unknown)
#     0      1      0            0   (B - known)
```
</details>

### Variation H3: Data Leakage
**Part 9.1**: What is data leakage in preprocessing?

**Part 9.2**: Give an example of leakage in normalization.

**Part 9.3**: How to prevent leakage?

<details>
<summary>Solution H3</summary>

**9.1**: Data leakage occurs when information from the test set influences training, leading to overly optimistic performance estimates.

**9.2**: Leakage in normalization:
```python
# WRONG: Normalize using entire dataset
df_all = pd.concat([df_train, df_test])
mean_all = df_all['feature'].mean()
std_all = df_all['feature'].std()

df_train['feature_norm'] = (df_train['feature'] - mean_all) / std_all
df_test['feature_norm'] = (df_test['feature'] - mean_all) / std_all
# Test statistics influenced training normalization!
```

**9.3**: Prevention:
```python
# CORRECT: Learn parameters from train only
mean_train = df_train['feature'].mean()
std_train = df_train['feature'].std()

df_train['feature_norm'] = (df_train['feature'] - mean_train) / std_train
df_test['feature_norm'] = (df_test['feature'] - mean_train) / std_train
# Only train statistics used!
```

**Rule**: Always `fit()` on training data, then `transform()` both train and test.
</details>

### Variation H4: Constant Features
Given a column where all values are the same: [5, 5, 5, 5, 5]

**Part 9.1**: What happens with z-score standardization?

**Part 9.2**: What happens with variance threshold?

**Part 9.3**: How should you handle constant features?

<details>
<summary>Solution H4</summary>

**9.1**: Z-score fails:
```python
x = np.array([5, 5, 5, 5, 5])
mean = x.mean()  # 5
std = x.std()    # 0
z = (x - mean) / std  # Division by zero!
```

**9.2**: Variance threshold removes them:
```python
# Constant features have zero variance
# Variance threshold removes features with variance < threshold
# With threshold=0.0, constant features are removed
```

**9.3**: Handling strategies:
```python
def handle_constant_features(df, threshold=0.0):
    """Remove or flag constant features"""
    variances = df.var()
    constant_cols = variances[variances <= threshold].index.tolist()

    print(f"Constant/low-variance columns: {constant_cols}")

    # Option 1: Remove
    df_cleaned = df.drop(columns=constant_cols)

    # Option 2: Keep but handle in normalization
    # Return 0 or original value if std=0

    return df_cleaned, constant_cols
```
</details>

### Variation H5: High Cardinality Categories
Given a category with 10,000 unique values (e.g., zip codes):

**Part 9.1**: Why is one-hot encoding problematic?

**Part 9.2**: What alternatives exist?

<details>
<summary>Solution H5</summary>

**9.1**: Problems with one-hot:
- Creates 10,000 new columns (high dimensionality)
- Sparse representation (mostly zeros)
- Memory intensive
- Can cause overfitting

**9.2**: Alternatives:
1. **Frequency encoding**: Replace with count of occurrences
2. **Target encoding**: Replace with mean target per category
3. **Binary encoding**: Uses log₂(10000) ≈ 14 columns
4. **Embedding**: Learn dense representation (for neural nets)
5. **Grouping**: Combine rare categories into "Other"
6. **Feature hashing**: Hash categories to fixed number of buckets

```python
def reduce_cardinality(df, col, threshold=100):
    """Replace rare categories with 'Other'"""
    freq = df[col].value_counts()
    rare_cats = freq[freq < threshold].index
    df[col] = df[col].replace(rare_cats, 'Other')
    return df

# Example: 10,000 zip codes → ~500 frequent + "Other"
```
</details>

---

## KEY FORMULAS SUMMARY

| Technique | Formula |
|-----------|---------|
| Min-Max Normalization | x' = (x - min) / (max - min) |
| Range Scaling [a, b] | x' = a + (b-a)(x - min) / (max - min) |
| Z-Score Standardization | z = (x - μ) / σ |
| Robust Scaling | x' = (x - median) / IQR |
| Log Transform | x' = log(x) or log(1 + x) |
| Box-Cox | x' = (x^λ - 1) / λ |

| Encoding | Use Case |
|----------|----------|
| One-Hot | Nominal, low cardinality |
| Label | Ordinal categories |
| Target | High cardinality, regression |
| Frequency | High cardinality |
| Binary | Medium-high cardinality |

---

## ATOMIC SKILLS CHECKLIST

- [ ] Explore DataFrame (shape, dtypes, head, describe)
- [ ] Identify and handle missing values
- [ ] Implement min-max normalization from scratch
- [ ] Implement z-score standardization from scratch
- [ ] Implement one-hot encoding from scratch
- [ ] Apply conditional filtering with boolean masks
- [ ] Use groupby for aggregations
- [ ] Create bins (equal width and equal frequency)
- [ ] Split data into train/test (with stratification)
- [ ] Prevent data leakage in preprocessing
- [ ] Handle edge cases (constant features, unknown categories, outliers)

---

## COMMON MISCONCEPTIONS

1. **Fitting on entire dataset**: Always fit preprocessors on TRAIN data only, then transform both train and test.

2. **Normalizing before splitting**: Split first, then normalize to prevent leakage.

3. **One-hot for ordinal data**: Use label encoding for ordinal categories to preserve order.

4. **Ignoring missing value patterns**: Missingness itself can be informative—consider adding indicator features.

5. **Same preprocessing for all features**: Different features may need different treatments (numeric vs. categorical).

6. **Forgetting to handle new categories**: Test data may have categories not seen in training.

7. **Using mean for skewed data**: Median is more robust for skewed distributions with outliers.
