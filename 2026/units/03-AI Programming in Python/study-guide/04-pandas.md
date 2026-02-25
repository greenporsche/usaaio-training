# Pandas -- Tabular Data Manipulation

**Prerequisites**: Python fundamentals, NumPy basics
**USAAIO Relevance**: Many USAAIO problems involve loading, cleaning, and transforming structured data. Pandas is the standard tool for this. You need to know groupby, merge, and vectorized operations on DataFrames.

---

## Discovery

### Why Pandas Exists

NumPy is great for numerical arrays where every element is the same type. But real-world data is messy:

- A dataset might have columns of different types (name: string, age: int, score: float)
- Rows might have missing values
- You need to filter, group, and aggregate by categories
- Data comes in CSV, JSON, Excel files -- not raw arrays

Pandas solves these problems. It builds on NumPy but adds:
- **Labels**: rows and columns have names, not just indices
- **Mixed types**: each column can have a different dtype
- **Missing data handling**: built-in NaN support
- **I/O**: read/write CSV, JSON, Excel, SQL, and more
- **GroupBy**: split-apply-combine operations

The name comes from "**pan**el **da**ta" -- a term from econometrics for multi-dimensional structured datasets.

> **Socratic question**: Why not just use a list of dictionaries? Think about what happens when you want to compute the average of a column across 1 million rows. How does Pandas make this fast?

**Answer**: Under the hood, each Pandas column is a NumPy array. So `df['score'].mean()` runs at NumPy speed, not Python loop speed. Pandas gives you dictionary-like access with array-like performance.

---

## Intuition

### DataFrame: A Table with Superpowers

```
              DataFrame
     ┌─────────────────────────┐
     │  name    age   score    │  <- columns (each is a Series)
     │─────────────────────────│
  0  │  Alice   25    92.5     │  <- rows (each has an index)
  1  │  Bob     30    87.3     │
  2  │  Carol   28    95.1     │
  3  │  Dave    22    78.9     │
     └─────────────────────────┘
           ↑
      Index (0, 1, 2, 3)

  - Column access:  df['score']      -> Series
  - Row access:     df.loc[0]        -> Series
  - Cell access:    df.loc[0, 'score'] -> 92.5
```

### loc vs iloc: The Two Indexing Systems

```
loc:  Label-based indexing (by name)
      df.loc[0:2, 'name':'score']     <- INCLUSIVE on both ends
      df.loc[[0, 2], ['name', 'age']] <- specific labels

iloc: Integer-based indexing (by position)
      df.iloc[0:2, 0:3]              <- EXCLUSIVE end (like Python)
      df.iloc[[0, 2], [0, 1]]        <- specific positions

KEY DIFFERENCE:
  df.loc[0:2]   -> rows with labels 0, 1, 2   (3 rows, inclusive)
  df.iloc[0:2]  -> rows at positions 0, 1      (2 rows, exclusive end)
```

### GroupBy: Split-Apply-Combine

```
Original DataFrame:
┌────────┬───────┬───────┐
│ team   │ name  │ score │
│────────│───────│───────│
│ A      │ Alice │  90   │
│ A      │ Bob   │  85   │
│ B      │ Carol │  92   │
│ B      │ Dave  │  78   │
│ B      │ Eve   │  88   │
└────────┴───────┴───────┘

df.groupby('team')['score'].mean()

  Step 1: SPLIT by team
  ┌─────────────┐    ┌─────────────────┐
  │ Team A      │    │ Team B          │
  │ Alice: 90   │    │ Carol: 92       │
  │ Bob:   85   │    │ Dave:  78       │
  └─────────────┘    │ Eve:   88       │
                     └─────────────────┘

  Step 2: APPLY mean()
  Team A: (90 + 85) / 2 = 87.5
  Team B: (92 + 78 + 88) / 3 = 86.0

  Step 3: COMBINE
  ┌────────┬───────┐
  │ team   │ score │
  │ A      │ 87.5  │
  │ B      │ 86.0  │
  └────────┴───────┘
```

### Merge: Combining DataFrames

```
df1 (students):          df2 (courses):
┌────┬───────┐           ┌────┬─────────┐
│ id │ name  │           │ id │ course  │
│ 1  │ Alice │           │ 1  │ Math    │
│ 2  │ Bob   │           │ 1  │ Science │
│ 3  │ Carol │           │ 3  │ Math    │
└────┴───────┘           └────┴─────────┘

pd.merge(df1, df2, on='id', how='inner')  # only matching ids
┌────┬───────┬─────────┐
│ id │ name  │ course  │
│ 1  │ Alice │ Math    │
│ 1  │ Alice │ Science │
│ 3  │ Carol │ Math    │
└────┴───────┴─────────┘

how='left'   -> all rows from df1, NaN where no match in df2
how='right'  -> all rows from df2, NaN where no match in df1
how='outer'  -> all rows from both, NaN where no match
how='inner'  -> only rows with matches in both (default)
```

---

## Math

Pandas operations correspond to common statistical and relational algebra operations:

| Pandas Operation | Statistical / Math Concept |
|-----------------|---------------------------|
| `df.mean()` | $\bar{x} = \frac{1}{n}\sum x_i$ |
| `df.std()` | $s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$ (Bessel's correction by default) |
| `df.corr()` | Pearson correlation matrix |
| `df.groupby().mean()` | Conditional expectation $E[Y \mid X=x]$ |
| `pd.merge(df1, df2)` | Relational join (from database theory) |
| `df.pivot_table()` | Cross-tabulation |

**Note on `std()`**: Pandas uses `ddof=1` (sample std) by default, while NumPy uses `ddof=0` (population std). This matters for small samples.

---

## Code

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol', 'Dave'],
    'age': [25, 30, 28, 22],
    'score': [92.5, 87.3, 95.1, 78.9],
    'team': ['A', 'A', 'B', 'B']
})

# From NumPy array
data = np.random.randn(100, 3)
df_np = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'feature_3'])

# From CSV
# df = pd.read_csv('data.csv')
# df = pd.read_csv('data.csv', index_col=0, parse_dates=['date'])

# Quick inspection
df.head()          # first 5 rows
df.tail(3)         # last 3 rows
df.shape           # (4, 4)
df.dtypes          # dtype of each column
df.describe()      # count, mean, std, min, 25%, 50%, 75%, max
df.info()          # summary with memory usage
df.columns         # Index(['name', 'age', 'score', 'team'])
df.index           # RangeIndex(start=0, stop=4, step=1)
```

### Selection and Indexing

```python
# Column selection
df['name']                    # Series
df[['name', 'age']]           # DataFrame (note double brackets)

# Row selection with loc (label-based, inclusive)
df.loc[0]                     # Series (single row)
df.loc[0:2]                   # DataFrame (rows 0, 1, 2 -- inclusive!)
df.loc[0:2, 'name':'score']   # DataFrame (rows 0-2, columns name to score)
df.loc[[0, 3], ['name', 'score']]  # specific rows and columns

# Row selection with iloc (integer-based, exclusive end)
df.iloc[0]                    # Series (first row)
df.iloc[0:2]                  # DataFrame (rows 0, 1 -- exclusive end!)
df.iloc[0:2, 0:2]             # first 2 rows, first 2 columns
df.iloc[[0, 3], [0, 2]]       # specific positions

# Boolean indexing (filtering)
df[df['age'] > 25]            # rows where age > 25
df[df['team'] == 'A']         # rows where team is A
df[(df['age'] > 22) & (df['score'] > 85)]  # compound condition

# query() -- string-based filtering (often cleaner)
df.query('age > 25')
df.query('team == "A" and score > 85')

# Setting values
df.loc[0, 'score'] = 93.0     # set single cell
df.loc[df['team'] == 'A', 'score'] = 100  # set by condition
```

### Adding and Modifying Columns

```python
# New column from computation
df['score_pct'] = df['score'] / 100.0

# New column from condition
df['passed'] = df['score'] > 80

# New column from multiple columns
df['adjusted'] = df['score'] * (df['age'] / 30)

# New column with apply (use sparingly -- prefer vectorized ops)
df['name_upper'] = df['name'].apply(lambda x: x.upper())

# New column with np.where (vectorized conditional)
df['grade'] = np.where(df['score'] >= 90, 'A',
              np.where(df['score'] >= 80, 'B', 'C'))

# Drop columns
df = df.drop(columns=['score_pct', 'passed'])

# Rename columns
df = df.rename(columns={'score': 'exam_score'})
```

### Sorting and Ranking

```python
# Sort by column
df.sort_values('score')                    # ascending (default)
df.sort_values('score', ascending=False)   # descending
df.sort_values(['team', 'score'], ascending=[True, False])  # multi-column

# Rank
df['rank'] = df['score'].rank(ascending=False)  # 1 = highest score
```

### GroupBy Operations

```python
# Basic groupby
df.groupby('team')['score'].mean()
# team
# A    89.9
# B    87.0

# Multiple aggregations
df.groupby('team')['score'].agg(['mean', 'std', 'count'])
#       mean       std  count
# A    89.9  3.677   2
# B    87.0  8.131   2

# Multiple columns
df.groupby('team').agg({
    'score': ['mean', 'max'],
    'age': 'mean'
})

# Transform: apply function and broadcast back to original shape
df['team_mean_score'] = df.groupby('team')['score'].transform('mean')
# Each row gets its team's mean score (same length as original df)

# Filter groups
df.groupby('team').filter(lambda g: g['score'].mean() > 85)

# Custom aggregation
df.groupby('team').apply(
    lambda g: pd.Series({
        'top_scorer': g.loc[g['score'].idxmax(), 'name'],
        'avg_score': g['score'].mean()
    })
)
```

### Merge and Join

```python
# Setup
students = pd.DataFrame({
    'student_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Carol', 'Dave']
})
grades = pd.DataFrame({
    'student_id': [1, 1, 2, 3],
    'course': ['Math', 'Sci', 'Math', 'Sci'],
    'grade': [95, 88, 76, 92]
})

# Inner join (default) -- only matching rows
pd.merge(students, grades, on='student_id')

# Left join -- all students, NaN for missing grades
pd.merge(students, grades, on='student_id', how='left')

# Merge on different column names
df1 = pd.DataFrame({'id': [1, 2], 'val': ['a', 'b']})
df2 = pd.DataFrame({'student_id': [1, 2], 'score': [90, 85]})
pd.merge(df1, df2, left_on='id', right_on='student_id')

# Concatenate DataFrames (stacking)
df_all = pd.concat([df1, df2, df3], axis=0)  # vertical stack
df_wide = pd.concat([df1, df2], axis=1)       # horizontal stack
```

### Handling Missing Data

```python
# Create data with NaN
df = pd.DataFrame({
    'a': [1, 2, np.nan, 4],
    'b': [np.nan, 2, 3, 4],
    'c': [1, 2, 3, 4]
})

# Detection
df.isna()              # DataFrame of True/False
df.isna().sum()        # count NaN per column
df.isna().sum().sum()  # total NaN count

# Removal
df.dropna()            # drop rows with ANY NaN
df.dropna(subset=['a'])  # drop rows where 'a' is NaN
df.dropna(how='all')   # drop rows where ALL values are NaN

# Filling
df.fillna(0)           # fill NaN with 0
df.fillna(df.mean())   # fill NaN with column means
df['a'].fillna(method='ffill')  # forward fill (carry last valid value)
df['a'].fillna(method='bfill')  # backward fill

# Interpolation
df['a'].interpolate()  # linear interpolation
```

### Value Counts and Pivot Tables

```python
# Value counts -- frequency of each unique value
df['team'].value_counts()
# A    2
# B    2

# Crosstab
pd.crosstab(df['team'], df['grade'])

# Pivot table
sales = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'revenue': [100, 150, 200, 120, 180, 90]
})

sales.pivot_table(
    values='revenue',
    index='region',
    columns='product',
    aggfunc='mean'
)
#          A      B
# East  140.0  150.0
# West  200.0  105.0
```

### Method Chaining (Pipelines)

```python
# Pandas supports chaining -- each method returns a DataFrame
result = (
    df
    .query('age > 20')
    .assign(score_norm=lambda d: d['score'] / d['score'].max())
    .groupby('team')
    .agg(avg_score=('score_norm', 'mean'), count=('name', 'count'))
    .sort_values('avg_score', ascending=False)
    .reset_index()
)

# This is the idiomatic Pandas style -- no intermediate variables,
# each step is clear, and the data flows top to bottom.
```

### Converting Between Pandas and NumPy

```python
# DataFrame to NumPy
arr = df[['age', 'score']].to_numpy()     # shape: (N, 2)
arr = df[['age', 'score']].values         # same thing (older API)

# NumPy to DataFrame
df_new = pd.DataFrame(arr, columns=['age', 'score'])

# Series to NumPy
ages = df['age'].to_numpy()  # 1D array

# USAAIO pattern: load data with Pandas, compute with NumPy
df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2', 'feature3']].to_numpy()  # (N, 3)
y = df['label'].to_numpy()                                 # (N,)
# Now use NumPy for vectorized computation
```

---

## Resources

- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Pandas Cheat Sheet (official)](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
