# Pandas Exercises

**Time target**: 2-5 minutes each | **Total**: 6 exercises

---

## Exercise 1: loc vs iloc

Given:

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
    'score': [92, 85, 78, 95, 88],
    'team': ['A', 'B', 'A', 'B', 'A']
}, index=[10, 20, 30, 40, 50])
```

What does each expression return?

```python
r1 = df.loc[10]
r2 = df.iloc[0]
r3 = df.loc[10:30]
r4 = df.iloc[0:2]
r5 = df.loc[10:30, 'name':'score']
r6 = df.iloc[0:2, 0:2]
```

<details>
<summary>Solution</summary>

```python
r1 = df.loc[10]
# name     Alice
# score       92
# team         A
# (Series -- row with label 10)

r2 = df.iloc[0]
# Same as r1 (position 0 happens to be label 10)

r3 = df.loc[10:30]
#     name  score team
# 10  Alice     92    A
# 20    Bob     85    B
# 30  Carol     78    A
# (3 rows -- loc is INCLUSIVE on both ends)

r4 = df.iloc[0:2]
#     name  score team
# 10  Alice     92    A
# 20    Bob     85    B
# (2 rows -- iloc is EXCLUSIVE on end, like Python slicing)

r5 = df.loc[10:30, 'name':'score']
#     name  score
# 10  Alice     92
# 20    Bob     85
# 30  Carol     78
# (loc column slicing is also INCLUSIVE)

r6 = df.iloc[0:2, 0:2]
#     name  score
# 10  Alice     92
# 20    Bob     85
# (iloc is EXCLUSIVE on both row and column end)
```

**Key takeaway**: `loc` uses labels and is inclusive. `iloc` uses integer positions and is exclusive at the end (like Python).
</details>

---

## Exercise 2: Filtering and Boolean Indexing

Given:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'],
    'math': [92, 78, 85, 95, 65, 88],
    'science': [88, 82, 90, 75, 95, 80],
    'grade': ['A', 'C', 'B', 'A', 'D', 'B']
})
```

Write one-line expressions for:

1. Students whose math score is above 85
2. Students who scored above 80 in BOTH math and science
3. The average math score of students with grade 'A'
4. The student with the highest science score

<details>
<summary>Solution</summary>

```python
# 1. Math above 85
df[df['math'] > 85]
# Returns rows for Alice (92), Dave (95), Frank (88)

# 2. Above 80 in both
df[(df['math'] > 80) & (df['science'] > 80)]
# Returns Alice (92, 88), Carol (85, 90), Frank (88, 80) -- wait, Frank's science is 80, not > 80
# Actually: Alice (92, 88) and Carol (85, 90)

# 3. Average math score of grade A students
df[df['grade'] == 'A']['math'].mean()
# (92 + 95) / 2 = 93.5

# 4. Student with highest science score
df.loc[df['science'].idxmax(), 'student']
# 'Eve' (science score 95)
# Alternative: df.sort_values('science', ascending=False).iloc[0]['student']
```

**Tip**: `idxmax()` returns the index (label) of the maximum value. Then `df.loc[idx, 'student']` gets the name at that index. This is more efficient than sorting the entire DataFrame.
</details>

---

## Exercise 3: GroupBy

Given:

```python
import pandas as pd

sales = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'East', 'West', 'East', 'West'],
    'product': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'B'],
    'revenue': [100, 150, 200, 120, 180, 250, 90, 80],
    'quantity': [10, 15, 20, 12, 18, 25, 9, 8]
})
```

What does each expression return?

```python
r1 = sales.groupby('region')['revenue'].sum()
r2 = sales.groupby(['region', 'product'])['revenue'].mean()
r3 = sales.groupby('region').agg({'revenue': 'sum', 'quantity': 'mean'})
```

<details>
<summary>Solution</summary>

```python
r1 = sales.groupby('region')['revenue'].sum()
# region
# East    520     (100 + 150 + 180 + 90)
# West    650     (200 + 120 + 250 + 80)

r2 = sales.groupby(['region', 'product'])['revenue'].mean()
# region  product
# East    A          140.0    (100 + 180) / 2
#         B          120.0    (150 + 90) / 2
# West    A          225.0    (200 + 250) / 2
#         B          100.0    (120 + 80) / 2

r3 = sales.groupby('region').agg({'revenue': 'sum', 'quantity': 'mean'})
#        revenue  quantity
# East       520     13.0    (10+15+18+9)/4
# West       650     16.25   (20+12+25+8)/4
```

**GroupBy mental model**: Split the DataFrame into groups by the key column(s), apply the aggregation function to each group independently, then combine the results.
</details>

---

## Exercise 4: Merge

Given:

```python
import pandas as pd

students = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Carol', 'Dave']
})

scores = pd.DataFrame({
    'student_id': [1, 2, 2, 3, 5],
    'subject': ['Math', 'Math', 'Sci', 'Math', 'Sci'],
    'score': [95, 82, 88, 91, 76]
})
```

What does each merge return? How many rows?

```python
r1 = pd.merge(students, scores, left_on='id', right_on='student_id', how='inner')
r2 = pd.merge(students, scores, left_on='id', right_on='student_id', how='left')
r3 = pd.merge(students, scores, left_on='id', right_on='student_id', how='outer')
```

<details>
<summary>Solution</summary>

```python
# r1: INNER join -- only rows with matches in BOTH tables
#    id  name  student_id subject  score
# 0   1  Alice          1    Math     95
# 1   2    Bob          2    Math     82
# 2   2    Bob          2     Sci     88
# 3   3  Carol          3    Math     91
# 4 rows (student 4 has no scores, student 5 not in students table)

# r2: LEFT join -- all students, NaN where no matching scores
#    id   name  student_id subject  score
# 0   1  Alice         1.0    Math   95.0
# 1   2    Bob         2.0    Math   82.0
# 2   2    Bob         2.0     Sci   88.0
# 3   3  Carol         3.0    Math   91.0
# 4   4   Dave         NaN     NaN    NaN
# 5 rows (Dave has no scores -> NaN)

# r3: OUTER join -- all rows from both tables
#     id   name  student_id subject  score
# 0  1.0  Alice         1.0    Math   95.0
# 1  2.0    Bob         2.0    Math   82.0
# 2  2.0    Bob         2.0     Sci   88.0
# 3  3.0  Carol         3.0    Math   91.0
# 4  4.0   Dave         NaN     NaN    NaN
# 5  NaN    NaN         5.0     Sci   76.0
# 6 rows (includes Dave with NaN scores AND student 5 with NaN name)
```

**Note**: Bob appears twice because he has two scores. Merge creates one row for each matching combination (like a SQL join).
</details>

---

## Exercise 5: Method Chaining

Rewrite this step-by-step code as a single chained expression:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'],
    'department': ['Eng', 'Eng', 'Sales', 'Sales', 'Eng', 'Sales'],
    'salary': [95000, 82000, 78000, 91000, 105000, 72000]
})

# Step-by-step version
filtered = df[df['salary'] > 75000]
filtered['salary_k'] = filtered['salary'] / 1000
grouped = filtered.groupby('department')['salary_k'].mean()
result = grouped.sort_values(ascending=False)
```

<details>
<summary>Solution</summary>

```python
result = (
    df
    .query('salary > 75000')
    .assign(salary_k=lambda d: d['salary'] / 1000)
    .groupby('department')['salary_k']
    .mean()
    .sort_values(ascending=False)
)
# department
# Eng      94.0      (95 + 82 + 105) / 3
# Sales    84.5      (78 + 91) / 2   (Frank filtered out)
```

**Method chaining tips**:
- Use `query()` instead of boolean indexing for cleaner chains
- Use `assign()` to add columns mid-chain (avoids SettingWithCopyWarning)
- Wrap in parentheses for multi-line readability
- Each method returns a new DataFrame/Series, enabling the chain
</details>

---

## Exercise 6: What Does This Output?

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
})

print(df.isna().sum())
print("---")
print(df.dropna().shape)
print("---")
print(df.fillna(df.mean()).round(1))
```

<details>
<summary>Solution</summary>

```
A    1
B    2
C    0
dtype: int64
---
(3, 3)
---
     A    B  C
0  1.0  3.3  1
1  2.0  2.0  2
2  3.0  3.0  3
3  4.0  3.3  4
4  5.0  5.0  5
```

**Explanation**:
- `isna().sum()`: Column A has 1 NaN (row 2), column B has 2 NaN (rows 0 and 3), column C has 0.
- `dropna()`: Drops rows with ANY NaN. Rows 0, 2, 3 have at least one NaN, so only rows 1 and 4 remain -> shape (2, 3). Wait, let me recheck: row 0 has NaN in B, row 2 has NaN in A, row 3 has NaN in B. So rows 1, 4 remain... actually row 4 has no NaN (A=5, B=5, C=5), and row 1 has no NaN (A=2, B=2, C=2). That's (2, 3), not (3, 3).

Correction: rows without NaN are 1 (A=2, B=2, C=2) and 4 (A=5, B=5, C=5). Shape is **(2, 3)**.

- `fillna(df.mean())`: mean of A = (1+2+4+5)/4 = 3.0, mean of B = (2+3+5)/3 = 3.333. NaN in A (row 2) filled with 3.0, NaN in B (rows 0, 3) filled with 3.3.

```
     A         B  C
0  1.0  3.333333  1
1  2.0  2.000000  2
2  3.0  3.000000  3
3  4.0  3.333333  4
4  5.0  5.000000  5
```

**Key insight**: `df.mean()` automatically skips NaN values (this is Pandas default behavior). The mean is computed over non-NaN values only.
</details>
