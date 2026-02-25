# Advanced Python for AI

**Prerequisites**: Python fundamentals (variables, loops, functions, data structures)
**USAAIO Relevance**: Clean, Pythonic code is faster to write under time pressure. Many AI libraries use decorators, generators, and lambdas heavily. Understanding these patterns lets you read library source code and write concise solutions.

---

## Discovery

### Why Python Became the Language of AI

In the early 2000s, AI researchers used MATLAB, R, and C++. Python was a scripting language. What changed?

1. **NumPy (2005)** gave Python fast array operations by wrapping C/Fortran libraries
2. **scikit-learn (2007)** made ML accessible with a clean API
3. **Theano (2010)**, then **TensorFlow (2015)** and **PyTorch (2016)** chose Python as their frontend
4. **Community effect**: once researchers adopted Python, everyone followed

The key insight: Python is slow, but it does not matter. Python is the **glue language** -- it orchestrates fast C/C++ code underneath. A single NumPy call `np.dot(A, B)` runs optimized BLAS routines in C. You get the speed of C with the readability of Python.

> **Socratic question**: If Python is slow, why not just use C++ directly? Think about what a researcher's day looks like -- writing code, debugging, iterating on ideas. What matters more: execution speed or development speed?

### The Pythonic Way

Python has a philosophy: there should be **one obvious way** to do things. AI code follows this -- there are idiomatic patterns that experienced Python programmers recognize instantly. This section teaches those patterns.

---

## Intuition

### List Comprehensions: Loops in Disguise

A list comprehension is a single-expression loop. It is not just shorter -- it is **faster** because Python optimizes the internal bytecode.

```
Traditional loop:              Comprehension:
┌─────────────────────┐        ┌──────────────────────────┐
│ result = []         │        │ result = [x**2           │
│ for x in data:      │   =>   │           for x in data  │
│     result.append(  │        │           if x > 0]      │
│         x**2)       │        └──────────────────────────┘
└─────────────────────┘
```

Read comprehensions inside-out: **what** (expression) <- **where** (for clause) <- **when** (if clause).

### Generators: Lazy Sequences

A generator produces values **one at a time** instead of building an entire list in memory:

```
List (eager):     [0, 1, 4, 9, 16, ...]  <- entire list in memory
Generator (lazy): (next?) -> 0 -> (next?) -> 1 -> (next?) -> 4 -> ...
```

Use generators when you only need to iterate once, or when the data is too large to fit in memory.

### Decorators: Functions That Modify Functions

```
┌──────────────────────┐
│     @timer           │  <- decorator wraps the function
│     def train():     │
│         ...          │
│                      │
│  Equivalent to:      │
│  train = timer(train)│
└──────────────────────┘
```

A decorator takes a function as input and returns a new function with added behavior (timing, logging, caching, etc.).

---

## Math

No heavy math in this section -- advanced Python is about **patterns**, not formulas. However, understanding computational complexity helps:

| Pattern | Time Complexity | Space Complexity |
|---------|----------------|-----------------|
| List comprehension | O(n) | O(n) -- builds full list |
| Generator expression | O(n) | O(1) -- yields one at a time |
| `map()` + `lambda` | O(n) | O(1) if consumed lazily |
| Dictionary comprehension | O(n) | O(n) |

---

## Code

### List Comprehensions

```python
# Basic: square each element
squares = [x**2 for x in range(10)]

# Filtered: only even squares
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested: flatten a 2D list
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [val for row in matrix for val in row]  # [1, 2, 3, 4, 5, 6]
# Read as: for each row, for each val in row, take val

# Dictionary comprehension
word_lengths = {word: len(word) for word in ['numpy', 'pandas', 'matplotlib']}
# {'numpy': 5, 'pandas': 6, 'matplotlib': 10}

# Set comprehension
unique_lengths = {len(word) for word in ['cat', 'dog', 'rat']}  # {3}

# Conditional expression (ternary) inside comprehension
labels = ['pos' if x > 0 else 'neg' for x in [-1, 2, -3, 4]]
# ['neg', 'pos', 'neg', 'pos']
```

### Generators

```python
# Generator expression -- note parentheses instead of brackets
squares_gen = (x**2 for x in range(1_000_000))
# No memory allocated for 1M items -- produces them on demand

# Generator function with yield
def fibonacci(n):
    """Yield first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci(10):
    print(num, end=' ')  # 0 1 1 2 3 5 8 13 21 34

# Practical: reading large files line by line
def read_large_csv(filepath):
    """Yield rows from a CSV without loading entire file."""
    with open(filepath) as f:
        header = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))
```

### Lambda Functions

```python
# Lambda: anonymous single-expression function
square = lambda x: x ** 2
square(5)  # 25

# Useful with map, filter, sorted
data = [3, 1, 4, 1, 5, 9]
sorted_data = sorted(data, key=lambda x: -x)  # [9, 5, 4, 3, 1, 1]

# With map and filter
doubled = list(map(lambda x: x * 2, data))      # [6, 2, 8, 2, 10, 18]
positives = list(filter(lambda x: x > 3, data))  # [4, 5, 9]

# In pandas -- very common pattern
# df['col'].apply(lambda x: x.strip().lower())
```

### Map, Filter, Reduce

```python
from functools import reduce

data = [1, 2, 3, 4, 5]

# map: apply function to every element
squares = list(map(lambda x: x**2, data))  # [1, 4, 9, 16, 25]

# filter: keep elements where function returns True
evens = list(filter(lambda x: x % 2 == 0, data))  # [2, 4]

# reduce: accumulate with a binary function
total = reduce(lambda acc, x: acc + x, data)  # 15
product = reduce(lambda acc, x: acc * x, data)  # 120

# In practice, prefer comprehensions over map/filter:
squares = [x**2 for x in data]          # clearer than map
evens = [x for x in data if x % 2 == 0] # clearer than filter
```

### Decorators

```python
import time
from functools import wraps

def timer(func):
    """Decorator that prints execution time."""
    @wraps(func)  # preserves func.__name__ and __doc__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # Prints: slow_function took 1.00xxs

# Decorator with arguments
def repeat(n):
    """Decorator that calls function n times."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    return f"Hello, {name}!"

greet("USAAIO")  # ['Hello, USAAIO!', 'Hello, USAAIO!', 'Hello, USAAIO!']
```

### Context Managers

```python
# Built-in: file handling
with open('data.csv') as f:
    data = f.read()
# File is automatically closed, even if exception occurs

# Custom context manager with __enter__ / __exit__
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False  # don't suppress exceptions

with Timer() as t:
    # ... your code here ...
    total = sum(range(1_000_000))

# Using contextlib (simpler)
from contextlib import contextmanager

@contextmanager
def timer_ctx():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.4f}s")

with timer_ctx():
    total = sum(range(1_000_000))
```

### Type Hints

```python
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import numpy.typing as npt

# Basic type hints
def normalize(data: list[float]) -> list[float]:
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# NumPy type hints
def compute_mean(arr: npt.NDArray[np.float64]) -> np.float64:
    return np.mean(arr)

# Complex types
def load_dataset(
    path: str,
    columns: list[str] | None = None,
    transform: Callable[[float], float] | None = None
) -> dict[str, list[float]]:
    ...

# Type hints don't enforce types at runtime -- they're for
# documentation, IDE support, and static analysis tools (mypy).
```

### Putting It All Together: AI-Style Python

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timer
def preprocess(data: list[dict]) -> list[dict]:
    """Clean and transform raw records."""
    return [
        {k: v.strip().lower() if isinstance(v, str) else v
         for k, v in record.items()}
        for record in data
        if record.get('label') is not None  # filter incomplete records
    ]

# Generator for batch processing
def batch_iter(data: list, batch_size: int = 32):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Usage pattern you'll see in ML code
raw_data = [{'text': ' Hello ', 'label': 1}, {'text': 'World', 'label': 0}]
clean = preprocess(raw_data)
for batch in batch_iter(clean, batch_size=1):
    print(batch)
```

---

## Resources

- [Python Official Tutorial -- Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- [Real Python -- Decorators](https://realpython.com/primer-on-python-decorators/)
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
