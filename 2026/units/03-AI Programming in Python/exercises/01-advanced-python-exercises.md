# Advanced Python Exercises

**Time target**: 2-5 minutes each | **Total**: 6 exercises

---

## Exercise 1: List Comprehension -- Flatten and Filter

What does this code output?

```python
matrix = [[1, -2, 3], [-4, 5, -6], [7, -8, 9]]
result = [x ** 2 for row in matrix for x in row if x > 0]
print(result)
```

<details>
<summary>Solution</summary>

```python
[1, 9, 25, 49, 81]
```

**Explanation**: The comprehension iterates outer-to-inner: `for row in matrix` then `for x in row`. The filter `if x > 0` keeps only positive values (1, 3, 5, 7, 9), then squares each: 1, 9, 25, 49, 81.

The reading order is: expression <- outer loop <- inner loop <- filter.
</details>

---

## Exercise 2: Generator vs List

What is the key difference between these two?

```python
list_result = [x ** 2 for x in range(1_000_000)]
gen_result = (x ** 2 for x in range(1_000_000))
```

Now: what does `sum(gen_result)` return? What happens if you call `sum(gen_result)` again?

<details>
<summary>Solution</summary>

`list_result` creates a list of 1 million elements in memory immediately (~8 MB).

`gen_result` creates a generator that produces values on demand -- uses almost no memory.

`sum(gen_result)` returns `333332833333500000` (sum of squares 0 through 999999).

Calling `sum(gen_result)` **a second time** returns `0`, because generators are single-use. Once exhausted, they produce no more values. A list can be iterated multiple times; a generator cannot.

</details>

---

## Exercise 3: Lambda and Sorting

What does this code output?

```python
students = [
    {'name': 'Alice', 'score': 92, 'age': 20},
    {'name': 'Bob', 'score': 85, 'age': 22},
    {'name': 'Carol', 'score': 92, 'age': 19},
    {'name': 'Dave', 'score': 78, 'age': 21},
]

result = sorted(students, key=lambda s: (-s['score'], s['age']))
print([s['name'] for s in result])
```

<details>
<summary>Solution</summary>

```python
['Carol', 'Alice', 'Bob', 'Dave']
```

**Explanation**: The `key` function returns a tuple `(-score, age)`. Python sorts tuples lexicographically:
- First by `-score` (ascending, so highest score first): Alice and Carol tie at -92
- Then by `age` (ascending, so youngest first): Carol (19) before Alice (20)
- Bob (-85) comes next, then Dave (-78)

This is a common pattern for multi-criteria sorting. Negating a numeric key reverses its sort order.
</details>

---

## Exercise 4: Decorator Behavior

What does this code print?

```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"result: {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

x = add(3, 4)
print(f"x = {x}")
```

<details>
<summary>Solution</summary>

```
calling add
result: 7
x = 7
```

**Explanation**: `@logger` replaces `add` with `wrapper`. When `add(3, 4)` is called:
1. `wrapper(3, 4)` runs
2. Prints "calling add"
3. Calls the original `add(3, 4)` which returns 7
4. Prints "result: 7"
5. Returns 7

Then `print(f"x = {x}")` prints "x = 7".

Note: `func.__name__` still says "add" because it references the original function object. To make `wrapper.__name__` also say "add", use `@functools.wraps(func)`.
</details>

---

## Exercise 5: Map, Filter, and Reduce

Rewrite this code using (a) map/filter and (b) a list comprehension:

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

result = []
for n in numbers:
    if n % 3 == 0:
        result.append(n ** 2)

print(result)
```

<details>
<summary>Solution</summary>

Output: `[9, 36, 81]`

**(a) Using map/filter:**
```python
result = list(map(lambda n: n ** 2, filter(lambda n: n % 3 == 0, numbers)))
```

**(b) Using list comprehension (preferred in Python):**
```python
result = [n ** 2 for n in numbers if n % 3 == 0]
```

The list comprehension is clearly more readable. Use map/filter when you already have named functions to pass, but prefer comprehensions in most cases.

</details>

---

## Exercise 6: Context Manager

What does this code print? What happens if the division raises an exception?

```python
class TrackTime:
    def __enter__(self):
        self.log = []
        self.log.append("entered")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.append("exited")
        if exc_type is not None:
            self.log.append(f"error: {exc_val}")
        print(" -> ".join(self.log))
        return True  # suppress exceptions

with TrackTime() as t:
    t.log.append("working")
    x = 10 / 0
    t.log.append("done")  # does this run?

print("after block")
```

<details>
<summary>Solution</summary>

```
entered -> working -> exited -> error: division by zero
after block
```

**Explanation**:
1. `__enter__` runs: log = ["entered"], returns `self`
2. `t.log.append("working")`: log = ["entered", "working"]
3. `x = 10 / 0` raises `ZeroDivisionError`
4. `t.log.append("done")` is **never reached** (exception jumps to `__exit__`)
5. `__exit__` runs with `exc_type=ZeroDivisionError`, `exc_val="division by zero"`
6. Appends "exited" and "error: division by zero"
7. `return True` **suppresses** the exception, so execution continues
8. "after block" prints normally

If `__exit__` returned `False` (or `None`), the `ZeroDivisionError` would propagate and "after block" would not print.
</details>
