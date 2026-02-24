# 2026 USAAIO Round 1 Sample Problems

**USA AI Olympiad (USAAIO)**  
Version: January 1, 2026

---

## Problem 1

Consider a sample data point **x ∈ ℝ^d**. Let **ê ∈ ℝ^d** be a principal component with unit length.

### Part 1.1

Let d = 3. Consider vector **v** = [1, -2, 3]^T. Let **ê** be in the same direction as **v**. We can write **ê** in the following form:

**ê** = (1/√a) · [b, c, d]^T

where b, c, d are integers and a is not a perfect square. What is the value of **a + b + c + d**?

- A. 0
- B. 12
- C. 14
- D. 16
- E. 20

### Part 1.2

Let **x** = [-1, -4, 6]^T. Compute the projection of x onto ê. Your answer can be written as √b/a, where b is an integer and a is not a perfect square. What is the value of **a + b**?

- A. 14
- B. -6
- C. -11
- D. 34
- E. 39

### Part 1.3

Let **r** be the residual after x is projected onto ê. We can write r in the following form:

**r** = (1/a) · [b, c, d]^T

where b, c, d are integers, a is a positive integer and gcd(a, b, c, d) = 1. What is the value of **a + b + c + d**?

- A. 14
- B. -22
- C. -36
- D. 50
- E. 64

---

## Problem 2

Consider matrix **A ∈ ℝ^(4×4)**. For any vector **x** = [x₀, x₁, x₂, x₃]^T, we have:

**Ax** = [x₂, x₃, x₁, x₀]^T

### Part 2.1

What operation does A perform on x?

- A. Rotation
- B. Reflection
- C. Permutation
- D. Dilation
- E. Translation

### Part 2.2

Write A in matrix form.

*Reasoning is not required.*

### Part 2.3

We can write A in the following decomposition form:

**A** = Σ(i=0 to 3) ê^(f(i)) · ê^(i),T

where **ê^(i) ∈ ℝ^4** is a column unit vector whose i-th component is equal to 1 and all other components are equal to 0, and f(i) ∈ {0, 1, 2, 3} for i ∈ {0, 1, 2, 3}.

Compute f(0), f(1), f(2) and f(3).

*Reasoning is not required.*

---

## Problem 3

### Part 3.1

Consider two regularized linear regression models trained on the same dataset:
- Model A uses L1-regularization
- Model B uses L2-regularization
- Both use the same regularization strength λ > 0

Which statement is most likely true?

- A. Models A and B have the same number of non-zero weights
- B. Model A has fewer non-zero weights than Model B
- C. Model B has fewer non-zero weights than Model A
- D. Neither model can produce zero weights
- E. There is no way to know which model has fewer non-zero weights

### Part 3.2

In supervised machine learning, as you increase the complexity of a model (for example, by increasing the degree of a polynomial in a regression model), which of the following best describes the typical behavior of the error components due to bias and variance?

- A. Both Bias and Variance increase
- B. Both Bias and Variance decrease
- C. Bias and Variance remain constant regardless of complexity
- D. Bias decreases and Variance increases
- E. Bias increases and Variance decreases

---

## Problem 4

Copy the following code:

```python
import numpy as np
```

Do the following coding tasks:

### Part 4.1
Generate a NumPy array with shape (5, 8, 3, 1, 2). Each entry is a standard normal. Use random seed 2026.

### Part 4.2
Remove the dimension whose length is 1.

### Part 4.3
Insert a new dimension at axis 2.

### Part 4.4
Swap axes 0 and 1.

### Part 4.5
For those entries whose values are above 1, reset their values as 100.

### Part 4.6
Flatten the array.

---

## Problem 5

Consider a dataset with N samples {**x^(n) ∈ ℝ**}_(n=0 to N-1) with N ≥ 1000. Let **φ(x^(n)) ∈ ℝ^d** be a feature function of **x^(n)**.

Define the kernel function:

**κᵢⱼ** = φ(x^(i))^T · φ(x^(j))

Define kernel matrix **K**:

**K** = [κᵢⱼ] for i, j = 0 to N-1

### Part 5.1

Suppose **κᵢⱼ = 1 + x^(i)x^(j) + (x^(i)x^(j))²**. Compute φ(x).

*Reasoning is not required.*

### Part 5.2

Suppose **κᵢⱼ = (1 + x^(i)x^(j) + 2(x^(i)x^(j))²)²**. Compute φ(x).

*Reasoning is not required.*

### Part 5.3

Suppose **κᵢⱼ = (1 + x^(i)x^(j) + 2(x^(i)x^(j))²)²**. Compute the rank of K.

*Reasoning is required.*

### Part 5.4

Let **Φ** = [φ(x^(0))^T; φ(x^(1))^T; ...; φ(x^(N-1))^T] ∈ ℝ^(N×d).

Suppose Φ has the following singular-value-decomposition (SVD):

**Φ = UΣV^T**

Write the trace and the determinant of K in terms of the SVD of Φ.

*Reasoning is required.*

### Part 5.5

This is a coding task.

Copy the following code:

```python
import numpy as np
```

Write a function in the following way:
- The input is {**x^(n) ∈ ℝ**}_(n=0 to N-1) (a 1-dim NumPy array with shape (N,))
- The return is the kernel matrix K with **κᵢⱼ = (1 + x^(i)x^(j) + 2(x^(i)x^(j))²)²**

In your code, do NOT use any loop. Do NOT use np.linalg.

---

## Problem 6

Compute the following derivatives.

### Part 6.1

d(tanh x)/dx

### Part 6.2

∇_θ Σ(n=0 to N-1) (y^(n) - θ^T x^(n))²

where **θ, x^(n) ∈ ℝ^d** and **y^(n) ∈ ℝ**.

*Reasoning is not required.*

---

## Problem 7

Consider a dataset that consists of two samples (x^(0), 1) and (x^(1), -1), where **x^(0), x^(1) ∈ ℝ^d**.

Define a separating hyperplane in the form:

**θ̂^T x + b = 0**

where **θ̂ ∈ ℝ^d** is a unit vector, **b ∈ ℝ**, x^(0) is in the upper half of this hyperplane and x^(0) and x^(1) have equal distance to this hyperplane.

### Part 7.1

Let x^(0) = (-3, 0) and x^(1) = (3, 0).

Compute θ̂ and b.

*Reasoning is not required.*

### Part 7.2

Let x^(0) = (5, 7) and x^(1) = (-3, 2).

Compute θ̂ and b.

*Reasoning is not required.*

### Part 7.3

This is a coding task.

Write a function to compute θ̂ and b.

In this function:
- The input is a NumPy array with shape (2, d)
- In the output, θ̂ is a NumPy array with shape (d,) and b is a NumPy array with shape ()

---

## Problem 8

Consider the following confusion matrix:

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 45 (TP)            | 5 (FN)             |
| **Actual Negative**  | 10 (FP)            | 40 (TN)            |

Do the following tasks. *Reasoning is not required.*

### Part 8.1
Compute accuracy.

### Part 8.2
Compute precision scores for positive and negative classes, respectively.

### Part 8.3
Compute recall scores for positive and negative classes, respectively.

### Part 8.4
Compute F1-scores for positive and negative classes, respectively.

---

## Problem 9

In this problem, we study the **Bank Marketing dataset**. You can load the dataset by using the following code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url_part1 = "https://huggingface.co/datasets/usaaio-official/"
url_part2 = "2026_USAAIO_samples/raw/main/"
url_part3 = "2026_USAAIO_samples_bank.csv"
url = url_part1 + url_part2 + url_part3
df = pd.read_csv(url, sep=";")
```

Do the following tasks:

### Part 9.1
Print the data type of df.

### Part 9.2
Print the shape of df.

### Part 9.3
Print all column names.

### Part 9.4
For each column, print its name and the data type of that column.

### Part 9.5
Convert data type "object" to "category".

### Part 9.6
Print the first 10 rows.

### Part 9.7
Some entry values are "unknown". Suppose you interpret as missing value. Count the number of missing values in each column.

### Part 9.8
In this part, you do not need to consider the last column.

For each column that is numeric, normalize values in this column between 0 and 1. After normalization, print out the following statistics of this column: max value, min value, mean value, standard deviation.

In your solution, you are not allowed to directly use any existing normalization function. That is, you need to do this task from scratch.

### Part 9.9
In this part, you do not need to consider the last column.

For each column that is categorical, do one hot encoding. Below is an example: Suppose one categorical column has name "ABC". All possible values are "X", "Y", "Z". Then after one-hot encoding, you should create three new columns called "ABC_X", "ABC_Y", and "ABC_Z".

In your solution, you are not allowed to directly use any existing one-hot encoding function. That is, you need to do this task from scratch.

### Part 9.10
Column "y" is the target. How many target values are "yes" and how many are "no"?

### Part 9.11
Consider those whose "marital" is "married" and "age" is odd.

Among these people, how many target values are "yes" and how many are "no"?

### Part 9.12
Put ages into 10 bins that are evenly split. For each bin, compute the subscription ratio. That is, within a given bin, the ratio of the number of "y" that are "yes" to the number of people in that bin.

Generate a plot of the ratio vs. age bin.

### Part 9.13
Define X to be with all features (no column "y") and y with only column "y".

### Part 9.14
Split the dataset into training and test datasets, where 80% of data is used for training.

Do random splitting. The random seed shall be 2026.

---

## Problem 10

Build a ReLU module that subclasses `torch.nn.Module` and is named `My_ReLU`.

---

## Problem 11

Use PyTorch to build a multi-layer perceptron model that does the following inference:

On a 2-dim plane, consider a triangle enclosed by (0, 0), (1, 0), (0, 2). For your model, each input sample is a data point on a 2-dim plane. The output value is 1 (resp. 0) if this data point is (resp. is not) an interior point of this triangle.

Your model architecture and parameter values shall be ready for doing this inference. That is, there is **no training** in this problem.

While doing inference, the input is a tensor with shape (B, 2), where B is a batch size.

Since this problem does not require you to do training, please feel free to use the following threshold function as your activation function:

**Θ(x)** = 1 if x ≥ 0; 0 if x < 0

---

## Problem 12

Run the following code:

```python
from torchvision.models import resnet34
import torchsummary

model = resnet34(pretrained=True)
```

### Part 12.1
What is the total number of learnable parameters in model?

*Hint: For an object with type `torch.nn.parameter.Parameter`, you can use the method `numel()` to count the number of learnable parameters in it.*

### Part 12.2
Suppose the input has shape (B, 3, 224, 224). What is the shape of the output from `model.layer3`?

### Part 12.3
What is the total number of learnable parameters in all convolutional modules in `model.layer2[0]`?

To solve this problem, you are not allowed to use the method `numel()` for an object with type `torch.nn.parameter.Parameter`.

*Reasoning is required.*

### Part 12.4
We want to use the pretrained resnet34 model as a backbone for a classification task with 12 labels.

Build such a model.

In your solution, all parameters in the backbone shall be frozen (that is, they are fixed and not learnable).

---

## Problem 13

In this problem, we study the **Breast Cancer dataset**, which is a binary classification task. All features are numeric.

You can access the training dataset by running the following code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url_part1 = "https://huggingface.co/datasets/usaaio-official/"
url_part2 = "2026_USAAIO_samples/raw/main/"
url_part3 = "2026_USAAIO_samples_breast_cancer_train.csv"
url = url_part1 + url_part2 + url_part3
df = pd.read_csv(url)

X = df.drop("target", axis=1)
y = df["target"]
```

In this training dataset, X contains 30 input features, and y contains the binary target labels.

We also have a **hidden test dataset** that you cannot access during the competition.

### Submission Requirements

You must submit a single Jupyter notebook (.ipynb) containing your complete solution, including (but not limited to):

1. Data preprocessing (if needed)
2. Model construction
3. Model training
4. Inference logic

### Inference Requirements

For inference, you must define a function with the following signature:

```python
def my_prediction(X_test):
    ###INSERT YOUR CODE HERE###
    return y_pred
```

In this function:
1. `X_test` is a pandas DataFrame containing all input features from the hidden test set
2. `y_pred` must be a pandas Series containing your predicted labels

After the competition, we will execute all code in your submitted notebook from top to bottom. During evaluation, we will load the hidden test features as `X_test` and call your function:

```python
my_prediction(X_test)
```

Your predictions `y_pred` will be evaluated using the **macro-averaged F1 score (F1-macro)**.

### Model Constraints

In your solution, you must use **k-Nearest Neighbors (kNN)** as part of your classification approach.

However, this does not imply that you must directly apply kNN to the raw training and test data. You may apply any data preprocessing, feature engineering, or pipeline you find appropriate.

You may use any module from **scikit-learn**.

**Do not use deep neural network** to solve this problem.

### Notebook Requirements

In your submitted notebook, please ensure the following:

1. Your code includes sufficient comments to make it readable
2. At the end of the notebook, include a text cell summarizing:
   - Your overall approach
   - The intuition behind your design choices
   - Any alternative approaches or models you considered but did not pursue

Your report does not need to be long, but it should be clear, concise, and well-reasoned.

### Grading Criteria

Your submission will be evaluated based on:

1. Whether the entire notebook runs successfully from start to finish
2. Performance on the hidden test dataset
3. The quality of your reasoning and problem-solving approach

We do **not** evaluate code style or quality.
