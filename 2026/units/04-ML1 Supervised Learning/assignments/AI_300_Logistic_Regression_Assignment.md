# AI 300 Logistic Regression, Assignment

**Beaver-Edge AI Institute**

---

## Problem 1

Consider the following Sigmoid function:

**σ(z) = 1 / (1 + exp(-z))**, ∀z ∈ ℝ

- **(a)** What is the range of σ(z)?

- **(b)** Compute σ(-∞), σ(∞), σ(0).

- **(c)** Compute 1 - σ(z).

- **(d)** Compute dσ(z)/dz. Write your solution in terms of σ(z).

- **(e)** Compute d²σ(z)/dz². Write your solution in terms of σ(z).

- **(f)** Discuss the monotonicity property of σ(z).

- **(g)** Plot σ(z).

---

## Problem 2

Consider the following binary classification model:

**ŷ = { 1 with probability σ(θ^T x̄); -1 with probability 1 - σ(θ^T x̄) }**

The ground-truth output values are 1 and -1.

- **(a)** Write down the cross entropy loss function, denoted as **L_θ**.

- **(b)** Compute **∇_θ L_θ**.

- **(c)** Compute **∇²_θ L_θ**.

- **(d)** Prove that **∇²_θ L_θ** is positive semi-definite.

---

## Problem 3

Consider the following binary classification model:

**ŷ = { 1 with probability σ(θ^T x̄); 0 with probability 1 - σ(θ^T x̄) }**

The ground-truth output values are 1 and 0.

Define a class called `MyBinaryClassLogisticRegression`.

- **(a)** The attribute is **θ**.

- **(b)** Methods include `fit`, `predict`, `score`. All these mimic those Sklearn models, except `score` has an input that allows you to choose between outputting an average score or an F1 score.

---

## Problem 4

- **(a)** Find any dataset for binary classification (e.g., Breast Cancer Wisconsin, Titanic, Pima Indians Diabetes).

- **(b)** Pre-process data in an appropriate way (e.g., removing missing data, one-hot encoding for categorical data, normalizing numerical features).

- **(c)** Split the dataset into the training dataset and test dataset.

- **(d)** Train your model defined in Problem 3.

- **(e)** Test your model with the test dataset.

---

## Problem 5

There exists a function **f: ℝ^K → ℝ**, such that **∇_z f(z) = Softmax(z)**.

What is **f**?

---

## Problem 6

Mathematically compute:

**∂ Softmax_i(z) / ∂z_j**

Your solution should be expressed in terms of the softmax function.

---

## Problem 7

This is a coding task.

Let **N** be the number of samples and **K** be the number of labels.

Let **z** be with shape (N, K).

- **(a)** Write code to compute **Softmax(z)**, denoted as `softmax_val`. The shape is (N, K).

- **(b)** Write code to compute **∇_z Softmax(z)**, denoted as `D_softmax_val`. The shape is (N, K, K). Thus, `D_softmax_val[n, i, j]` refers to **∂ Softmax_i(z^(n)) / ∂z_j**.

In your solution, avoid using any loop.

---

## Problem 8

Consider the following multi-class logistic regression model.

Let **x̄ ∈ ℝ^d**. Let **θ_k ∈ ℝ^d** for k ∈ {0, ⋯, K-1}.

Denote:

**Θ = [θ₀^T; θ₁^T; ...; θ_(K-1)^T] ∈ ℝ^(K×d)**

Then the model predicts:

**ŷ = k with probability Softmax_k(Θx̄)**

- **(a)** Write down the cross-entropy loss function, denoted as **L_Θ = (1/N) Σ(n=0 to N-1) L_Θ^(n)**.

- **(b)** Mathematically compute **∇_Θ L_Θ**. The shape is (K, d).

---

## Problem 9

In this problem, you are asked to prove that **L_Θ^(n)** is convex (positive semi-definite Hessian).

**Hint:** If f(x) is a convex function, then f(Ax + b) is also convex.

---

## Problem 10

Define a class called `MyMultiClassLogisticRidgeRegression`.

- **(a)** In training, the loss function is the sum of the cross-entropy and ridge regularization (not regularized on intercepts).

- **(b)** The attributes are **Θ** (learnable parameters) and **weight decay** (hyperparameter).

- **(c)** Methods include `fit`, `predict`, `score`. All these mimic those Sklearn models, except `score` has an input that allows you to choose between outputting an average score or an F1 score.

---

## Problem 11

- **(a)** Find any dataset for multi-class classification (e.g., MNIST, CIFAR10).

- **(b)** Pre-process data in an appropriate way (e.g., one-hot encoding for categorical data, normalizing numerical features).

- **(c)** Split the dataset into the training dataset and test dataset.

- **(d)** Train with the multi-class logistic regression with ridge regularization model, defined in Problem 10.

- **(e)** Do K-fold cross validation on the above training dataset to optimize over the hyperparameter (weight decay).

- **(f)** After optimizing your hyperparameter and learnable parameters, compute the accuracy score and the F1 score with the test dataset.

---

## Problem 12

Define:

**f_i = exp(z_i/τ) / Σ(j=0 to N-1) exp(z_j/τ)**

- **(a)** Compute **lim(τ→0⁺) f_i**.

- **(b)** Compute **lim(τ→∞) f_i**.

- **(c)** Explain the intuition of your results above.

---

*Copyright © Beaver-Edge AI Institute. All Rights Reserved. No part of this document may be copied or reproduced without the written permission of Beaver-Edge AI Institute.*
