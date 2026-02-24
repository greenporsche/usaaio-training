# AI 200 Linear Algebra: Eigenvalue Decomposition Assignment

**Beaver-Edge AI Institute**  
Website: www.beaver-edge.ai  
Email: info@beaver-edge.ai  
WeChat ID: beaver-AI

---

## Problem 1

Let $A \in \mathbb{R}^{N \times N}$ be a real-valued square matrix. Suppose it has $N$ distinct eigenvalues $\lambda_0, \ldots, \lambda_{N-1} \in \mathbb{C}$ and $N$ associated eigenvectors $x_0, \ldots, x_{N-1} \in \mathbb{C}^{N \times 1}$.

**Prove with math induction that all these eigenvectors are linearly independent.**

---

## Problem 2

For real-valued square matrix $A \in \mathbb{R}^{N \times N}$, denote

$$Q = [q_0 \quad q_1 \quad \cdots \quad q_{N-1}]$$

where each column $q_n \in \mathbb{R}^{N \times 1}$ is an eigenvector associated with eigenvalue $\lambda_n$.

Denote

$$\Lambda = \begin{bmatrix} \lambda_0 & & \\ & \ddots & \\ & & \lambda_{N-1} \end{bmatrix}$$

Prove the following statements:

**(a)** $A = Q\Lambda Q^{-1}$

**(b)** Define $P = [p_0 \quad \cdots \quad p_{N-1}] = Q^{-1,\top}$

Then 

$$A = \sum_{n=0}^{N-1} \lambda_n q_n p_n^\top$$

---

## Problem 3

Let a real-valued square matrix $A \in \mathbb{R}^{N \times N}$ be with the following eigenvalue decomposition:

$$A = Q\Lambda Q^{-1}$$

Define

$$P = [p_0 \quad \cdots \quad p_{N-1}] = Q^{-1,\top}$$

Prove the following statements:

**(a)** $A^\top$ has the following eigenvalue decomposition: $A^\top = P\Lambda P^{-1}$

**(b)** We have

$$A^\top p_n = \lambda_n p_n, \quad \forall n \in \{0, \ldots, N-1\}$$

---

## Problem 4

Let

$$A = \begin{bmatrix} 2 & -1 & 1 & 5 \\ 0 & 1 & 3 & 1 \\ 0 & 0 & 4 & 7 \\ 0 & 0 & 0 & -1 \end{bmatrix}$$

**(a)** Manually compute eigenvalues and right and left eigenvectors of this eigenvalue equation. (In case there are eigenvectors associated with the same eigenvalue, ensure these eigenvectors are orthogonal to each other.)

**(b)** Manually do eigendecomposition of $A$.

---

## Problem 5

**(a)** Given $A$ in Problem 4, use the eigendecomposition found in that problem to compute $4A^3 - 3A^2 + 5A + 1$.

**(b)** Verify your solution with `np.linalg.matrix_power`.

---

## Problem 6

Given $A$ in Problem 4, use the eigendecomposition found in that problem to solve the following differential equation:

$$\frac{dx}{dt} = Ax$$

with the initial value $x(0)$.

---

## Problem 7

Consider the following Markov transitional matrix from states in period $t$ to states in period $t + 1$:

$$P = \begin{bmatrix} 0.2 & 0.3 & 0.5 \\ 0.4 & 0.5 & 0.1 \\ 0.6 & 0.2 & 0.2 \end{bmatrix}$$

That is, $p_{ij} = P(X_{t+1} = j \mid X_t = i)$, for all $t \geq 0$.

Suppose at time 0, the probability of being at state $i$ is $p_{0,i}$. Denote

$$p_0 = \begin{bmatrix} p_{0,0} \\ p_{0,1} \\ p_{0,2} \end{bmatrix}$$

**(a)** Prove by math induction that the probability distribution vector on different states in period $t$ is given by

$$p_t = P^t p_0$$

**(b)** Do eigendecomposition of matrix $P$ (Hint: One eigenvalue is 1).

**(c)** Use the eigendecomposition of matrix $P$ to derive a closed-form of $p_t$.

**(d)** Compute

$$\lim_{t \to \infty} p_t$$

Does your solution depend on the initial condition $p_0$? Please interpret this result from the geometric perspective of the eigendecomposition.

---

## Problem 8

**This is a coding task.**

Use NumPy to write a function called `power_dom_eigen` that uses the power method to compute the dominant eigenvalue of a real-valued square matrix:

**(a) Input:** a real-valued square matrix with shape `(N,N)`. You are guaranteed that the input matrix has all real and distinct eigenvalues.

**(b) Output:** three NumPy objects:
   - i. Eigenvalue with dimension 0
   - ii. Right eigenvector with shape `(N,)`
   - iii. Left eigenvector with shape `(N,)`

---

## Problem 9

**This is a coding task.**

Use NumPy to write a function called `power_top_eigen` that uses the power method and the deflated matrix formula to compute all eigenvectors.

**(a) Input:**
   - i. A real-valued square matrix with shape `(N,N)`
   - ii. The number top eigenvalues that shall be computed (in terms of their absolute values), with the argument name `K`
   - iii. You are guaranteed that the input matrix has all real and distinct eigenvalues

**(b) Output:** three NumPy objects:
   - i. Top eigenvalues with shape `(K,)`
   - ii. Associated right eigenvectors with shape `(K,N)`
   - iii. Associated left eigenvector with shape `(K,N)`

**(c)** In the body of this function, you can call function `power_dom_eigen` that you defined in Problem 8.

---

## Problem 10

**(a)** Use the function you defined in Problem 9 to do eigendecomposition of the matrix defined in Problem 7.

**(b)** Verify your solution with `np.linalg.eig`.

---

*Copyright © Beaver-Edge AI Institute. All Rights Reserved. No part of this document may be copied or reproduced without the written permission of Beaver-Edge AI Institute.*
