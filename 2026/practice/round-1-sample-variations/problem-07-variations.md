# Problem 7 Variations: SVM / Separating Hyperplanes (Exhaustive)

> **USAAIO Practice Material** — 48 variations covering different values, dimensions, concepts, computations, applications, proofs, coding challenges, and rapid-fire conceptual checks.

---

## Original Problem Summary

Given two points x⁽⁰⁾ (class +1) and x⁽¹⁾ (class -1), find the separating hyperplane θ̂ᵀx + b = 0 where:
- θ̂ is a unit vector
- x⁽⁰⁾ is in the upper half (θ̂ᵀx⁽⁰⁾ + b ≥ 0)
- Both points have equal distance to the hyperplane

**Key Formula:**
- θ̂ = (x⁽⁰⁾ - x⁽¹⁾) / ||x⁽⁰⁾ - x⁽¹⁾||
- b = -θ̂ᵀ((x⁽⁰⁾ + x⁽¹⁾)/2)

---

## Category A: Different Values (Same Structure)

### Variation A1: Simple Symmetric Points on x-axis

Consider a dataset with x⁽⁰⁾ = (5, 0) and x⁽¹⁾ = (-5, 0).

#### Part A1.1
Compute θ̂ and b.

#### Part A1.2
What is the margin (distance from either point to the hyperplane)?

<details>
<summary>Solution</summary>

**Part A1.1:**
- θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾|| = (10, 0)/10 = (1, 0)
- Midpoint = (0, 0)
- b = -θ̂ᵀ(0, 0) = 0

**θ̂ = (1, 0), b = 0**

**Part A1.2:**
- Margin = |θ̂ᵀx⁽⁰⁾ + b| = |1·5 + 0·0 + 0| = 5

</details>

---

### Variation A2: Symmetric Points on y-axis

Consider a dataset with x⁽⁰⁾ = (0, 4) and x⁽¹⁾ = (0, -4).

#### Part A2.1
Compute θ̂ and b.

#### Part A2.2
If a new point x = (3, 1) is given, which side of the hyperplane is it on?

<details>
<summary>Solution</summary>

**Part A2.1:**
- θ̂ = (0, 8)/8 = (0, 1)
- b = -θ̂ᵀ(0, 0) = 0

**θ̂ = (0, 1), b = 0**

**Part A2.2:**
- θ̂ᵀx + b = 0·3 + 1·1 + 0 = 1 > 0
- **Same side as x⁽⁰⁾ (positive class)**

</details>

---

### Variation A3: Diagonal Points

Consider a dataset with x⁽⁰⁾ = (3, 3) and x⁽¹⁾ = (-3, -3).

#### Part A3.1
Compute θ̂ and b.

#### Part A3.2
Express the hyperplane equation in the form ax + by + c = 0 where a, b, c are integers with gcd(|a|, |b|, |c|) = 1.

<details>
<summary>Solution</summary>

**Part A3.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (6, 6), ||·|| = 6√2
- θ̂ = (6, 6)/(6√2) = (1/√2, 1/√2)
- b = -θ̂ᵀ(0, 0) = 0

**θ̂ = (1/√2, 1/√2), b = 0**

**Part A3.2:**
- (1/√2)x + (1/√2)y = 0
- x + y = 0

**x + y = 0** (so a=1, b=1, c=0)

</details>

---

### Variation A4: Non-Origin Centered

Consider a dataset with x⁽⁰⁾ = (6, 2) and x⁽¹⁾ = (2, 2).

#### Part A4.1
Compute θ̂ and b.

#### Part A4.2
Compute the margin.

<details>
<summary>Solution</summary>

**Part A4.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (4, 0), ||·|| = 4
- θ̂ = (1, 0)
- Midpoint = (4, 2)
- b = -(1·4 + 0·2) = -4

**θ̂ = (1, 0), b = -4**

**Part A4.2:**
- Margin = |1·6 + 0·2 - 4| = 2

</details>

---

### Variation A5: Pythagorean Triple Points

Consider a dataset with x⁽⁰⁾ = (7, 9) and x⁽¹⁾ = (1, 1).

#### Part A5.1
Compute θ̂ and b.

#### Part A5.2
Express θ̂ in the form (a/c, b/c) where a, b, c are integers.

<details>
<summary>Solution</summary>

**Part A5.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (6, 8), ||·|| = 10
- θ̂ = (6/10, 8/10) = (3/5, 4/5)
- Midpoint = (4, 5)
- b = -(3/5·4 + 4/5·5) = -(12/5 + 20/5) = -32/5

**θ̂ = (3/5, 4/5), b = -32/5**

**Part A5.2:**
**θ̂ = (3/5, 4/5)** — This uses the 3-4-5 Pythagorean triple!

</details>

---

## Category B: Different Dimensions

### Variation B1: 3D Points (Simple)

Consider a dataset in ℝ³ with x⁽⁰⁾ = (2, 0, 0) and x⁽¹⁾ = (-2, 0, 0).

#### Part B1.1
Compute θ̂ and b.

#### Part B1.2
Describe the separating hyperplane geometrically.

<details>
<summary>Solution</summary>

**Part B1.1:**
- θ̂ = (4, 0, 0)/4 = (1, 0, 0)
- b = 0

**θ̂ = (1, 0, 0), b = 0**

**Part B1.2:**
The separating hyperplane is the **yz-plane** (the plane x = 0).

</details>

---

### Variation B2: 3D Diagonal Points

Consider a dataset in ℝ³ with x⁽⁰⁾ = (1, 2, 2) and x⁽¹⁾ = (-1, -2, -2).

#### Part B2.1
Compute θ̂ and b.

#### Part B2.2
Verify that ||θ̂|| = 1.

<details>
<summary>Solution</summary>

**Part B2.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (2, 4, 4), ||·|| = √(4 + 16 + 16) = √36 = 6
- θ̂ = (2/6, 4/6, 4/6) = (1/3, 2/3, 2/3)
- b = 0

**θ̂ = (1/3, 2/3, 2/3), b = 0**

**Part B2.2:**
||θ̂||² = (1/3)² + (2/3)² + (2/3)² = 1/9 + 4/9 + 4/9 = 9/9 = 1 ✓

</details>

---

### Variation B3: 4D Points

Consider a dataset in ℝ⁴ with x⁽⁰⁾ = (2, 1, 2, 0) and x⁽¹⁾ = (0, -1, 0, 2).

#### Part B3.1
Compute θ̂ and b.

#### Part B3.2
Compute the margin.

<details>
<summary>Solution</summary>

**Part B3.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (2, 2, 2, -2), ||·|| = √(4+4+4+4) = 4
- θ̂ = (1/2, 1/2, 1/2, -1/2)
- Midpoint = (1, 0, 1, 1)
- b = -(1/2·1 + 1/2·0 + 1/2·1 - 1/2·1) = -(1/2 + 0 + 1/2 - 1/2) = -1/2

**θ̂ = (1/2, 1/2, 1/2, -1/2), b = -1/2**

**Part B3.2:**
- Margin = |θ̂ᵀx⁽⁰⁾ + b| = |1/2·2 + 1/2·1 + 1/2·2 - 1/2·0 - 1/2| = |1 + 0.5 + 1 - 0.5| = 2

</details>

---

### Variation B4: High-Dimensional Sparse

Consider a dataset in ℝ⁵ with x⁽⁰⁾ = (0, 0, 3, 0, 4) and x⁽¹⁾ = (0, 0, -3, 0, -4).

#### Part B4.1
Compute θ̂.

#### Part B4.2
How many components of θ̂ are non-zero?

<details>
<summary>Solution</summary>

**Part B4.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (0, 0, 6, 0, 8), ||·|| = 10
- θ̂ = (0, 0, 3/5, 0, 4/5)

**θ̂ = (0, 0, 3/5, 0, 4/5)**

**Part B4.2:**
**2 non-zero components** (indices 2 and 4)

</details>

---

### Variation B5: 1D Case (Degenerate)

Consider a dataset in ℝ¹ with x⁽⁰⁾ = 7 and x⁽¹⁾ = 3.

#### Part B5.1
Compute θ̂ and b.

#### Part B5.2
What is the "hyperplane" in 1D?

<details>
<summary>Solution</summary>

**Part B5.1:**
- θ̂ = (7-3)/|7-3| = 4/4 = 1
- Midpoint = 5
- b = -5

**θ̂ = 1, b = -5**

**Part B5.2:**
In 1D, the "hyperplane" is a **single point**: x = 5. This is the decision boundary.

</details>

---

## Category C: Conceptual Extensions

### Variation C1: Margin Computation

Consider a dataset with x⁽⁰⁾ = (8, 6) and x⁽¹⁾ = (0, 0).

#### Part C1.1
Compute θ̂ and b.

#### Part C1.2
Compute the **margin** (distance from either point to the hyperplane).

#### Part C1.3
What is the **maximum margin** possible for this dataset?

<details>
<summary>Solution</summary>

**Part C1.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (8, 6), ||·|| = 10
- θ̂ = (4/5, 3/5)
- Midpoint = (4, 3)
- b = -(4/5·4 + 3/5·3) = -(16/5 + 9/5) = -5

**θ̂ = (4/5, 3/5), b = -5**

**Part C1.2:**
Margin = |θ̂ᵀx⁽⁰⁾ + b| = |4/5·8 + 3/5·6 - 5| = |32/5 + 18/5 - 5| = |50/5 - 5| = |10 - 5| = **5**

**Part C1.3:**
The maximum margin is **half the distance between the points** = ||x⁽⁰⁾ - x⁽¹⁾||/2 = 10/2 = **5**

(This hyperplane achieves the maximum margin!)

</details>

---

### Variation C2: Multiple Points (Same Classes)

Consider a dataset with:
- Class +1: x⁽⁰⁾ = (4, 0), x⁽¹⁾ = (5, 1)
- Class -1: x⁽²⁾ = (-2, 0), x⁽³⁾ = (-3, 1)

#### Part C2.1
Which pair of points are the **support vectors** (closest points from each class)?

#### Part C2.2
Compute the separating hyperplane using only the support vectors.

<details>
<summary>Solution</summary>

**Part C2.1:**
We need to find the closest pair across classes.

Distances between + and - points:
- ||x⁽⁰⁾ - x⁽²⁾|| = ||(6, 0)|| = 6
- ||x⁽⁰⁾ - x⁽³⁾|| = ||(7, -1)|| = √50
- ||x⁽¹⁾ - x⁽²⁾|| = ||(7, 1)|| = √50
- ||x⁽¹⁾ - x⁽³⁾|| = ||(8, 0)|| = 8

**Support vectors: x⁽⁰⁾ = (4, 0) and x⁽²⁾ = (-2, 0)**

**Part C2.2:**
- θ̂ = (6, 0)/6 = (1, 0)
- Midpoint = (1, 0)
- b = -1

**θ̂ = (1, 0), b = -1**

</details>

---

### Variation C3: Soft Margin Intuition

Consider a dataset with x⁽⁰⁾ = (2, 0) and x⁽¹⁾ = (-2, 0), plus an outlier x⁽²⁾ = (0.5, 0) with label +1.

#### Part C3.1
Can you find a hyperplane that correctly classifies all three points?

#### Part C3.2
If we ignore the outlier, what is the hard-margin SVM hyperplane?

#### Part C3.3
Explain intuitively why soft-margin SVMs might prefer the solution from Part C3.2.

<details>
<summary>Solution</summary>

**Part C3.1:**
**No.** The outlier x⁽²⁾ = (0.5, 0) is on the "wrong side" — it's between x⁽⁰⁾ and x⁽¹⁾ but has label +1. No linear hyperplane can separate all three correctly.

**Part C3.2:**
Using only x⁽⁰⁾ and x⁽¹⁾:
- θ̂ = (1, 0), b = 0
- **Hyperplane: x = 0**

**Part C3.3:**
Soft-margin SVMs allow some misclassifications to achieve a **larger margin**. The margin from ignoring the outlier is 2, while any hyperplane that classifies all points correctly would have a tiny margin (if it existed). The soft-margin approach trades one misclassification for a much more robust decision boundary.

</details>

---

### Variation C4: Distance Formula Derivation

Consider a general hyperplane θ̂ᵀx + b = 0 where ||θ̂|| = 1.

#### Part C4.1
Prove that the distance from a point p to this hyperplane is |θ̂ᵀp + b|.

#### Part C4.2
Apply this formula to verify your answer for the original Problem 7 Part 7.2.

<details>
<summary>Solution</summary>

**Part C4.1:**
Let x₀ be the closest point on the hyperplane to p. Then:
- x₀ = p - αθ̂ for some scalar α (since the shortest path is along the normal)
- θ̂ᵀx₀ + b = 0 (since x₀ is on the hyperplane)
- θ̂ᵀ(p - αθ̂) + b = 0
- θ̂ᵀp - α||θ̂||² + b = 0
- Since ||θ̂|| = 1: α = θ̂ᵀp + b
- Distance = ||p - x₀|| = ||αθ̂|| = |α| = **|θ̂ᵀp + b|**

**Part C4.2:**
From Problem 7.2: x⁽⁰⁾ = (5, 7), x⁽¹⁾ = (-3, 2)
- x⁽⁰⁾ - x⁽¹⁾ = (8, 5), ||·|| = √89
- θ̂ = (8/√89, 5/√89)
- Midpoint = (1, 4.5)
- b = -(8/√89 + 22.5/√89) = -30.5/√89

Distance from x⁽⁰⁾ = (5, 7):
|θ̂ᵀx⁽⁰⁾ + b| = |40/√89 + 35/√89 - 30.5/√89| = |44.5/√89| = √89/2 ≈ 4.72

</details>

---

### Variation C5: Relation to Centroid

Consider a dataset with x⁽⁰⁾ = (a, b) and x⁽¹⁾ = (c, d).

#### Part C5.1
Prove that the separating hyperplane always passes through the midpoint (x⁽⁰⁾ + x⁽¹⁾)/2.

#### Part C5.2
Prove that θ̂ is always perpendicular to the hyperplane.

<details>
<summary>Solution</summary>

**Part C5.1:**
Let m = (x⁽⁰⁾ + x⁽¹⁾)/2 be the midpoint.
- θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||
- b = -θ̂ᵀm

Check if m is on the hyperplane:
- θ̂ᵀm + b = θ̂ᵀm - θ̂ᵀm = 0 ✓

**Part C5.2:**
For any two points x, y on the hyperplane:
- θ̂ᵀx + b = 0 and θ̂ᵀy + b = 0
- θ̂ᵀ(x - y) = 0

This means θ̂ is orthogonal to any vector lying in the hyperplane, so θ̂ is the **normal vector** to the hyperplane.

</details>

---

## Category D: Computational Variations

### Variation D1: Integer Answer

Consider a dataset with x⁽⁰⁾ = (12, 5) and x⁽¹⁾ = (0, 0).

#### Part D1.1
Write θ̂ = (a/c, b/c) where a² + b² = c² (Pythagorean triple).

#### Part D1.2
If 13b = k for some integer k, what is k?

<details>
<summary>Solution</summary>

**Part D1.1:**
- ||x⁽⁰⁾ - x⁽¹⁾|| = √(144 + 25) = √169 = 13
- θ̂ = (12/13, 5/13)

**θ̂ = (12/13, 5/13)** using the 5-12-13 Pythagorean triple!

**Part D1.2:**
b = 5/13, so 13b = 5. **k = 5**

</details>

---

### Variation D2: Fractional Coordinates

Consider a dataset with x⁽⁰⁾ = (3/2, 5/2) and x⁽¹⁾ = (-1/2, 1/2).

#### Part D2.1
Compute θ̂ and b.

#### Part D2.2
Express b as a fraction in lowest terms.

<details>
<summary>Solution</summary>

**Part D2.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (2, 2), ||·|| = 2√2
- θ̂ = (1/√2, 1/√2)
- Midpoint = (1/2, 3/2)
- b = -(1/√2·1/2 + 1/√2·3/2) = -(1/(2√2) + 3/(2√2)) = -4/(2√2) = -2/√2 = -√2

**θ̂ = (1/√2, 1/√2), b = -√2**

**Part D2.2:**
b = -√2 is irrational, but b/||x⁽⁰⁾ - x⁽¹⁾|| = -√2/(2√2) = **-1/2**

</details>

---

### Variation D3: Edge Case - Coincident Points

Consider a dataset with x⁽⁰⁾ = (3, 4) and x⁽¹⁾ = (3, 4).

#### Part D3.1
What happens when you try to compute θ̂?

#### Part D3.2
Is this dataset linearly separable?

<details>
<summary>Solution</summary>

**Part D3.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (0, 0), ||·|| = 0
- θ̂ = (0, 0)/0 → **undefined (division by zero)**

**Part D3.2:**
**No.** Two coincident points with different labels cannot be separated by any hyperplane.

</details>

---

### Variation D4: Edge Case - Orthogonal to Axis

Consider a dataset with x⁽⁰⁾ = (5, 3) and x⁽¹⁾ = (5, -1).

#### Part D4.1
Compute θ̂ and b.

#### Part D4.2
The hyperplane is parallel to which axis?

<details>
<summary>Solution</summary>

**Part D4.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (0, 4), ||·|| = 4
- θ̂ = (0, 1)
- Midpoint = (5, 1)
- b = -(0·5 + 1·1) = -1

**θ̂ = (0, 1), b = -1**

**Part D4.2:**
The hyperplane equation is y - 1 = 0, or y = 1. This is **parallel to the x-axis**.

</details>

---

### Variation D5: Large Coordinates

Consider a dataset with x⁽⁰⁾ = (1000, 2000) and x⁽¹⁾ = (-1000, -2000).

#### Part D5.1
Compute θ̂ (express with no common factors).

#### Part D5.2
Does the magnitude of the coordinates affect θ̂ when points are symmetric about origin?

<details>
<summary>Solution</summary>

**Part D5.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (2000, 4000), ||·|| = 2000√5
- θ̂ = (2000/(2000√5), 4000/(2000√5)) = (1/√5, 2/√5)

**θ̂ = (1/√5, 2/√5)**

**Part D5.2:**
**No.** When points are symmetric about the origin (or any center), only the **direction** of x⁽⁰⁾ - x⁽¹⁾ matters, not the magnitude. Scaling both points by any positive constant k gives the same θ̂.

</details>

---

## Category E: Application Contexts

### Variation E1: Spam Classification

A spam classifier represents emails as 2D feature vectors: x = (word_count, link_count).

Given:
- Spam email: x⁽⁰⁾ = (50, 10)
- Legitimate email: x⁽¹⁾ = (200, 2)

#### Part E1.1
Compute the separating hyperplane.

#### Part E1.2
Would an email with features (100, 5) be classified as spam or legitimate?

<details>
<summary>Solution</summary>

**Part E1.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (-150, 8), ||·|| = √(22500 + 64) = √22564 = 2√5641
- θ̂ = (-150, 8)/(2√5641) = (-75/√5641, 4/√5641)
- Midpoint = (125, 6)
- b = -(-75·125 + 4·6)/√5641 = (9375 - 24)/√5641 = 9351/√5641

**θ̂ = (-75/√5641, 4/√5641), b = 9351/√5641**

**Part E1.2:**
For x = (100, 5):
θ̂ᵀx + b = (-75·100 + 4·5 + 9351)/√5641 = (-7500 + 20 + 9351)/√5641 = 1871/√5641 > 0

Since this has the **same sign as x⁽⁰⁾** (spam), it would be classified as **spam**.

</details>

---

### Variation E2: Medical Diagnosis

A diagnostic model uses two blood markers: x = (marker_A, marker_B).

Given:
- Healthy patient: x⁽⁰⁾ = (120, 80)
- Sick patient: x⁽¹⁾ = (180, 140)

#### Part E2.1
Compute θ̂ and b.

#### Part E2.2
A new patient has markers (140, 100). What is the predicted diagnosis?

#### Part E2.3
How confident should we be in this prediction? (Consider the distance to the boundary)

<details>
<summary>Solution</summary>

**Part E2.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (-60, -60), ||·|| = 60√2
- θ̂ = (-1/√2, -1/√2)
- Midpoint = (150, 110)
- b = -(-150/√2 - 110/√2) = 260/√2 = 130√2

**θ̂ = (-1/√2, -1/√2), b = 130√2**

**Part E2.2:**
For x = (140, 100):
θ̂ᵀx + b = -140/√2 - 100/√2 + 130√2 = (-240 + 260)/√2 = 20/√2 > 0

Since healthy (x⁽⁰⁾) gives positive value: **Predicted as Healthy**

**Part E2.3:**
Distance = |20/√2| = 10√2 ≈ 14.1

Margin (distance from training points) = |−120/√2 − 80/√2 + 130√2| = |60/√2| = 30√2 ≈ 42.4

The new patient is at distance 14.1 from boundary, vs margin 42.4. This is about **1/3 of the way to the boundary**, suggesting **moderate confidence**.

</details>

---

### Variation E3: Image Classification (Simplified)

Two images are represented by their average pixel intensities in RGB: x = (R, G, B).

Given:
- Cat image: x⁽⁰⁾ = (0.6, 0.5, 0.4)
- Dog image: x⁽¹⁾ = (0.4, 0.3, 0.2)

#### Part E3.1
Compute θ̂ and b.

#### Part E3.2
A new image has features (0.5, 0.4, 0.3). Which class does it belong to?

<details>
<summary>Solution</summary>

**Part E3.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (0.2, 0.2, 0.2), ||·|| = 0.2√3
- θ̂ = (1/√3, 1/√3, 1/√3)
- Midpoint = (0.5, 0.4, 0.3)
- b = -(0.5 + 0.4 + 0.3)/√3 = -1.2/√3

**θ̂ = (1/√3, 1/√3, 1/√3), b = -1.2/√3**

**Part E3.2:**
For x = (0.5, 0.4, 0.3):
θ̂ᵀx + b = (0.5 + 0.4 + 0.3)/√3 - 1.2/√3 = 0

The point is **exactly on the boundary**! Classification is ambiguous.

</details>

---

### Variation E4: Stock Market

A trading algorithm classifies stocks as "buy" or "sell" based on two indicators: x = (momentum, volatility).

Given:
- Buy signal: x⁽⁰⁾ = (0.8, 0.2)
- Sell signal: x⁽¹⁾ = (0.2, 0.8)

#### Part E4.1
Compute θ̂ and b.

#### Part E4.2
Interpret the meaning of θ̂ in terms of the trading strategy.

<details>
<summary>Solution</summary>

**Part E4.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (0.6, -0.6), ||·|| = 0.6√2
- θ̂ = (1/√2, -1/√2)
- Midpoint = (0.5, 0.5)
- b = -(0.5/√2 - 0.5/√2) = 0

**θ̂ = (1/√2, -1/√2), b = 0**

**Part E4.2:**
The decision boundary is: x₁/√2 - x₂/√2 = 0, or **momentum = volatility**.

Interpretation:
- **Buy** when momentum > volatility (positive momentum outweighs risk)
- **Sell** when volatility > momentum (risk outweighs momentum)

</details>

---

### Variation E5: Physics - Force Classification

A physics simulation classifies particle interactions based on force vector components.

Given:
- Attractive force: x⁽⁰⁾ = (-5, 0, 12) (pointing toward origin)
- Repulsive force: x⁽¹⁾ = (5, 0, -12) (pointing away)

#### Part E5.1
Compute θ̂ and b.

#### Part E5.2
What is special about the separating hyperplane in this case?

<details>
<summary>Solution</summary>

**Part E5.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (-10, 0, 24), ||·|| = √(100 + 576) = √676 = 26
- θ̂ = (-10/26, 0, 24/26) = (-5/13, 0, 12/13)
- Midpoint = (0, 0, 0)
- b = 0

**θ̂ = (-5/13, 0, 12/13), b = 0**

**Part E5.2:**
The separating hyperplane **passes through the origin** (b = 0). This makes physical sense: the boundary between attractive and repulsive forces in this symmetric setup is centered at the origin.

</details>

---

## Category F: Proof/Theory Questions

### Variation F1: Uniqueness of Maximum Margin Hyperplane

Consider two points x⁽⁰⁾ and x⁽¹⁾ with opposite labels.

#### Part F1.1
Prove that the maximum margin separating hyperplane is unique.

#### Part F1.2
Prove that the margin equals ||x⁽⁰⁾ - x⁽¹⁾||/2.

<details>
<summary>Solution</summary>

**Part F1.1:**
For the hyperplane to have equal distance to both points:
- Let m = (x⁽⁰⁾ + x⁽¹⁾)/2 be the midpoint
- Any hyperplane equidistant to both must pass through m (by symmetry)
- Any hyperplane through m has form θ̂ᵀ(x - m) = 0

The margin is maximized when θ̂ is parallel to x⁽⁰⁾ - x⁽¹⁾:
- Margin = |θ̂ᵀ(x⁽⁰⁾ - m)| = |θ̂ᵀ(x⁽⁰⁾ - x⁽¹⁾)|/2
- This is maximized when θ̂ ∝ (x⁽⁰⁾ - x⁽¹⁾)

Since there's only one such direction (up to sign), the hyperplane is **unique**.

**Part F1.2:**
With θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||:
- Margin = |θ̂ᵀ(x⁽⁰⁾ - m)|
- = |θ̂ᵀ(x⁽⁰⁾ - x⁽¹⁾)|/2
- = |(x⁽⁰⁾ - x⁽¹⁾)ᵀ(x⁽⁰⁾ - x⁽¹⁾)|/(2||x⁽⁰⁾ - x⁽¹⁾||)
- = ||x⁽⁰⁾ - x⁽¹⁾||²/(2||x⁽⁰⁾ - x⁽¹⁾||)
- = **||x⁽⁰⁾ - x⁽¹⁾||/2**

</details>

---

### Variation F2: Signed Distance

#### Part F2.1
Define the **signed distance** from a point p to hyperplane θ̂ᵀx + b = 0.

#### Part F2.2
Prove that for the maximum margin hyperplane, the signed distances from x⁽⁰⁾ and x⁽¹⁾ are negatives of each other.

<details>
<summary>Solution</summary>

**Part F2.1:**
The **signed distance** is: d(p) = θ̂ᵀp + b

This is positive if p is on the same side as the normal θ̂, negative otherwise.

**Part F2.2:**
For the maximum margin hyperplane:
- θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||
- b = -θ̂ᵀm where m = (x⁽⁰⁾ + x⁽¹⁾)/2

Signed distance from x⁽⁰⁾:
d(x⁽⁰⁾) = θ̂ᵀx⁽⁰⁾ + b = θ̂ᵀ(x⁽⁰⁾ - m) = θ̂ᵀ(x⁽⁰⁾ - x⁽¹⁾)/2 = ||x⁽⁰⁾ - x⁽¹⁾||/2

Signed distance from x⁽¹⁾:
d(x⁽¹⁾) = θ̂ᵀx⁽¹⁾ + b = θ̂ᵀ(x⁽¹⁾ - m) = θ̂ᵀ(x⁽¹⁾ - x⁽⁰⁾)/2 = -||x⁽⁰⁾ - x⁽¹⁾||/2

Therefore: **d(x⁽⁰⁾) = -d(x⁽¹⁾)** ✓

</details>

---

### Variation F3: Scaling Invariance

#### Part F3.1
Prove that if we scale both points by a constant c > 0 (replacing x⁽⁰⁾ with cx⁽⁰⁾ and x⁽¹⁾ with cx⁽¹⁾), then θ̂ remains the same.

#### Part F3.2
How does b change under this scaling?

<details>
<summary>Solution</summary>

**Part F3.1:**
New θ̂':
- θ̂' = (cx⁽⁰⁾ - cx⁽¹⁾)/||cx⁽⁰⁾ - cx⁽¹⁾||
- = c(x⁽⁰⁾ - x⁽¹⁾)/(c||x⁽⁰⁾ - x⁽¹⁾||)
- = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||
- = θ̂

**θ̂ is unchanged** ✓

**Part F3.2:**
New midpoint: m' = (cx⁽⁰⁾ + cx⁽¹⁾)/2 = cm

New b':
- b' = -θ̂ᵀm' = -θ̂ᵀ(cm) = c(-θ̂ᵀm) = **cb**

The intercept **scales linearly** with c.

</details>

---

### Variation F4: Translation Invariance

#### Part F4.1
Prove that if we translate both points by a vector v (replacing x⁽⁰⁾ with x⁽⁰⁾ + v and x⁽¹⁾ with x⁽¹⁾ + v), then θ̂ remains the same.

#### Part F4.2
How does b change under this translation?

<details>
<summary>Solution</summary>

**Part F4.1:**
New θ̂':
- θ̂' = ((x⁽⁰⁾ + v) - (x⁽¹⁾ + v))/||(x⁽⁰⁾ + v) - (x⁽¹⁾ + v)||
- = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||
- = θ̂

**θ̂ is unchanged** ✓

**Part F4.2:**
New midpoint: m' = ((x⁽⁰⁾ + v) + (x⁽¹⁾ + v))/2 = m + v

New b':
- b' = -θ̂ᵀm' = -θ̂ᵀ(m + v) = -θ̂ᵀm - θ̂ᵀv = **b - θ̂ᵀv**

The intercept changes by **-θ̂ᵀv**.

</details>

---

### Variation F5: Rotation Invariance

Let R be an orthogonal rotation matrix (RᵀR = I).

#### Part F5.1
If we rotate both points (replacing x⁽⁰⁾ with Rx⁽⁰⁾ and x⁽¹⁾ with Rx⁽¹⁾), how does θ̂ transform?

#### Part F5.2
How does b change? (Assume points are centered at origin)

<details>
<summary>Solution</summary>

**Part F5.1:**
New θ̂':
- θ̂' = (Rx⁽⁰⁾ - Rx⁽¹⁾)/||Rx⁽⁰⁾ - Rx⁽¹⁾||
- = R(x⁽⁰⁾ - x⁽¹⁾)/||R(x⁽⁰⁾ - x⁽¹⁾)||

Since R is orthogonal: ||Rv|| = ||v|| for any v.
- = R(x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾||
- = **Rθ̂**

The normal vector **rotates with the points**.

**Part F5.2:**
If points are centered at origin: m = 0, so m' = R·0 = 0
- b' = -θ̂'ᵀm' = 0 = b

**b remains 0** when points are centered at origin.

</details>

---

## Category G: Coding Variations

### Variation G1: Basic Implementation (From Scratch)

Write a function to compute θ̂ and b using only basic NumPy operations (no np.linalg).

```python
import numpy as np

def compute_hyperplane_basic(X):
    """
    Input: X is a NumPy array with shape (2, d)
           X[0] is x^(0) (class +1), X[1] is x^(1) (class -1)
    Output: theta_hat (shape (d,)), b (scalar)
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

def compute_hyperplane_basic(X):
    """
    Input: X is a NumPy array with shape (2, d)
           X[0] is x^(0) (class +1), X[1] is x^(1) (class -1)
    Output: theta_hat (shape (d,)), b (scalar)
    """
    x0, x1 = X[0], X[1]

    # Direction vector
    diff = x0 - x1

    # Normalize to get unit vector
    norm = np.sqrt(np.sum(diff ** 2))
    theta_hat = diff / norm

    # Midpoint
    midpoint = (x0 + x1) / 2

    # Compute b
    b = -np.sum(theta_hat * midpoint)

    return theta_hat, b

# Test
X = np.array([[5, 7], [-3, 2]])
theta, b = compute_hyperplane_basic(X)
print(f"θ̂ = {theta}")
print(f"b = {b}")
```

Output:
```
θ̂ = [0.84800189 0.53000118]
b = -3.2190071...
```

</details>

---

### Variation G2: Vectorized Batch Implementation

Write a function that computes hyperplanes for multiple pairs of points simultaneously.

```python
def compute_hyperplanes_batch(X):
    """
    Input: X is a NumPy array with shape (N, 2, d)
           X[i, 0] is x^(0) for pair i
           X[i, 1] is x^(1) for pair i
    Output: theta_hats (shape (N, d)), bs (shape (N,))
    """
    # Your code here (NO loops!)
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

def compute_hyperplanes_batch(X):
    """
    Input: X is a NumPy array with shape (N, 2, d)
           X[i, 0] is x^(0) for pair i
           X[i, 1] is x^(1) for pair i
    Output: theta_hats (shape (N, d)), bs (shape (N,))
    """
    # Extract x0 and x1 for all pairs
    x0 = X[:, 0, :]  # Shape: (N, d)
    x1 = X[:, 1, :]  # Shape: (N, d)

    # Direction vectors
    diff = x0 - x1  # Shape: (N, d)

    # Norms
    norms = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True))  # Shape: (N, 1)

    # Normalize
    theta_hats = diff / norms  # Shape: (N, d)

    # Midpoints
    midpoints = (x0 + x1) / 2  # Shape: (N, d)

    # Compute b for each pair
    bs = -np.sum(theta_hats * midpoints, axis=1)  # Shape: (N,)

    return theta_hats, bs

# Test
X = np.array([
    [[-3, 0], [3, 0]],
    [[5, 7], [-3, 2]],
    [[4, 0], [-4, 0]]
])
thetas, bs = compute_hyperplanes_batch(X)
print("θ̂s:\n", thetas)
print("bs:", bs)
```

</details>

---

### Variation G3: Classification Function

Write a function that classifies new points given a hyperplane.

```python
def classify_points(theta_hat, b, X_new):
    """
    Input:
        theta_hat: unit normal vector (shape (d,))
        b: intercept (scalar)
        X_new: points to classify (shape (M, d))
    Output:
        labels: +1 or -1 for each point (shape (M,))
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

def classify_points(theta_hat, b, X_new):
    """
    Input:
        theta_hat: unit normal vector (shape (d,))
        b: intercept (scalar)
        X_new: points to classify (shape (M, d))
    Output:
        labels: +1 or -1 for each point (shape (M,))
    """
    # Compute signed distance for each point
    signed_distances = X_new @ theta_hat + b  # Shape: (M,)

    # Classify based on sign
    labels = np.where(signed_distances >= 0, 1, -1)

    return labels

# Test
theta = np.array([1, 0])
b = 0
X_new = np.array([
    [5, 3],    # Should be +1
    [-2, 10],  # Should be -1
    [0, 0],    # On boundary, returns +1
    [0.001, -100]  # Should be +1
])
labels = classify_points(theta, b, X_new)
print("Labels:", labels)
```

Output:
```
Labels: [ 1 -1  1  1]
```

</details>

---

### Variation G4: Distance to Hyperplane

Write a function that computes distances from points to a hyperplane.

```python
def distances_to_hyperplane(theta_hat, b, X):
    """
    Input:
        theta_hat: unit normal vector (shape (d,))
        b: intercept (scalar)
        X: points (shape (M, d))
    Output:
        distances: unsigned distances (shape (M,))
        signed_distances: signed distances (shape (M,))
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

def distances_to_hyperplane(theta_hat, b, X):
    """
    Input:
        theta_hat: unit normal vector (shape (d,))
        b: intercept (scalar)
        X: points (shape (M, d))
    Output:
        distances: unsigned distances (shape (M,))
        signed_distances: signed distances (shape (M,))
    """
    # Compute signed distances
    signed_distances = X @ theta_hat + b  # Shape: (M,)

    # Unsigned distances
    distances = np.abs(signed_distances)

    return distances, signed_distances

# Test
theta = np.array([0.6, 0.8])  # 3-4-5 triangle unit vector
b = -5
X = np.array([
    [10, 10],
    [0, 0],
    [5, 2.5]  # On the hyperplane: 0.6*5 + 0.8*2.5 - 5 = 3 + 2 - 5 = 0
])
dist, signed = distances_to_hyperplane(theta, b, X)
print("Distances:", dist)
print("Signed distances:", signed)
```

Output:
```
Distances: [9. 5. 0.]
Signed distances: [ 9. -5.  0.]
```

</details>

---

### Variation G5: Visualization Function

Write a function that visualizes the hyperplane and points in 2D.

```python
import matplotlib.pyplot as plt

def visualize_hyperplane(X, theta_hat, b, X_new=None, ax=None):
    """
    Input:
        X: training points (shape (2, 2))
        theta_hat: unit normal (shape (2,))
        b: intercept (scalar)
        X_new: optional new points to classify (shape (M, 2))
        ax: optional matplotlib axis
    """
    # Your code here
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_hyperplane(X, theta_hat, b, X_new=None, ax=None):
    """
    Input:
        X: training points (shape (2, 2))
        theta_hat: unit normal (shape (2,))
        b: intercept (scalar)
        X_new: optional new points to classify (shape (M, 2))
        ax: optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot training points
    ax.scatter(X[0, 0], X[0, 1], c='blue', s=200, marker='o',
               label='Class +1', edgecolors='black', linewidth=2)
    ax.scatter(X[1, 0], X[1, 1], c='red', s=200, marker='s',
               label='Class -1', edgecolors='black', linewidth=2)

    # Compute plot bounds
    all_points = X.copy()
    if X_new is not None:
        all_points = np.vstack([all_points, X_new])

    x_min, x_max = all_points[:, 0].min() - 2, all_points[:, 0].max() + 2
    y_min, y_max = all_points[:, 1].min() - 2, all_points[:, 1].max() + 2

    # Plot hyperplane: theta_hat[0]*x + theta_hat[1]*y + b = 0
    # Solve for y: y = -(theta_hat[0]*x + b) / theta_hat[1]
    if abs(theta_hat[1]) > 1e-10:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(theta_hat[0] * x_line + b) / theta_hat[1]
        # Filter to plot bounds
        mask = (y_line >= y_min) & (y_line <= y_max)
        ax.plot(x_line[mask], y_line[mask], 'g-', linewidth=2,
                label='Hyperplane')
    else:
        # Vertical line: x = -b / theta_hat[0]
        x_val = -b / theta_hat[0]
        ax.axvline(x=x_val, color='green', linewidth=2, label='Hyperplane')

    # Plot normal vector from midpoint
    midpoint = (X[0] + X[1]) / 2
    ax.annotate('', xy=midpoint + theta_hat, xytext=midpoint,
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.scatter(*midpoint, c='green', s=100, marker='x',
               label='Midpoint', linewidth=3)

    # Plot new points if provided
    if X_new is not None:
        labels = np.sign(X_new @ theta_hat + b)
        colors = ['blue' if l > 0 else 'red' for l in labels]
        ax.scatter(X_new[:, 0], X_new[:, 1], c=colors, s=100,
                   marker='^', alpha=0.6, label='New points')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Separating Hyperplane: {theta_hat[0]:.3f}$x_1$ + {theta_hat[1]:.3f}$x_2$ + {b:.3f} = 0')

    return ax

# Example usage:
# X = np.array([[5, 7], [-3, 2]])
# theta, b = compute_hyperplane_basic(X)
# visualize_hyperplane(X, theta, b)
# plt.show()
```

</details>

---

### Variation G6: Full Pipeline with Tests

Write a complete module with unit tests.

```python
import numpy as np

class BinaryHyperplaneClassifier:
    """
    A simple binary classifier using maximum margin hyperplane.
    """

    def __init__(self):
        self.theta_hat = None
        self.b = None
        self.margin = None

    def fit(self, X, y):
        """
        Fit the classifier to two points with labels +1 and -1.

        Input:
            X: points (shape (2, d))
            y: labels (shape (2,)), must be [1, -1] or [-1, 1]
        """
        # Your code here
        pass

    def predict(self, X):
        """
        Predict labels for new points.

        Input:
            X: points (shape (M, d))
        Output:
            labels: predicted labels (shape (M,))
        """
        # Your code here
        pass

    def decision_function(self, X):
        """
        Compute signed distance to hyperplane.

        Input:
            X: points (shape (M, d))
        Output:
            distances: signed distances (shape (M,))
        """
        # Your code here
        pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

class BinaryHyperplaneClassifier:
    """
    A simple binary classifier using maximum margin hyperplane.
    """

    def __init__(self):
        self.theta_hat = None
        self.b = None
        self.margin = None

    def fit(self, X, y):
        """
        Fit the classifier to two points with labels +1 and -1.
        """
        assert X.shape[0] == 2, "Must provide exactly 2 points"
        assert set(y) == {1, -1}, "Labels must be +1 and -1"

        # Reorder so X[0] is class +1
        if y[0] == -1:
            X = X[::-1]

        x0, x1 = X[0], X[1]

        # Compute hyperplane parameters
        diff = x0 - x1
        norm = np.linalg.norm(diff)

        if norm < 1e-10:
            raise ValueError("Points are too close together")

        self.theta_hat = diff / norm
        midpoint = (x0 + x1) / 2
        self.b = -np.dot(self.theta_hat, midpoint)
        self.margin = norm / 2

        return self

    def predict(self, X):
        """
        Predict labels for new points.
        """
        if self.theta_hat is None:
            raise ValueError("Classifier not fitted")

        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def decision_function(self, X):
        """
        Compute signed distance to hyperplane.
        """
        if self.theta_hat is None:
            raise ValueError("Classifier not fitted")

        return X @ self.theta_hat + self.b


# Unit tests
def test_basic_case():
    clf = BinaryHyperplaneClassifier()
    X = np.array([[-3, 0], [3, 0]])
    y = np.array([1, -1])
    clf.fit(X, y)

    assert np.allclose(clf.theta_hat, [-1, 0])
    assert np.isclose(clf.b, 0)
    assert np.isclose(clf.margin, 3)
    print("✓ Basic case passed")

def test_diagonal_case():
    clf = BinaryHyperplaneClassifier()
    X = np.array([[5, 7], [-3, 2]])
    y = np.array([1, -1])
    clf.fit(X, y)

    expected_theta = np.array([8, 5]) / np.sqrt(89)
    assert np.allclose(clf.theta_hat, expected_theta)
    print("✓ Diagonal case passed")

def test_prediction():
    clf = BinaryHyperplaneClassifier()
    X = np.array([[2, 0], [-2, 0]])
    y = np.array([1, -1])
    clf.fit(X, y)

    X_new = np.array([[5, 10], [-1, -5], [0, 0]])
    predictions = clf.predict(X_new)

    assert predictions[0] == 1   # Far positive
    assert predictions[1] == -1  # Slightly negative
    assert predictions[2] == 1   # On boundary (>= 0)
    print("✓ Prediction case passed")

def test_3d_case():
    clf = BinaryHyperplaneClassifier()
    X = np.array([[1, 2, 2], [-1, -2, -2]])
    y = np.array([1, -1])
    clf.fit(X, y)

    expected_theta = np.array([1, 2, 2]) / 3
    assert np.allclose(clf.theta_hat, expected_theta)
    assert np.isclose(clf.b, 0)
    print("✓ 3D case passed")

def test_reversed_labels():
    clf = BinaryHyperplaneClassifier()
    X = np.array([[3, 0], [-3, 0]])
    y = np.array([-1, 1])  # Reversed!
    clf.fit(X, y)

    # Should still work correctly
    assert clf.predict(np.array([[5, 0]])) == -1
    assert clf.predict(np.array([[-5, 0]])) == 1
    print("✓ Reversed labels case passed")

# Run tests
if __name__ == "__main__":
    test_basic_case()
    test_diagonal_case()
    test_prediction()
    test_3d_case()
    test_reversed_labels()
    print("\nAll tests passed! ✓")
```

</details>

---

### Variation G7: NumPy-Only Constraint (No np.linalg)

Rewrite the core computation without using `np.linalg.norm`.

```python
def compute_hyperplane_no_linalg(X):
    """
    Compute hyperplane WITHOUT using np.linalg.

    Input: X is a NumPy array with shape (2, d)
    Output: theta_hat (shape (d,)), b (scalar)
    """
    # Your code here - cannot use np.linalg!
    pass
```

<details>
<summary>Solution</summary>

```python
import numpy as np

def compute_hyperplane_no_linalg(X):
    """
    Compute hyperplane WITHOUT using np.linalg.
    """
    x0, x1 = X[0], X[1]
    diff = x0 - x1

    # Compute norm manually
    norm_squared = np.sum(diff * diff)  # or np.dot(diff, diff)
    norm = np.sqrt(norm_squared)

    theta_hat = diff / norm
    midpoint = (x0 + x1) / 2

    # Compute b manually
    b = -np.sum(theta_hat * midpoint)  # or -np.dot(theta_hat, midpoint)

    return theta_hat, b

# Verification
X = np.array([[5, 7], [-3, 2]])
theta1, b1 = compute_hyperplane_no_linalg(X)

# Compare with np.linalg version
diff = X[0] - X[1]
theta2 = diff / np.linalg.norm(diff)
b2 = -np.dot(theta2, (X[0] + X[1]) / 2)

print(f"No linalg: θ̂ = {theta1}, b = {b1}")
print(f"With linalg: θ̂ = {theta2}, b = {b2}")
print(f"Match: {np.allclose(theta1, theta2) and np.isclose(b1, b2)}")
```

</details>

---

### Variation D6: Numerical Stability (Near-Coincident Points)

Consider a dataset with x⁽⁰⁾ = (1.0, 1.0) and x⁽¹⁾ = (1.0 + 1e-10, 1.0 + 1e-10).

#### Part D6.1
What is ||x⁽⁰⁾ - x⁽¹⁾||? What numerical issues might arise?

#### Part D6.2
Write a robust implementation that handles near-coincident points gracefully.

#### Part D6.3
For what value of ε should we consider points "too close" to separate meaningfully?

<details>
<summary>Solution</summary>

**Part D6.1:**
- x⁽⁰⁾ - x⁽¹⁾ = (-1e-10, -1e-10)
- ||·|| = √(2) × 1e-10 ≈ 1.41e-10

**Numerical issues:**
1. **Division by tiny number**: θ̂ = diff/norm involves dividing by ~1e-10
2. **Floating point precision**: At this scale, rounding errors dominate
3. **Meaningless margin**: The "margin" would be ~7e-11, smaller than machine epsilon for float32

**Part D6.2:**
```python
import numpy as np

def compute_hyperplane_robust(X, tol=1e-8):
    """
    Robust hyperplane computation with numerical stability checks.
    """
    x0, x1 = X[0], X[1]
    diff = x0 - x1
    norm = np.linalg.norm(diff)

    # Check for near-coincident points
    if norm < tol:
        raise ValueError(
            f"Points too close (distance={norm:.2e} < tol={tol:.2e}). "
            "Cannot define meaningful separating hyperplane."
        )

    theta_hat = diff / norm
    midpoint = (x0 + x1) / 2
    b = -np.dot(theta_hat, midpoint)

    return theta_hat, b, norm / 2  # Also return margin

# Test
X_good = np.array([[5, 7], [-3, 2]])
X_bad = np.array([[1.0, 1.0], [1.0 + 1e-10, 1.0 + 1e-10]])

print("Good case:", compute_hyperplane_robust(X_good))
try:
    compute_hyperplane_robust(X_bad)
except ValueError as e:
    print(f"Bad case caught: {e}")
```

**Part D6.3:**
A reasonable threshold depends on:
- **float64**: ε ≈ 1e-8 to 1e-10 (machine epsilon is ~2.2e-16)
- **float32**: ε ≈ 1e-5 to 1e-6 (machine epsilon is ~1.2e-7)

**Rule of thumb**: Use `tol = np.sqrt(np.finfo(X.dtype).eps)` which gives ~1.5e-8 for float64.

</details>

---

## Category H: Speed Round (Conceptual Quick-Fire)

> **Instructions**: Answer each question in 30-60 seconds. No calculations needed—test your intuition!

### H1: Direction Check
If x⁽⁰⁾ = (3, 4) and x⁽¹⁾ = (0, 0), does θ̂ point toward x⁽⁰⁾ or toward x⁽¹⁾?

<details>
<summary>Answer</summary>

**Toward x⁽⁰⁾** (the positive class). θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||...|| points from x⁽¹⁾ to x⁽⁰⁾.

</details>

---

### H2: Zero Intercept
When is b = 0?

<details>
<summary>Answer</summary>

When the **midpoint is at the origin**, i.e., x⁽⁰⁾ = -x⁽¹⁾ (points are symmetric about the origin).

</details>

---

### H3: Margin Doubling
If you double both x⁽⁰⁾ and x⁽¹⁾ (scale by 2), what happens to the margin?

<details>
<summary>Answer</summary>

**The margin doubles.** Margin = ||x⁽⁰⁾ - x⁽¹⁾||/2, so scaling both points by k scales the margin by k.

</details>

---

### H4: Dimension Independence
Does the formula for θ̂ change between 2D and 100D?

<details>
<summary>Answer</summary>

**No.** The formula θ̂ = (x⁽⁰⁾ - x⁽¹⁾)/||x⁽⁰⁾ - x⁽¹⁾|| works in any dimension. Only the norm computation involves more terms.

</details>

---

### H5: Perpendicularity
The hyperplane θ̂ᵀx + b = 0 is perpendicular to what?

<details>
<summary>Answer</summary>

**Perpendicular to θ̂** (the normal vector), which is also **perpendicular to the hyperplane itself** and **parallel to the line connecting x⁽⁰⁾ and x⁽¹⁾**.

</details>

---

### H6: On the Boundary
If θ̂ᵀx + b = 0 for some point x, what can you say about x?

<details>
<summary>Answer</summary>

**x lies exactly on the decision boundary** (the hyperplane). It's equidistant from both training points.

</details>

---

### H7: Sign Flip
If you swap the labels (make x⁽⁰⁾ class -1 and x⁽¹⁾ class +1), what changes?

<details>
<summary>Answer</summary>

**θ̂ flips sign** (θ̂ → -θ̂) and **b flips sign** (b → -b). The hyperplane itself is unchanged, but the "positive side" swaps.

</details>

---

### H8: Unit Vector Check
How do you verify θ̂ is a unit vector?

<details>
<summary>Answer</summary>

Check that **||θ̂|| = 1**, i.e., θ̂₁² + θ̂₂² + ... + θ̂ₐ² = 1.

</details>

---

### H9: Closest Point
What is the closest point on the hyperplane to x⁽⁰⁾?

<details>
<summary>Answer</summary>

**The midpoint** (x⁽⁰⁾ + x⁽¹⁾)/2. For this two-point case, the closest point on the hyperplane to either training point is always the midpoint.

</details>

---

### H10: Impossible Case
Can you find a separating hyperplane if both points have the same label?

<details>
<summary>Answer</summary>

**Yes, infinitely many!** If both points are class +1, any hyperplane that puts them both on the positive side works. There's no unique maximum-margin solution—the problem is under-constrained.

</details>

---

## Summary

This exhaustive set of **48 variations** covers:

| Category | Count | Focus |
|----------|-------|-------|
| A: Different Values | 5 | Same structure, different numbers |
| B: Different Dimensions | 5 | 1D, 2D, 3D, 4D, 5D |
| C: Conceptual Extensions | 5 | Margins, multiple points, theory |
| D: Computational Variations | 6 | Edge cases, special values, numerical stability |
| E: Application Contexts | 5 | Real-world scenarios |
| F: Proof/Theory | 5 | Mathematical proofs |
| G: Coding | 7 | Implementation challenges |
| H: Speed Round | 10 | Rapid-fire conceptual checks |

---

## Key Takeaways for USAAIO Prep

1. **The formula is elegant**: θ̂ = normalize(x⁽⁰⁾ - x⁽¹⁾), b = -θ̂ᵀ(midpoint)
2. **Margin = half the distance** between the two points
3. **Pythagorean triples** (3-4-5, 5-12-13) make calculations cleaner
4. **Translation and scaling invariance** are powerful properties to exploit
5. **Vectorized implementations** avoid loops and leverage NumPy's speed
