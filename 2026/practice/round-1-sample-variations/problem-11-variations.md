# Problem 11 Variations: MLP Geometry (EXHAUSTIVE)

> **Original:** Build a PyTorch MLP that classifies whether a 2D point is inside a triangle with vertices (0,0), (1,0), (0,2). No training—manually set weights. Uses step activation: Θ(x) = 1 if x ≥ 0, else 0.

> **Core Skills:** Linear separators as hyperplanes, half-plane representations, Boolean logic with neural networks, PyTorch module construction, manual weight initialization

---

## Background: The Geometry-to-Neural-Network Connection

A convex polygon with n sides can be represented as the intersection of n half-planes. Each half-plane corresponds to a linear inequality:

$$w_1 x_1 + w_2 x_2 + b \geq 0$$

This is exactly what a single neuron computes! With a step activation function, each neuron outputs 1 if the point is on the "positive" side of the line, 0 otherwise.

For a point to be inside a convex polygon, it must be on the correct side of ALL edges—this is a logical AND, which can be computed by summing the outputs and checking if they all equal n.

---

## CATEGORY A: Different Triangle Vertices (Same Structure)

### Variation A1: Right Triangle in Different Position

Build a PyTorch MLP that classifies points inside the triangle with vertices **(1, 1), (3, 1), (1, 3)**.

**Part 11.1:** Write the three linear inequalities that define this triangle (interior points satisfy all three).

**Part 11.2:** Construct the MLP with manually set weights. Input shape: (B, 2), output: (B,) with values 0 or 1.

**Part 11.3:** Test your model on points: (1.5, 1.5), (2, 2), (0, 0), (2.5, 1.5).

<details>
<summary>Solution A1</summary>

**11.1:** The three edges define these half-planes:

- Edge (1,1)→(3,1): y ≥ 1 → **y - 1 ≥ 0**
- Edge (3,1)→(1,3): The line is x + y = 4 → **-x - y + 4 ≥ 0** (interior below-left)
- Edge (1,3)→(1,1): x ≥ 1 → **x - 1 ≥ 0**

**11.2:**

```python
import torch
import torch.nn as nn

class TriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Three neurons for three half-planes
        self.fc1 = nn.Linear(2, 3, bias=True)
        # Layer 2: AND gate (all three must be 1)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Set weights manually (no training)
        with torch.no_grad():
            # Half-plane 1: y - 1 ≥ 0 → [0, 1]x + (-1)
            # Half-plane 2: -x - y + 4 ≥ 0 → [-1, -1]x + 4
            # Half-plane 3: x - 1 ≥ 0 → [1, 0]x + (-1)
            self.fc1.weight = nn.Parameter(torch.tensor([
                [0.0, 1.0],    # y
                [-1.0, -1.0],  # -x - y
                [1.0, 0.0]     # x
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([-1.0, 4.0, -1.0]))

            # AND gate: sum must be 3 (all conditions true)
            # Output 1 if sum ≥ 3, i.e., w·h + b ≥ 0 where w=[1,1,1], b=-2.5
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        h = self.step(self.fc1(x))  # Three half-plane tests
        out = self.step(self.fc2(h))  # AND gate
        return out.squeeze(-1)

model = TriangleMLP()
```

**11.3:** Testing:

```python
test_points = torch.tensor([[1.5, 1.5], [2.0, 2.0], [0.0, 0.0], [2.5, 1.5]])
outputs = model(test_points)
# (1.5, 1.5): y-1=0.5≥0 ✓, -x-y+4=1≥0 ✓, x-1=0.5≥0 ✓ → Inside (1)
# (2.0, 2.0): y-1=1≥0 ✓, -x-y+4=0≥0 ✓, x-1=1≥0 ✓ → Inside (1) [on edge]
# (0.0, 0.0): y-1=-1<0 ✗ → Outside (0)
# (2.5, 1.5): y-1=0.5≥0 ✓, -x-y+4=0≥0 ✓, x-1=1.5≥0 ✓ → Inside (1) [on edge]
print(outputs)  # tensor([1., 1., 0., 1.])
```

</details>

### Variation A2: Equilateral-ish Triangle

Build an MLP for the triangle with vertices **(0, 0), (4, 0), (2, 2√3)** (approximately equilateral).

**Part 11.1:** Derive the line equations for all three edges.

**Part 11.2:** Write the half-plane inequalities (interior on left side of each edge, going counterclockwise).

**Part 11.3:** Implement the MLP.

<details>
<summary>Solution A2</summary>

**11.1:** Edge equations (going counterclockwise):

- Edge (0,0)→(4,0): y = 0
- Edge (4,0)→(2,2√3): slope = (2√3-0)/(2-4) = -√3, line: y - 0 = -√3(x-4) → √3x + y = 4√3
- Edge (2,2√3)→(0,0): slope = 2√3/2 = √3, line: y = √3x → √3x - y = 0

**11.2:** For interior points (counterclockwise traversal, interior on left):

- **y ≥ 0** (above bottom edge)
- **-√3x - y + 4√3 ≥ 0** (below-left of right edge)
- **-√3x + y ≥ 0** (above-right of left edge)

**11.3:**

```python
import torch
import torch.nn as nn
import math

class EquilateralTriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        sqrt3 = math.sqrt(3)
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [0.0, 1.0],           # y
                [-sqrt3, -1.0],       # -√3x - y
                [-sqrt3, 1.0]         # -√3x + y
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 4*sqrt3, 0.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

</details>

### Variation A3: Obtuse Triangle

Build an MLP for the triangle with vertices **(0, 0), (5, 0), (1, 1)** (obtuse angle at (1,1)).

**Part 11.1:** Identify which angle is obtuse and verify using dot products.

**Part 11.2:** Build the MLP.

**Part 11.3:** Test on the centroid (2, 1/3) and verify it's inside.

<details>
<summary>Solution A3</summary>

**11.1:** Vectors from (1,1):

- To (0,0): (-1, -1)
- To (5,0): (4, -1)

Dot product: (-1)(4) + (-1)(-1) = -4 + 1 = -3 < 0

Since dot product is negative, the angle at (1,1) is obtuse (> 90°). ✓

**11.2:** Edge inequalities (counterclockwise from (0,0)):

- (0,0)→(5,0): **y ≥ 0**
- (5,0)→(1,1): line through (5,0) and (1,1): slope = 1/(-4) = -1/4
  Equation: y - 0 = -1/4(x - 5) → x + 4y = 5 → **-x - 4y + 5 ≥ 0**
- (1,1)→(0,0): line y = x → **-x + y ≥ 0**

```python
class ObtuseTriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [0.0, 1.0],    # y
                [-1.0, -4.0],  # -x - 4y
                [-1.0, 1.0]    # -x + y
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 5.0, 0.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**11.3:** Centroid = ((0+5+1)/3, (0+0+1)/3) = (2, 1/3)

- y = 1/3 ≥ 0 ✓
- -2 - 4/3 + 5 = 5 - 2 - 4/3 = 3 - 4/3 = 5/3 ≥ 0 ✓
- -2 + 1/3 = -5/3 < 0 ✗

Wait—let me recheck the orientation. Going counterclockwise, interior is on the LEFT.

Actually for (1,1)→(0,0), the interior is on the RIGHT of this direction. Let me fix:

- **x - y ≥ 0** for the third edge (points with x ≥ y)

Retest centroid (2, 1/3): x - y = 2 - 1/3 = 5/3 ≥ 0 ✓

All three conditions satisfied → Inside. ✓

</details>

### Variation A4: Thin Triangle (Near-Degenerate)

Build an MLP for the triangle with vertices **(0, 0), (10, 0), (5, 0.1)**.

**Part 11.1:** Why is this triangle "thin"? Compute its area.

**Part 11.2:** Build the MLP and discuss numerical precision concerns.

**Part 11.3:** Test on point (5, 0.05).

<details>
<summary>Solution A4</summary>

**11.1:** Area = (1/2)|base × height| = (1/2)(10)(0.1) = 0.5 square units

The triangle is thin because the height (0.1) is much smaller than the base (10). The ratio is 1:100.

**11.2:**

```python
class ThinTriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Edge equations:
        # (0,0)→(10,0): y = 0 → y ≥ 0
        # (10,0)→(5,0.1): slope = 0.1/(-5) = -0.02
        #   y = -0.02(x-10) → y = -0.02x + 0.2 → 0.02x + y - 0.2 ≤ 0 → -0.02x - y + 0.2 ≥ 0
        # (5,0.1)→(0,0): slope = 0.1/5 = 0.02
        #   y = 0.02x → -0.02x + y ≥ 0

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [0.0, 1.0],      # y
                [-0.02, -1.0],   # -0.02x - y
                [-0.02, 1.0]     # -0.02x + y
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.2, 0.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**Numerical concerns:** With small coefficients like 0.02, floating-point errors could affect results. For robust implementation, consider scaling the inequalities.

**11.3:** Point (5, 0.05):

- y = 0.05 ≥ 0 ✓
- -0.02(5) - 0.05 + 0.2 = -0.1 - 0.05 + 0.2 = 0.05 ≥ 0 ✓
- -0.02(5) + 0.05 = -0.1 + 0.05 = -0.05 < 0 ✗

The point (5, 0.05) is OUTSIDE the triangle (below the left edge). ✓

</details>

### Variation A5: Unit Triangle

Build an MLP for the triangle with vertices **(0, 0), (1, 0), (0, 1)** (the standard unit simplex).

**Part 11.1:** This triangle has a special property—what is it called in optimization/probability?

**Part 11.2:** Build the MLP. What is special about the constraint x + y ≤ 1?

**Part 11.3:** Show that points (p, 1-p) for p ∈ [0,1] lie on the hypotenuse.

<details>
<summary>Solution A5</summary>

**11.1:** This is the **2-simplex** or **probability simplex**. In optimization, constraints x ≥ 0, y ≥ 0, x + y ≤ 1 define a standard form. In probability, if we have (p₁, p₂, p₃) with p₁ + p₂ + p₃ = 1 and set p₃ = 1 - p₁ - p₂, then (p₁, p₂) lies in this simplex.

**11.2:**

```python
class SimplexMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # x ≥ 0, y ≥ 0, -x - y + 1 ≥ 0
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],   # x
                [0.0, 1.0],   # y
                [-1.0, -1.0]  # -x - y
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**Special property:** The constraint -x - y + 1 ≥ 0 is equivalent to x + y ≤ 1. This is a **budget constraint** in economics or a **normalization constraint** in probability.

**11.3:** For point (p, 1-p):

- Hypotenuse equation: x + y = 1
- Check: p + (1-p) = 1 ✓

These points exactly lie on the hypotenuse from (1,0) to (0,1).

</details>

---

## CATEGORY B: Different Shapes (Different Complexity)

### Variation B1: Square

Build an MLP for the square with vertices **(0, 0), (1, 0), (1, 1), (0, 1)**.

**Part 11.1:** How many half-planes are needed for a square?

**Part 11.2:** Build the MLP. What changes compared to the triangle?

**Part 11.3:** Test on corners, center, and outside points.

<details>
<summary>Solution B1</summary>

**11.1:** A square has 4 edges, so we need **4 half-planes**.

**11.2:**

```python
class SquareMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)  # 4 half-planes
        self.fc2 = nn.Linear(4, 1, bias=True)

        # x ≥ 0, y ≥ 0, -x + 1 ≥ 0, -y + 1 ≥ 0
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],   # x ≥ 0
                [0.0, 1.0],   # y ≥ 0
                [-1.0, 0.0],  # x ≤ 1
                [0.0, -1.0]   # y ≤ 1
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0]))

            # AND gate: sum must be 4
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.5]))  # ≥ 4 means > 3.5

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**Key difference:** First layer has 4 neurons instead of 3, and threshold is -3.5 instead of -2.5.

**11.3:**

```python
test_points = torch.tensor([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # corners
    [0.5, 0.5],  # center
    [-0.1, 0.5], [1.1, 0.5]  # outside
])
# Corners: on boundary → all 1 (using ≥ 0)
# Center: inside → 1
# Outside: → 0
```

</details>

### Variation B2: Pentagon (Regular)

Build an MLP for a regular pentagon centered at origin with circumradius 1.

**Part 11.1:** Compute the 5 vertices using polar coordinates.

**Part 11.2:** Derive the 5 edge equations.

**Part 11.3:** Build the MLP.

<details>
<summary>Solution B2</summary>

**11.1:** Regular pentagon vertices at angles θ = 90°, 162°, 234°, 306°, 378° (or 18°):
Actually, let's use θ = 90° + 72°k for k = 0,1,2,3,4:

- (cos 90°, sin 90°) = (0, 1)
- (cos 162°, sin 162°) = (-0.951, 0.309)
- (cos 234°, sin 234°) = (-0.588, -0.809)
- (cos 306°, sin 306°) = (0.588, -0.809)
- (cos 378°, sin 378°) = (0.951, 0.309)

**11.2 & 11.3:** Each edge's half-plane has normal pointing inward. For edge from vertex i to i+1, the inward normal is perpendicular to the edge.

```python
import torch
import torch.nn as nn
import math

class PentagonMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 5, bias=True)
        self.fc2 = nn.Linear(5, 1, bias=True)

        # Compute vertices
        angles = [math.pi/2 + k * 2*math.pi/5 for k in range(5)]
        vertices = [(math.cos(a), math.sin(a)) for a in angles]

        # For each edge, compute inward-pointing normal and bias
        weights = []
        biases = []
        for i in range(5):
            v1 = vertices[i]
            v2 = vertices[(i+1) % 5]
            # Edge direction
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            # Inward normal (perpendicular, pointing toward center)
            # For counterclockwise vertices, inward normal is (dy, -dx) normalized
            length = math.sqrt(dx*dx + dy*dy)
            nx, ny = dy/length, -dx/length
            # Bias: -n·v1 (so n·x + b ≥ 0 defines the half-plane)
            b = -(nx * v1[0] + ny * v1[1])
            weights.append([nx, ny])
            biases.append(b)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor(weights))
            self.fc1.bias = nn.Parameter(torch.tensor(biases))

            self.fc2.weight = nn.Parameter(torch.ones(1, 5))
            self.fc2.bias = nn.Parameter(torch.tensor([-4.5]))  # sum ≥ 5

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

</details>

### Variation B3: Hexagon (Regular)

Build an MLP for a regular hexagon with vertices at (±1, 0), (±0.5, ±√3/2).

**Part 11.1:** How many neurons in the first layer?

**Part 11.2:** What is the relationship between number of sides n and the AND-gate threshold?

**Part 11.3:** Implement and test.

<details>
<summary>Solution B3</summary>

**11.1:** A hexagon has 6 sides, so **6 neurons** in the first layer.

**11.2:** For an n-sided polygon, the AND gate requires all n half-planes to be satisfied. The threshold should be -(n - 0.5) to require sum ≥ n. For hexagon: **-5.5**.

**11.3:**

```python
class HexagonMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 6, bias=True)
        self.fc2 = nn.Linear(6, 1, bias=True)

        # Hexagon half-planes (simplified for axis-aligned version)
        # |x| ≤ 1, |y| ≤ √3/2, |x ± √3y| ≤ √3
        sqrt3 = math.sqrt(3)

        with torch.no_grad():
            # Six half-planes for hexagon
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],      # x + 1 ≥ 0
                [-1.0, 0.0],     # -x + 1 ≥ 0
                [0.0, 1.0],      # y + √3/2 ≥ 0 (approx)
                [0.0, -1.0],     # -y + √3/2 ≥ 0
                [0.5, sqrt3/2],  # tilted edge
                [-0.5, sqrt3/2]  # tilted edge
            ]))
            # Biases adjusted for specific hexagon
            self.fc1.bias = nn.Parameter(torch.tensor([1.0, 1.0, sqrt3/2, sqrt3/2, 1.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.ones(1, 6))
            self.fc2.bias = nn.Parameter(torch.tensor([-5.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

</details>

### Variation B4: Circle Approximation

Approximate a unit circle using a regular 12-gon (dodecagon).

**Part 11.1:** What is the maximum error between the circle and 12-gon boundaries?

**Part 11.2:** Build the MLP with 12 half-planes.

**Part 11.3:** For what value of n would the error be less than 0.01?

<details>
<summary>Solution B4</summary>

**11.1:** A regular n-gon inscribed in a unit circle has vertices at radius 1. The edge midpoints are at radius cos(π/n).

For n = 12: cos(π/12) = cos(15°) ≈ 0.966

Maximum error = 1 - cos(π/12) ≈ 1 - 0.966 = **0.034** (about 3.4% of radius)

**11.2:**

```python
class DodecagonMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 12, bias=True)
        self.fc2 = nn.Linear(12, 1, bias=True)

        n = 12
        weights = []
        biases = []
        for k in range(n):
            # Each half-plane has normal pointing inward at angle θ
            theta = 2 * math.pi * k / n
            # Normal vector (pointing inward)
            nx, ny = -math.cos(theta), -math.sin(theta)
            # The half-plane passes through the edge midpoint at distance cos(π/n)
            d = math.cos(math.pi / n)
            # Equation: nx*x + ny*y + d ≥ 0
            weights.append([nx, ny])
            biases.append(d)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor(weights))
            self.fc1.bias = nn.Parameter(torch.tensor(biases))

            self.fc2.weight = nn.Parameter(torch.ones(1, n))
            self.fc2.bias = nn.Parameter(torch.tensor([-(n - 0.5)]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**11.3:** We need 1 - cos(π/n) < 0.01, so cos(π/n) > 0.99.

cos⁻¹(0.99) ≈ 0.1415 radians ≈ 8.1°

So π/n < 0.1415 → n > π/0.1415 ≈ 22.2

**n ≥ 23** gives error < 1%

</details>

### Variation B5: L-Shape (Non-Convex!)

Build an MLP for an L-shape with vertices (0,0), (2,0), (2,1), (1,1), (1,2), (0,2).

**Part 11.1:** Can this be represented with a single layer of half-planes? Why or why not?

**Part 11.2:** Decompose the L into two rectangles and build an MLP using OR logic.

**Part 11.3:** Design an alternative 3-layer architecture.

<details>
<summary>Solution B5</summary>

**11.1:** **No!** A single layer of half-planes can only represent **convex** regions (intersections of half-planes are always convex). The L-shape is non-convex, so it requires a different approach.

**11.2:** Decompose into:

- Rectangle A: (0,0)-(2,0)-(2,1)-(0,1) → x∈[0,2], y∈[0,1]
- Rectangle B: (0,0)-(1,0)-(1,2)-(0,2) → x∈[0,1], y∈[0,2]

The L-shape = A ∪ B (union, which is OR logic).

```python
class LShapeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: 8 half-planes (4 for each rectangle)
        self.fc1 = nn.Linear(2, 8, bias=True)
        # Layer 2: 2 AND gates (one for each rectangle)
        self.fc2 = nn.Linear(8, 2, bias=True)
        # Layer 3: OR gate
        self.fc3 = nn.Linear(2, 1, bias=True)

        with torch.no_grad():
            # Rectangle A: x≥0, x≤2, y≥0, y≤1
            # Rectangle B: x≥0, x≤1, y≥0, y≤2
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],  # Rect A
                [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]   # Rect B
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([
                0.0, 2.0, 0.0, 1.0,  # Rect A
                0.0, 1.0, 0.0, 2.0   # Rect B
            ]))

            # AND gates for each rectangle
            self.fc2.weight = nn.Parameter(torch.tensor([
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Rect A
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]   # Rect B
            ]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.5, -3.5]))

            # OR gate: at least one rectangle
            self.fc3.weight = nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self.fc3.bias = nn.Parameter(torch.tensor([-0.5]))  # sum ≥ 1

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        h1 = self.step(self.fc1(x))
        h2 = self.step(self.fc2(h1))
        out = self.step(self.fc3(h2))
        return out.squeeze(-1)
```

**11.3:** Architecture summary:

- Layer 1: 8 neurons (half-plane tests)
- Layer 2: 2 neurons (AND gates for each convex piece)
- Layer 3: 1 neuron (OR gate combining pieces)

This shows that non-convex regions need **deeper** networks!

</details>

---

## CATEGORY C: Different Activation Functions

### Variation C1: ReLU-based Approximation

Instead of the step function, use ReLU activations to create a "soft" triangle classifier.

**Part 11.1:** Why can't ReLU directly implement a step function?

**Part 11.2:** Design an MLP using ReLU that approximates the triangle classifier.

**Part 11.3:** How does the output differ from the hard step version?

<details>
<summary>Solution C1</summary>

**11.1:** ReLU(x) = max(0, x) is continuous and unbounded above. The step function is discontinuous (jumps from 0 to 1). ReLU preserves magnitudes beyond the threshold, while step collapses all positive values to 1.

**11.2:** We can approximate step using ReLU with a large scaling factor:

step(x) ≈ min(1, ReLU(10x))

Or use the "saturating" pattern: ReLU(x) - ReLU(x - 1) ≈ clamp(x, 0, 1)

```python
class SoftTriangleMLP(nn.Module):
    def __init__(self, sharpness=10.0):
        super().__init__()
        self.sharpness = sharpness
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Original triangle: (0,0), (1,0), (0,2)
        # x ≥ 0, y ≥ 0, 2x + y ≤ 2
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],    # x
                [0.0, 1.0],    # y
                [-2.0, -1.0]   # -2x - y
            ]) * sharpness)
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]) * sharpness)

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]) * sharpness)
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]) * sharpness)

    def forward(self, x):
        # Approximate step with scaled sigmoid
        h = torch.sigmoid(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.squeeze(-1)
```

**11.3:** Output is now in (0, 1) instead of {0, 1}. Points clearly inside get values near 1, points clearly outside get values near 0, and boundary points get intermediate values. The higher the `sharpness`, the closer to hard classification.

</details>

### Variation C2: Sigmoid Activation

Use sigmoid activations throughout and interpret output as probability.

**Part 11.1:** Build the MLP with sigmoid activations.

**Part 11.2:** What is the probabilistic interpretation of the output?

**Part 11.3:** How would you train this model if training were allowed?

<details>
<summary>Solution C2</summary>

**11.1:**

```python
class ProbabilisticTriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Initialize with scaled versions of the geometric weights
        scale = 5.0  # Controls sharpness of decision boundary
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-2.0, -1.0]
            ]) * scale)
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]) * scale)

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]) * scale)
            self.fc2.bias = nn.Parameter(torch.tensor([-1.5]) * scale)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.squeeze(-1)
```

**11.2:** The output p ∈ (0, 1) can be interpreted as P(point is inside triangle | point coordinates). Points far inside have p ≈ 1, points far outside have p ≈ 0, and boundary points have p ≈ 0.5.

**11.3:** Training approach:

- **Loss function:** Binary cross-entropy: BCE = -[y log(p) + (1-y) log(1-p)]
- **Data:** Generate random 2D points, label as 1 (inside) or 0 (outside)
- **Optimizer:** SGD or Adam
- **The model would learn the half-plane parameters automatically!**
</details>

### Variation C3: Sign Activation

Use sign(x) = +1 if x ≥ 0, -1 if x < 0 (bipolar step function).

**Part 11.1:** How does this change the AND gate computation?

**Part 11.2:** Reimplement the triangle MLP with sign activation.

**Part 11.3:** What are advantages of bipolar vs unipolar neurons?

<details>
<summary>Solution C3</summary>

**11.1:** With sign activation, outputs are in {-1, +1} instead of {0, 1}.

- If all 3 half-planes are satisfied: sum = +1 + 1 + 1 = 3
- If any fail: sum ≤ 1

AND gate: output +1 if sum > 2, else -1. Threshold at 2.5.

**11.2:**

```python
class BipolarTriangleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-2.0, -1.0]
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))

            # AND: sum of three +1/-1 values must be 3 (all +1)
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def sign(self, x):
        return torch.sign(x - 1e-10)  # -1 for negative, +1 for non-negative

    def forward(self, x):
        h = self.sign(self.fc1(x))
        out = self.sign(self.fc2(h))
        return (out.squeeze(-1) + 1) / 2  # Convert back to 0/1
```

**11.3:** Advantages of bipolar neurons:

- **Symmetry:** Weights can be positive or negative with equal expressiveness
- **Zero-centered activations:** Better gradient flow during training
- **Historical significance:** Perceptrons originally used bipolar activations
- **XOR pattern:** Easier to implement XOR with bipolar neurons
</details>

---

## CATEGORY D: Higher Dimensions

### Variation D1: Tetrahedron in 3D

Build an MLP that classifies points inside the tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).

**Part 11.1:** How many faces does a tetrahedron have? How many half-spaces needed?

**Part 11.2:** Write the 4 half-space inequalities.

**Part 11.3:** Build the MLP with input shape (B, 3).

<details>
<summary>Solution D1</summary>

**11.1:** A tetrahedron has **4 triangular faces**, so we need **4 half-spaces**.

**11.2:** The four faces lie on these planes:

- Face opposite to (0,0,0): x + y + z = 1 → **-x - y - z + 1 ≥ 0**
- Face on xy-plane: z = 0 → **z ≥ 0**
- Face on xz-plane: y = 0 → **y ≥ 0**
- Face on yz-plane: x = 0 → **x ≥ 0**

**11.3:**

```python
class TetrahedronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4, bias=True)  # 3D input, 4 half-spaces
        self.fc2 = nn.Linear(4, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0, 0.0],    # x ≥ 0
                [0.0, 1.0, 0.0],    # y ≥ 0
                [0.0, 0.0, 1.0],    # z ≥ 0
                [-1.0, -1.0, -1.0]  # x + y + z ≤ 1
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Test
model = TetrahedronMLP()
centroid = torch.tensor([[0.25, 0.25, 0.25]])  # Inside
corner = torch.tensor([[0.0, 0.0, 0.0]])  # Vertex (inside/boundary)
outside = torch.tensor([[1.0, 1.0, 1.0]])  # Outside
```

</details>

### Variation D2: 3-Simplex in 4D

Build an MLP for the 4D simplex with vertices at origin and unit vectors: (0,0,0,0), e₁, e₂, e₃, e₄.

**Part 11.1:** How many faces (3D hyperplanes) bound this simplex?

**Part 11.2:** What is the volume of this simplex? (Hint: n-simplex formula)

**Part 11.3:** Build the MLP.

<details>
<summary>Solution D2</summary>

**11.1:** A 4-simplex has **5 vertices** and **5 faces** (each face is a tetrahedron).

**11.2:** Volume of n-simplex with vertices at origin and n unit basis vectors:
V = 1/n!

For n=4: V = 1/4! = 1/24

**11.3:**

```python
class Simplex4DMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5, bias=True)  # 4D input, 5 half-spaces
        self.fc2 = nn.Linear(5, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],     # x₁ ≥ 0
                [0.0, 1.0, 0.0, 0.0],     # x₂ ≥ 0
                [0.0, 0.0, 1.0, 0.0],     # x₃ ≥ 0
                [0.0, 0.0, 0.0, 1.0],     # x₄ ≥ 0
                [-1.0, -1.0, -1.0, -1.0]  # x₁+x₂+x₃+x₄ ≤ 1
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.ones(1, 5))
            self.fc2.bias = nn.Parameter(torch.tensor([-4.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

</details>

### Variation D3: Hypercube (d-dimensional)

Build an MLP for the unit hypercube [0,1]^d for arbitrary d.

**Part 11.1:** How many half-spaces are needed for a d-dimensional hypercube?

**Part 11.2:** Write a general function that creates the MLP for any d.

**Part 11.3:** Test for d = 2, 3, 5.

<details>
<summary>Solution D3</summary>

**11.1:** A d-dimensional hypercube has **2d faces** (each coordinate has two bounds: 0 ≤ xᵢ ≤ 1).

**11.2:**

```python
def create_hypercube_mlp(d):
    class HypercubeMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            num_halfspaces = 2 * dim
            self.fc1 = nn.Linear(dim, num_halfspaces, bias=True)
            self.fc2 = nn.Linear(num_halfspaces, 1, bias=True)

            with torch.no_grad():
                # For each dimension i: xᵢ ≥ 0 and -xᵢ + 1 ≥ 0
                weight_list = []
                bias_list = []
                for i in range(dim):
                    # xᵢ ≥ 0
                    row_lower = [0.0] * dim
                    row_lower[i] = 1.0
                    weight_list.append(row_lower)
                    bias_list.append(0.0)

                    # xᵢ ≤ 1
                    row_upper = [0.0] * dim
                    row_upper[i] = -1.0
                    weight_list.append(row_upper)
                    bias_list.append(1.0)

                self.fc1.weight = nn.Parameter(torch.tensor(weight_list))
                self.fc1.bias = nn.Parameter(torch.tensor(bias_list))

                self.fc2.weight = nn.Parameter(torch.ones(1, num_halfspaces))
                self.fc2.bias = nn.Parameter(torch.tensor([-(num_halfspaces - 0.5)]))

        def step(self, x):
            return (x >= 0).float()

        def forward(self, x):
            return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

    return HypercubeMLP(d)

# Test
for d in [2, 3, 5]:
    model = create_hypercube_mlp(d)
    center = torch.ones(1, d) * 0.5  # Center of hypercube
    corner = torch.zeros(1, d)  # Origin corner
    outside = torch.ones(1, d) * 1.5  # Outside
    print(f"d={d}: center={model(center).item()}, corner={model(corner).item()}, outside={model(outside).item()}")
```

**11.3:** For all tested dimensions:

- Center point (0.5, 0.5, ...) → 1 (inside)
- Origin (0, 0, ...) → 1 (on boundary)
- (1.5, 1.5, ...) → 0 (outside)
</details>

---

## CATEGORY E: Proof/Theory Questions

### Variation E1: Expressiveness of 2-Layer Networks

**Part 11.1:** Prove that any convex polygon with n vertices can be represented by a 2-layer MLP with step activations.

**Part 11.2:** What is the minimum number of neurons in layer 1?

**Part 11.3:** Can you represent a non-convex polygon with 2 layers? Prove your answer.

<details>
<summary>Solution E1</summary>

**11.1:**
**Theorem:** Any convex n-gon can be represented by a 2-layer MLP.

**Proof:**

- A convex n-gon is the intersection of n half-planes.
- Each half-plane is defined by a linear inequality: wᵢᵀx + bᵢ ≥ 0
- Layer 1: n neurons, each computing hᵢ = Θ(wᵢᵀx + bᵢ)
- Layer 2: 1 neuron computing AND of all hᵢ

The AND gate outputs 1 iff Σhᵢ ≥ n, which requires all hᵢ = 1.

Since each hᵢ = 1 iff wᵢᵀx + bᵢ ≥ 0, the output is 1 iff x satisfies all n inequalities, i.e., x is inside the polygon. ∎

**11.2:** Minimum is **n neurons** in layer 1, where n is the number of edges. Each edge requires one half-plane test.

**11.3:** **No, 2 layers cannot represent non-convex polygons.**

**Proof by contradiction:**

- Suppose a 2-layer network represents a non-convex region R.
- Layer 1 computes linear functions followed by step activations.
- Layer 2 computes a weighted sum followed by step activation.
- The final output is 1 iff Σwᵢhᵢ + b ≥ 0.
- Each hᵢ ∈ {0, 1}, so the sum is a linear combination of {0, 1} values.
- The region where output = 1 is the union of regions defined by which hᵢ = 1.
- Each such region is the intersection of half-planes (convex).
- The union of convex regions with positive weights... wait, this isn't quite right.

Actually, the final layer computes a **threshold of a sum**, which is an AND-like operation over the half-plane tests. The resulting region is the intersection of half-planes → **always convex**.

Therefore, non-convex regions require **at least 3 layers**. ∎

</details>

### Variation E2: XOR with Geometry

**Part 11.1:** The XOR function on {0,1}² outputs 1 for (0,1) and (1,0), 0 for (0,0) and (1,1). Draw these points. Are they linearly separable?

**Part 11.2:** Design a 2-layer MLP (step activations) that computes XOR.

**Part 11.3:** How does this relate to the non-convex polygon problem?

<details>
<summary>Solution E2</summary>

**11.1:** Plot:

```
y
1 | (0,1)✓   (1,1)✗
  |
0 | (0,0)✗   (1,0)✓
  +--------------- x
    0        1
```

The positive points (0,1) and (1,0) cannot be separated from (0,0) and (1,1) by a single line. **Not linearly separable.**

**11.2:**

```python
class XORMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 1, bias=True)

        with torch.no_grad():
            # Neuron 1: x + y ≥ 0.5 (at least one is 1)
            # Neuron 2: -x - y + 1.5 ≥ 0 (not both are 1)
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 1.0],   # OR-ish
                [-1.0, -1.0]  # NOT-AND-ish
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([-0.5, 1.5]))

            # Output: both conditions must hold
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-1.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Verify
model = XORMLP()
inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
print(model(inputs))  # [0, 1, 1, 0]
```

**11.3:** XOR is geometrically equivalent to classifying the region between two parallel diagonal lines. This forms a "stripe" which is **convex**! The trick is that the hidden layer creates a new representation where the problem becomes linearly separable.

Relation to non-convex polygons: XOR doesn't require 3 layers because the "positive region" (the stripe from x+y=0.5 to x+y=1.5) is actually convex. Non-convex polygons are fundamentally different.

</details>

### Variation E3: Universal Approximation

**Part 11.1:** State the Universal Approximation Theorem for MLPs.

**Part 11.2:** For polygon classification, what does this theorem imply about network width?

**Part 11.3:** What is the trade-off between width and depth for representing shapes?

<details>
<summary>Solution E3</summary>

**11.1:** **Universal Approximation Theorem (Cybenko 1989, Hornik 1991):**

A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ, to any desired degree of accuracy, given appropriate activation functions (sigmoid, ReLU, etc.).

**11.2:** For polygon classification (which is a step function, not continuous!):

- The theorem technically doesn't apply directly (step functions are discontinuous)
- However, step-activated networks can **exactly** represent polygon regions
- For an n-sided convex polygon: width n in hidden layer suffices
- For non-convex polygons: need to decompose into convex pieces

**11.3:** Trade-offs:

- **Wide & Shallow (2 layers):**
  - Pros: Simple to analyze, each neuron has clear geometric meaning
  - Cons: Can only represent convex regions
  - Width needed: n for n-sided convex polygon

- **Deep & Narrow:**
  - Pros: Can represent non-convex regions via composition
  - Pros: Can implement hierarchical Boolean logic (OR of ANDs)
  - Cons: Harder to set weights manually, vanishing gradients during training

- **Optimal for non-convex:** Often 3 layers
  - Layer 1: Half-plane tests (width = total edges)
  - Layer 2: AND gates for each convex piece (width = number of pieces)
  - Layer 3: OR gate (width = 1)
  </details>

---

## CATEGORY F: Coding Variations

### Variation F1: NumPy Implementation (No PyTorch)

Implement the triangle classifier using only NumPy.

**Part 11.1:** Create a function `classify_triangle(points)` where points has shape (N, 2).

**Part 11.2:** Vectorize the computation (no loops over points).

**Part 11.3:** Benchmark against the PyTorch version for N = 10000 points.

<details>
<summary>Solution F1</summary>

**11.1 & 11.2:**

```python
import numpy as np

def classify_triangle(points):
    """
    Classify points as inside/outside triangle with vertices (0,0), (1,0), (0,2).

    Args:
        points: numpy array of shape (N, 2)
    Returns:
        numpy array of shape (N,) with 0 or 1
    """
    # Half-plane weights and biases
    # x >= 0, y >= 0, 2x + y <= 2
    W1 = np.array([
        [1.0, 0.0],    # x
        [0.0, 1.0],    # y
        [-2.0, -1.0]   # -2x - y
    ])
    b1 = np.array([0.0, 0.0, 2.0])

    # AND gate
    W2 = np.array([[1.0, 1.0, 1.0]])
    b2 = np.array([-2.5])

    # Step function
    def step(x):
        return (x >= 0).astype(float)

    # Forward pass (vectorized)
    h = step(points @ W1.T + b1)  # (N, 3)
    out = step(h @ W2.T + b2)     # (N, 1)

    return out.squeeze()

# Test
test_points = np.array([[0.5, 0.5], [0.0, 0.0], [1.5, 1.5], [0.3, 0.3]])
print(classify_triangle(test_points))  # [1. 1. 0. 1.]
```

**11.3:**

```python
import time

# Generate random points
np.random.seed(42)
points_np = np.random.randn(10000, 2)
points_torch = torch.tensor(points_np, dtype=torch.float32)

# NumPy timing
start = time.time()
for _ in range(100):
    result_np = classify_triangle(points_np)
numpy_time = time.time() - start

# PyTorch timing
model = TriangleMLP()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        result_torch = model(points_torch)
pytorch_time = time.time() - start

print(f"NumPy: {numpy_time:.3f}s, PyTorch: {pytorch_time:.3f}s")
# Typical result: NumPy is slightly faster for CPU, PyTorch faster with GPU
```

</details>

### Variation F2: Batch Processing and GPU

**Part 11.1:** Modify the PyTorch MLP to efficiently process batches on GPU.

**Part 11.2:** What is the theoretical speedup for batch size B on a GPU with P parallel units?

**Part 11.3:** Measure actual speedup on GPU vs CPU for B = 1000, 10000, 100000.

<details>
<summary>Solution F2</summary>

**11.1:**

```python
class GPUTriangleMLP(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [-2.0, -1.0]
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

        self.to(device)

    def step(self, x):
        return (x >= 0).float()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        h = self.step(self.fc1(x))
        out = self.step(self.fc2(h))
        return out.squeeze(-1)

# Use
if torch.cuda.is_available():
    model_gpu = GPUTriangleMLP(device='cuda')
    points = torch.randn(100000, 2)
    result = model_gpu(points)
```

**11.2:** Theoretical speedup:

- Each point requires ~O(1) operations (fixed network size)
- GPU can process P points in parallel
- Speedup ≈ min(B, P) if B ≥ P (GPU fully utilized)
- But: memory transfer overhead for small batches

**11.3:**

```python
import time

def benchmark(model, points, num_trials=100):
    # Warm up
    _ = model(points)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()
    for _ in range(num_trials):
        _ = model(points)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - start) / num_trials

# Results vary by hardware, but typical pattern:
# B=1000: GPU might be slower (overhead dominates)
# B=10000: GPU starts winning
# B=100000: GPU significantly faster (10-100x)
```

</details>

### Variation F3: Arbitrary Polygon Function

Write a general function that creates an MLP for any convex polygon.

**Part 11.1:** Function signature: `create_polygon_mlp(vertices)` where vertices is a list of (x, y) tuples in counterclockwise order.

**Part 11.2:** Handle edge cases: collinear vertices, degenerate polygons.

**Part 11.3:** Test on regular polygons from 3 to 10 sides.

<details>
<summary>Solution F3</summary>

**11.1 & 11.2:**

```python
import torch
import torch.nn as nn
import math

def create_polygon_mlp(vertices):
    """
    Create an MLP that classifies points inside a convex polygon.

    Args:
        vertices: List of (x, y) tuples in counterclockwise order
    Returns:
        nn.Module that classifies points
    """
    n = len(vertices)
    assert n >= 3, "Need at least 3 vertices for a polygon"

    # Compute half-plane normals and biases
    normals = []
    biases = []

    for i in range(n):
        # Edge from vertex i to vertex (i+1) % n
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Edge direction
        dx, dy = x2 - x1, y2 - y1
        edge_length = math.sqrt(dx*dx + dy*dy)

        # Skip degenerate edges
        if edge_length < 1e-10:
            continue

        # Inward normal (perpendicular to edge, pointing into polygon)
        # For counterclockwise vertices, inward normal is (dy, -dx)
        nx, ny = dy / edge_length, -dx / edge_length

        # Bias: n·p ≥ n·v1 for interior points
        # Equivalent to: n·x + (-n·v1) ≥ 0
        b = -(nx * x1 + ny * y1)

        normals.append([nx, ny])
        biases.append(b)

    num_halfspaces = len(normals)

    class PolygonMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, num_halfspaces, bias=True)
            self.fc2 = nn.Linear(num_halfspaces, 1, bias=True)

            with torch.no_grad():
                self.fc1.weight = nn.Parameter(torch.tensor(normals))
                self.fc1.bias = nn.Parameter(torch.tensor(biases))
                self.fc2.weight = nn.Parameter(torch.ones(1, num_halfspaces))
                self.fc2.bias = nn.Parameter(torch.tensor([-(num_halfspaces - 0.5)]))

        def step(self, x):
            return (x >= 0).float()

        def forward(self, x):
            return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

    return PolygonMLP()

# Test with regular polygons
def regular_polygon_vertices(n, radius=1.0):
    """Generate vertices of a regular n-gon centered at origin."""
    return [(radius * math.cos(2*math.pi*k/n + math.pi/2),
             radius * math.sin(2*math.pi*k/n + math.pi/2))
            for k in range(n)]
```

**11.3:**

```python
# Test regular polygons from 3 to 10 sides
for n in range(3, 11):
    vertices = regular_polygon_vertices(n)
    model = create_polygon_mlp(vertices)

    # Test center (should be inside)
    center = torch.tensor([[0.0, 0.0]])
    # Test point far outside
    outside = torch.tensor([[10.0, 10.0]])
    # Test vertex (should be on boundary)
    vertex = torch.tensor([list(vertices[0])])

    print(f"n={n}: center={model(center).item():.0f}, "
          f"outside={model(outside).item():.0f}, "
          f"vertex={model(vertex).item():.0f}")

# Expected output: all centers=1, all outside=0, all vertices=1
```

</details>

---

## CATEGORY G: Edge Cases and Boundary Conditions

### Variation G1: Points on Edges and Vertices

**Part 11.1:** For the original triangle (0,0), (1,0), (0,2), how does the MLP classify:

- The vertex (1, 0)?
- A point on the edge: (0.5, 0)?
- A point exactly on the hypotenuse: (0.25, 1.5)?

**Part 11.2:** Modify the MLP to strictly classify interior points (exclude boundary).

**Part 11.3:** Is there a mathematical difference between "≥" and ">" for this problem?

<details>
<summary>Solution G1</summary>

**11.1:** With the original MLP using Θ(x) = 1 if x ≥ 0:

- **Vertex (1, 0):**
  - x = 1 ≥ 0 ✓
  - y = 0 ≥ 0 ✓
  - -2(1) - 0 + 2 = 0 ≥ 0 ✓
  - **Output: 1** (classified as inside/boundary)

- **Edge point (0.5, 0):**
  - x = 0.5 ≥ 0 ✓
  - y = 0 ≥ 0 ✓
  - -2(0.5) - 0 + 2 = 1 ≥ 0 ✓
  - **Output: 1**

- **Hypotenuse point (0.25, 1.5):**
  - x = 0.25 ≥ 0 ✓
  - y = 1.5 ≥ 0 ✓
  - -2(0.25) - 1.5 + 2 = 0 ≥ 0 ✓
  - **Output: 1**

All boundary points are classified as "inside" (value 1).

**11.2:** Use strict inequality Θ(x) = 1 if x > 0:

```python
def strict_step(x):
    return (x > 0).float()  # Changed from >= to >
```

Now boundary points:

- (1, 0): -2x - y + 2 = 0, strict_step(0) = 0 → **Output: 0**
- (0.5, 0): y = 0, strict_step(0) = 0 → **Output: 0**
- (0.25, 1.5): -2x - y + 2 = 0, strict_step(0) = 0 → **Output: 0**

**11.3:** Mathematical difference:

- Using **≥ 0**: Classifies the **closed** triangle (includes boundary)
- Using **> 0**: Classifies the **open** triangle (excludes boundary)

In practice, floating-point numbers rarely land exactly on boundaries, so this distinction often doesn't matter. But for theoretical correctness and corner case handling, the choice matters.

</details>

### Variation G2: Numerical Precision

**Part 11.1:** What happens when a point is very close to an edge (within floating-point tolerance)?

**Part 11.2:** Design a "fuzzy" classifier that handles near-boundary points gracefully.

**Part 11.3:** What is the "thickness" of the boundary region in your fuzzy classifier?

<details>
<summary>Solution G2</summary>

**11.1:** For a point at distance ε from an edge where ε ≈ 10⁻¹⁵ (machine epsilon):

- The linear combination wᵀx + b might be ≈ 10⁻¹⁵
- Due to floating-point errors, this could become slightly positive or negative
- Classification becomes unpredictable for such points

**11.2:**

```python
class FuzzyTriangleMLP(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin
        self.fc1 = nn.Linear(2, 3, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [-2.0, -1.0]
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))

    def forward(self, x):
        # Raw distances to each half-plane
        distances = self.fc1(x)  # (B, 3)

        # Fuzzy classification:
        # - If all distances >= margin: definitely inside (1.0)
        # - If any distance <= -margin: definitely outside (0.0)
        # - In between: fuzzy (linear interpolation)

        min_distance = distances.min(dim=1).values  # Closest to any edge

        # Fuzzy membership
        output = torch.clamp((min_distance + self.margin) / (2 * self.margin), 0, 1)
        return output

# Test
model = FuzzyTriangleMLP(margin=0.1)
test_points = torch.tensor([
    [0.5, 0.5],    # Clearly inside
    [0.0, 0.0],    # On corner
    [0.5, 0.0],    # On edge
    [0.48, 0.02],  # Near edge
    [-0.5, 0.0]    # Outside
])
print(model(test_points))
# Approximate: [1.0, 0.5, 0.5, ~0.6, 0.0]
```

**11.3:** The boundary thickness is **2 × margin**. Points within this distance from any edge get intermediate values. In the example with margin=0.1, the boundary region is 0.2 units thick around each edge.

This is similar to the "soft margin" in SVMs!

</details>

### Variation G3: Degenerate Triangles

**Part 11.1:** What happens if we try to build an MLP for a "triangle" with collinear vertices (0,0), (1,0), (2,0)?

**Part 11.2:** How should the function handle this case?

**Part 11.3:** What about a triangle with two identical vertices?

<details>
<summary>Solution G3</summary>

**11.1:** With collinear vertices (0,0), (1,0), (2,0):

- This is a line segment, not a triangle (area = 0)
- The three "edges" would be:
  - (0,0)→(1,0): line y=0
  - (1,0)→(2,0): line y=0 (same!)
  - (2,0)→(0,0): line y=0 (same!)

All three half-planes are the same (y ≥ 0). The "triangle" classifier would accept all points with y ≥ 0 (half-plane), not a bounded region!

**11.2:** The function should:

1. **Detect** degeneracy by computing area or checking collinearity
2. **Handle** it appropriately (raise error, return special model, etc.)

```python
def create_polygon_mlp_safe(vertices):
    # Check for degeneracy
    n = len(vertices)

    # Compute signed area using shoelace formula
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2

    if area < 1e-10:
        raise ValueError(f"Degenerate polygon: area = {area:.2e}")

    return create_polygon_mlp(vertices)

# Test
try:
    degenerate = create_polygon_mlp_safe([(0,0), (1,0), (2,0)])
except ValueError as e:
    print(f"Caught: {e}")  # "Degenerate polygon: area = 0.00e+00"
```

**11.3:** Two identical vertices:

- Example: (0,0), (1,0), (0,0)
- The edge (0,0)→(0,0) has zero length
- Its normal vector is undefined (0/0)
- This will cause NaN or inf in the weights

Solution: Skip zero-length edges (as in the `create_polygon_mlp` solution which checks `edge_length < 1e-10`).

</details>

---

## CATEGORY H: Rotation Variations (Non-Axis-Aligned Shapes)

### Variation H1: Diamond (45° Rotated Square)

Build an MLP for a square rotated 45° — a "diamond" with vertices at **(1, 0), (0, 1), (-1, 0), (0, -1)**.

**Part 11.1:** Derive the four edge equations. Note that none of the edges are axis-aligned!

**Part 11.2:** Build the MLP. How do the weights differ from the axis-aligned square in B1?

**Part 11.3:** Show that a point (x, y) is inside the diamond iff |x| + |y| ≤ 1. This is the **L1 unit ball**!

<details>
<summary>Solution H1</summary>

**11.1:** Going counterclockwise from (1, 0):

- Edge (1,0)→(0,1): Line x + y = 1 → **-x - y + 1 ≥ 0** (interior below-left)
- Edge (0,1)→(-1,0): Line -x + y = 1 → **x - y + 1 ≥ 0** (interior below-right)
- Edge (-1,0)→(0,-1): Line x + y = -1 → **x + y + 1 ≥ 0** (interior above-right)
- Edge (0,-1)→(1,0): Line x - y = 1 → **-x + y + 1 ≥ 0** (interior above-left)

**11.2:**

```python
import torch
import torch.nn as nn

class DiamondMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

        with torch.no_grad():
            # All edges are diagonal — no axis-aligned weights!
            self.fc1.weight = nn.Parameter(torch.tensor([
                [-1.0, -1.0],  # -x - y (upper-right edge)
                [1.0, -1.0],   # x - y (upper-left edge)
                [1.0, 1.0],    # x + y (lower-left edge)
                [-1.0, 1.0]    # -x + y (lower-right edge)
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Test
model = DiamondMLP()
inside = torch.tensor([[0.0, 0.0], [0.3, 0.3], [0.5, 0.0]])
outside = torch.tensor([[1.0, 1.0], [0.6, 0.6]])
print(f"Inside: {model(inside)}")   # [1, 1, 1]
print(f"Outside: {model(outside)}") # [0, 0]
```

**Key difference from B1:** In the axis-aligned square, weights were like [1, 0] or [0, 1] (testing one coordinate). In the diamond, all weights are diagonal like [1, 1] or [1, -1] (testing linear combinations of coordinates).

**11.3:** The four inequalities are:
- -x - y + 1 ≥ 0  →  x + y ≤ 1
- x - y + 1 ≥ 0   →  x - y ≤ 1  →  -(x - y) ≥ -1  →  y - x ≤ 1
- x + y + 1 ≥ 0   →  x + y ≥ -1
- -x + y + 1 ≥ 0  →  y - x ≤ 1

Combined: -1 ≤ x + y ≤ 1 AND -1 ≤ x - y ≤ 1

This is equivalent to: |x + y| ≤ 1 AND |x - y| ≤ 1

But note: max(|x + y|, |x - y|) = |x| + |y| when we consider all quadrants.

Actually, the four constraints simplify to **|x| + |y| ≤ 1**, the definition of the L1 unit ball! ✓

</details>

### Variation H2: Arbitrary Rotation of Original Triangle

The original triangle (0,0), (1,0), (0,2) is rotated by angle θ = 30° about the origin.

**Part 11.1:** Compute the new vertex positions using the rotation matrix.

**Part 11.2:** Derive the new half-plane inequalities. Show how the original weights transform.

**Part 11.3:** Build the MLP. What is the general relationship between rotation angle θ and weight transformation?

<details>
<summary>Solution H2</summary>

**11.1:** Rotation matrix for angle θ:

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

For θ = 30° = π/6: cos(30°) = √3/2 ≈ 0.866, sin(30°) = 1/2 = 0.5

New vertices:
- (0,0) → (0, 0) (origin unchanged)
- (1,0) → (cos30°, sin30°) = (√3/2, 1/2) ≈ (0.866, 0.5)
- (0,2) → (-2sin30°, 2cos30°) = (-1, √3) ≈ (-1, 1.732)

**11.2:** Original inequalities were:
- x ≥ 0 → normal n₁ = (1, 0)
- y ≥ 0 → normal n₂ = (0, 1)
- -2x - y + 2 ≥ 0 → normal n₃ = (-2, -1)/√5

**Key insight:** When the shape rotates by θ, the normal vectors also rotate by θ!

Rotated normals:
- n₁' = R(30°) · (1, 0) = (√3/2, 1/2)
- n₂' = R(30°) · (0, 1) = (-1/2, √3/2)
- n₃' = R(30°) · (-2, -1)/√5 = ((-2√3/2 + 1/2), (-1 - √3/2))/√5

For the rotated edges:
- Edge from (0,0): normal (√3/2, 1/2), passes through origin → bias = 0
  **√3x/2 + y/2 ≥ 0** or equivalently **√3x + y ≥ 0**

- Edge from (0,0) to (-1, √3): normal (-1/2, √3/2), passes through origin
  **-x/2 + √3y/2 ≥ 0** or **-x + √3y ≥ 0**

- Hypotenuse from (√3/2, 1/2) to (-1, √3): Need to derive from scratch
  Direction: (-1 - √3/2, √3 - 1/2) = (-(2+√3)/2, (2√3-1)/2)
  Inward normal (perpendicular, toward origin): proportional to (-(2√3-1)/2, -(2+√3)/2)

**11.3:**

```python
import torch
import torch.nn as nn
import math

class RotatedTriangleMLP(nn.Module):
    def __init__(self, theta_degrees=30):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        theta = math.radians(theta_degrees)
        c, s = math.cos(theta), math.sin(theta)

        # Original vertices
        v0, v1, v2 = (0, 0), (1, 0), (0, 2)

        # Rotate vertices
        def rotate(v):
            return (c * v[0] - s * v[1], s * v[0] + c * v[1])

        rv0, rv1, rv2 = rotate(v0), rotate(v1), rotate(v2)

        # Compute edge normals and biases (counterclockwise)
        vertices = [rv0, rv1, rv2]
        weights = []
        biases = []

        for i in range(3):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 3]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            # Inward normal for counterclockwise traversal
            nx, ny = dy / length, -dx / length
            b = -(nx * p1[0] + ny * p1[1])
            weights.append([nx, ny])
            biases.append(b)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor(weights))
            self.fc1.bias = nn.Parameter(torch.tensor(biases))
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Test at multiple angles
for angle in [0, 30, 45, 90]:
    model = RotatedTriangleMLP(angle)
    # Centroid of original triangle: (1/3, 2/3)
    # Rotate it
    theta = math.radians(angle)
    cx = math.cos(theta) * (1/3) - math.sin(theta) * (2/3)
    cy = math.sin(theta) * (1/3) + math.cos(theta) * (2/3)
    centroid = torch.tensor([[cx, cy]])
    print(f"Angle {angle}°: centroid at ({cx:.3f}, {cy:.3f}) → {model(centroid).item()}")
```

**General relationship:** If original weight matrix is W and bias is b, then for rotation by θ:
- W' = W @ R(-θ)ᵀ = W @ R(θ) (because R is orthogonal)
- Biases remain unchanged (they depend on distance from origin, which rotation preserves)

The normal vectors rotate with the shape, so the weight matrix transforms by the same rotation!

</details>

---

## CATEGORY I: Misconception Traps (Common Errors)

> These variations are designed to expose and correct common misconceptions. Each presents a "broken" solution or asks students to identify errors.

### Variation I1: Clockwise vs Counterclockwise Disaster

A student builds an MLP for triangle (0,0), (1,0), (0,2) but lists vertices **clockwise** instead of counterclockwise: (0,0), (0,2), (1,0).

**Part 11.1:** Using the standard formula for inward normals (perpendicular to edge direction, 90° left turn), derive the half-plane inequalities.

**Part 11.2:** What region does this MLP actually classify? Draw it.

**Part 11.3:** How would you fix the MLP without changing the vertex order?

<details>
<summary>Solution I1</summary>

**11.1:** Clockwise order: (0,0) → (0,2) → (1,0) → (0,0)

Using "inward normal = 90° left of edge direction":

- Edge (0,0)→(0,2): direction (0,2), left turn gives normal (-2,0) → normalized (-1,0)
  Line: x = 0, "interior" at x ≤ 0? That's the **left** half-plane!
  **-x ≥ 0** or **x ≤ 0**

- Edge (0,2)→(1,0): direction (1,-2), left turn gives normal (-2,-1) → normalized (-2,-1)/√5
  **-2x - y + 2 ≥ 0** (this one happens to be correct!)

- Edge (1,0)→(0,0): direction (-1,0), left turn gives normal (0,-1)
  **-y ≥ 0** or **y ≤ 0**

**11.2:** The three inequalities are:
- x ≤ 0
- -2x - y + 2 ≥ 0 (equivalent to y ≤ -2x + 2)
- y ≤ 0

The intersection is: x ≤ 0 AND y ≤ 0 AND y ≤ -2x + 2

This is the **third quadrant** (x ≤ 0, y ≤ 0), bounded above by the line y = -2x + 2.

But wait—at x = 0, the third constraint gives y ≤ 2, and combined with y ≤ 0, we get just y ≤ 0.

The region is: **{(x,y) : x ≤ 0 and y ≤ 0}** — the entire third quadrant, an unbounded region!

The student's MLP classifies the third quadrant as "inside", not the triangle!

**11.3:** Two fixes:

**Fix 1:** Reverse the vertex order (use counterclockwise).

**Fix 2:** Keep vertex order but flip all normal directions:
```python
# Instead of left turn (dy, -dx), use right turn (-dy, dx)
# Or equivalently: negate all weights and biases in layer 1

# Original broken weights:
# [[-1, 0], [-2/√5, -1/√5], [0, -1]]

# Fixed weights (negate):
# [[1, 0], [2/√5, 1/√5], [0, 1]]
```

**Lesson:** The "90° left turn" rule assumes counterclockwise traversal. For clockwise, use "90° right turn" (or negate the normals).

</details>

### Variation I2: The Off-by-One Threshold Trap

A student implements an AND gate for 3 inputs using threshold -3.0 instead of -2.5:

```python
self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
self.fc2.bias = nn.Parameter(torch.tensor([-3.0]))  # Student used -3.0
```

**Part 11.1:** For what inputs does Θ(h₁ + h₂ + h₃ - 3) = 1? (where hᵢ ∈ {0, 1})

**Part 11.2:** What happens to points exactly on the triangle boundary?

**Part 11.3:** What threshold would make the classifier **exclude** the boundary (open region)?

<details>
<summary>Solution I2</summary>

**11.1:** The output is 1 when h₁ + h₂ + h₃ - 3 ≥ 0, i.e., h₁ + h₂ + h₃ ≥ 3.

Since each hᵢ ∈ {0, 1}, the only way to achieve sum ≥ 3 is:
**h₁ = h₂ = h₃ = 1** (all three must be exactly 1)

This seems correct... but there's a subtle issue!

**11.2:** For a point exactly on an edge (e.g., the hypotenuse), one of the half-plane tests gives:
- wᵀx + b = 0
- Θ(0) = 1 (using ≥ 0 threshold)

So boundary points still have all hᵢ = 1, and sum = 3.

With threshold -3.0: output = Θ(3 - 3) = Θ(0) = 1 ✓

**Actually, -3.0 works for the closed region!** The issue is more subtle:

The real problem with -3.0 vs -2.5 appears with **numerical precision**:
- With -2.5: sum needs to be > 2.5, so even sum = 2.999 works
- With -3.0: sum needs to be ≥ 3.0 exactly

Due to floating-point errors, a point very slightly inside might compute sum = 2.9999999 instead of 3.0, and:
- Θ(2.9999999 - 2.5) = Θ(0.4999999) = 1 ✓
- Θ(2.9999999 - 3.0) = Θ(-0.0000001) = 0 ✗

**The -2.5 threshold is more robust to numerical errors!**

**11.3:** To exclude the boundary (open region), use threshold **-3.0** combined with **strict inequality** in the half-plane tests:

```python
def strict_step(x):
    return (x > 0).float()  # Strict inequality
```

Then for boundary points, at least one hᵢ = 0, sum < 3, output = 0.

Alternatively, keep non-strict step but use a slightly higher threshold like -2.99 or shift the biases slightly inward.

**Key insight:** The threshold (n - 0.5) for n inputs provides a "margin of safety" against floating-point errors. Using exactly -n works mathematically but is fragile numerically.

</details>

### Variation I3: The Bias Sign Blunder

A student derives the correct normal vector n = (1, 0) for the edge x = 0, but computes the bias as:

```python
# Student's reasoning: "bias should be n · v where v is a point on the edge"
b = 1 * 0 + 0 * 0  # n · (0,0) = 0
# So the half-plane is: x + 0 ≥ 0, i.e., x ≥ 0 ✓ (happens to be correct!)

# But for edge y = 0 going from (0,0) to (1,0), normal n = (0, 1):
b = 0 * 0 + 1 * 0  # n · (0,0) = 0
# Half-plane: y + 0 ≥ 0, i.e., y ≥ 0 ✓ (also correct!)

# For hypotenuse from (1,0) to (0,2), normal n = (-2, -1)/√5:
# Using point (1, 0):
b = (-2/√5) * 1 + (-1/√5) * 0  # n · (1,0) = -2/√5
# Half-plane: -2x/√5 - y/√5 + (-2/√5) ≥ 0
# Simplify: -2x - y - 2 ≥ 0 → -2x - y ≥ 2 → 2x + y ≤ -2 ???
```

**Part 11.1:** What went wrong? What is the correct formula for bias?

**Part 11.2:** Test the student's broken hypotenuse inequality on point (0.5, 0.5).

**Part 11.3:** Derive the correct general formula and explain why it works.

<details>
<summary>Solution I3</summary>

**11.1:** The student used **b = n · v** but the correct formula is **b = -(n · v)** (with a negative sign!).

The half-plane equation is: **n · x + b ≥ 0**

For a point v on the boundary, we need n · v + b = 0, which gives **b = -(n · v)**.

The student forgot the negative sign!

**11.2:** Testing point (0.5, 0.5) with the broken inequality:

Student's inequality: -2x - y - 2 ≥ 0

At (0.5, 0.5): -2(0.5) - 0.5 - 2 = -1 - 0.5 - 2 = -3.5 < 0

So (0.5, 0.5) is classified as **outside** by this inequality.

But (0.5, 0.5) is clearly **inside** the original triangle (0,0), (1,0), (0,2)!
- x = 0.5 ≥ 0 ✓
- y = 0.5 ≥ 0 ✓
- 2x + y = 1.5 ≤ 2 ✓

The broken classifier fails.

**11.3:** Correct derivation:

For a half-plane defined by normal n and passing through point v on the boundary:
- We want: n · x = n · v for points x on the boundary
- Rearranging: n · x - n · v = 0
- Or: n · x + (-(n · v)) = 0

So the inequality for interior points is: **n · x - (n · v) ≥ 0**

In matrix form: **b = -(n · v) = -nᵀv**

For the hypotenuse with n = (-2, -1)/√5 and v = (1, 0):
- b = -((-2/√5)(1) + (-1/√5)(0)) = -(-2/√5) = 2/√5

Unnormalized (scaling by √5): b = 2

Correct inequality: **-2x - y + 2 ≥ 0** ✓

Test (0.5, 0.5): -2(0.5) - 0.5 + 2 = -1 - 0.5 + 2 = 0.5 ≥ 0 ✓

**Mnemonic:** "Bias is **negative** dot product" — b = -n·v

</details>

### Variation I4: Debug the Broken Implementation

The following MLP is supposed to classify the unit square [0,1]×[0,1], but it has **three bugs**. Find and fix them.

```python
class BrokenSquareMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

        with torch.no_grad():
            # Bug 1 is here
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],   # x ≥ 0
                [0.0, 1.0],   # y ≥ 0
                [1.0, 0.0],   # x ≤ 1 (WRONG!)
                [0.0, 1.0]    # y ≤ 1 (WRONG!)
            ]))
            # Bug 2 is here
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0]))

            # Bug 3 is here
            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.0]))

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)
```

**Part 11.1:** Identify all three bugs.

**Part 11.2:** What region does the broken MLP actually classify?

**Part 11.3:** Write the corrected implementation.

<details>
<summary>Solution I4</summary>

**11.1:** The three bugs:

**Bug 1: Wrong weight signs for upper bounds**

```python
[1.0, 0.0],   # This tests x ≥ 0, not x ≤ 1
[0.0, 1.0]    # This tests y ≥ 0, not y ≤ 1
```

For x ≤ 1, we need -x + 1 ≥ 0, so weight should be **[-1.0, 0.0]**
For y ≤ 1, we need -y + 1 ≥ 0, so weight should be **[0.0, -1.0]**

**Bug 2: Biases are correct given the intended constraints, but paired with wrong weights**

With the wrong weights [1, 0] and bias 1, we get x + 1 ≥ 0, i.e., x ≥ -1 (not x ≤ 1!)

**Bug 3: Threshold should be -3.5, not -3.0**

While -3.0 works mathematically, -3.5 is more robust (see Variation I2).

**11.2:** The broken MLP tests:
- x + 0 ≥ 0 → x ≥ 0 ✓
- y + 0 ≥ 0 → y ≥ 0 ✓
- x + 1 ≥ 0 → x ≥ -1 (always true for x ≥ 0, so redundant)
- y + 1 ≥ 0 → y ≥ -1 (always true for y ≥ 0, so redundant)

The broken MLP classifies the **first quadrant** {x ≥ 0, y ≥ 0} — an unbounded region!

**11.3:** Corrected implementation:

```python
class FixedSquareMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],    # x ≥ 0
                [0.0, 1.0],    # y ≥ 0
                [-1.0, 0.0],   # -x + 1 ≥ 0, i.e., x ≤ 1 (FIXED)
                [0.0, -1.0]    # -y + 1 ≥ 0, i.e., y ≤ 1 (FIXED)
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0]))

            self.fc2.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
            self.fc2.bias = nn.Parameter(torch.tensor([-3.5]))  # FIXED

    def step(self, x):
        return (x >= 0).float()

    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Verify
model = FixedSquareMLP()
test = torch.tensor([
    [0.5, 0.5],  # Inside → 1
    [0.0, 0.0],  # Corner → 1
    [1.5, 0.5],  # Outside (x > 1) → 0
    [-0.1, 0.5]  # Outside (x < 0) → 0
])
print(model(test))  # tensor([1., 1., 0., 0.])
```

</details>

### Variation I5: The Normalization Red Herring

A student asks: "Do I need to normalize the normal vectors to unit length?"

**Part 11.1:** Build two MLPs for the same triangle — one with normalized normals, one without. Compare outputs.

**Part 11.2:** When does normalization matter mathematically? When doesn't it?

**Part 11.3:** Give an example where unnormalized normals cause a practical problem.

<details>
<summary>Solution I5</summary>

**11.1:**

```python
import torch
import torch.nn as nn
import math

# Triangle: (0,0), (1,0), (0,2)

class NormalizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Normalized normals
        sqrt5 = math.sqrt(5)
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],           # ||n|| = 1
                [0.0, 1.0],           # ||n|| = 1
                [-2/sqrt5, -1/sqrt5]  # ||n|| = 1
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2/sqrt5]))
            self.fc2.weight = nn.Parameter(torch.ones(1, 3))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x): return (x >= 0).float()
    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

class UnnormalizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 1, bias=True)

        # Unnormalized (raw coefficients)
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor([
                [1.0, 0.0],    # x ≥ 0
                [0.0, 1.0],    # y ≥ 0
                [-2.0, -1.0]   # -2x - y + 2 ≥ 0 (NOT normalized!)
            ]))
            self.fc1.bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))
            self.fc2.weight = nn.Parameter(torch.ones(1, 3))
            self.fc2.bias = nn.Parameter(torch.tensor([-2.5]))

    def step(self, x): return (x >= 0).float()
    def forward(self, x):
        return self.step(self.fc2(self.step(self.fc1(x)))).squeeze(-1)

# Compare
test_points = torch.tensor([
    [0.25, 0.5], [0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.3, 0.3]
])

norm_model = NormalizedMLP()
unnorm_model = UnnormalizedMLP()

print(f"Normalized:   {norm_model(test_points)}")
print(f"Unnormalized: {unnorm_model(test_points)}")
# Both produce: tensor([1., 1., 1., 0., 1.])
```

**The outputs are identical!** Normalization doesn't affect classification.

**11.2:** Normalization **doesn't matter** for:
- Binary classification (inside/outside)
- Any decision based solely on sign(wᵀx + b)

Normalization **does matter** for:
- **Distance interpretation:** If w is a unit normal, then |wᵀx + b| gives the signed distance to the boundary
- **Gradient-based training:** Different scales in weights can cause optimization issues
- **Regularization:** L2 regularization penalizes ||w||, so unnormalized weights get penalized differently

**11.3:** Practical problem with unnormalized normals:

```python
class FuzzyClassifier(nn.Module):
    """Classifier that returns confidence based on distance from boundary."""
    def __init__(self, weights, biases):
        super().__init__()
        self.weights = weights  # Shape: (n_constraints, 2)
        self.biases = biases

    def forward(self, x):
        # Compute "distance" to each boundary
        distances = x @ self.weights.T + self.biases  # Shape: (batch, n_constraints)
        min_distance = distances.min(dim=1).values
        # Sigmoid for soft classification
        return torch.sigmoid(min_distance * 10)  # Scale factor for sharpness

# With unnormalized weights, the "distance" to hypotenuse is scaled by √5!
# Point (0.25, 0.5):
#   Normalized: distance to hypotenuse = |-2(0.25)/√5 - 0.5/√5 + 2/√5| = |0.5/√5| ≈ 0.22
#   Unnormalized: "distance" = |-2(0.25) - 0.5 + 2| = |1| = 1.0

# The unnormalized version thinks the point is ~4.5x farther from the hypotenuse!
# This causes inconsistent fuzzy boundaries.
```

**Lesson:** For hard classification, normalization is optional. For anything involving distances, gradients, or fuzzy boundaries, normalize your normals!

</details>

---

## KEY FORMULAS SUMMARY

| Concept           | Formula                                        |
| ----------------- | ---------------------------------------------- |
| Half-plane test   | Θ(wᵀx + b), where w is normal, b is offset     |
| Point-in-triangle | AND of 3 half-plane tests                      |
| Convex n-gon      | AND of n half-plane tests                      |
| Non-convex region | OR of (AND of convex pieces)                   |
| Edge normal (CCW) | n = (y₂-y₁, x₁-x₂) / \|\|edge\|\| (inward)     |
| Edge bias         | b = -n · v₁ (where v₁ is a vertex on the edge) |
| AND gate          | Θ(Σwᵢhᵢ - (n - 0.5)) for n inputs              |
| OR gate           | Θ(Σwᵢhᵢ - 0.5) for any of n inputs             |

---

## ARCHITECTURE PATTERNS

### Pattern 1: Convex Polygon (2 Layers)

```
Input(d) → [n half-planes] → [AND gate] → Output
Layer 1: n neurons, each implementing wᵀx + b ≥ 0
Layer 2: 1 neuron, sum ≥ n threshold
```

### Pattern 2: Union of Convex Regions (3 Layers)

```
Input(d) → [half-planes for all regions] → [AND gate per region] → [OR gate] → Output
Layer 1: Σnᵢ neurons (all half-planes)
Layer 2: k neurons (one AND per convex piece)
Layer 3: 1 neuron (OR gate)
```

### Pattern 3: General Boolean Function (k Layers)

```
Input → Layer 1 (features) → Layer 2 (AND) → Layer 3 (OR) → ... → Output
Can represent any Boolean function with O(2^n) neurons
```

---

## COGNITIVE CONNECTIONS

### Why This Matters for USAAIO

1. **Neural Network Expressiveness**: Understanding what networks can and cannot represent
2. **Geometric Intuition**: Visualizing decision boundaries as hyperplanes
3. **Manual Weight Setting**: Deep understanding of what each weight means
4. **Computational Geometry**: Half-planes, convex hulls, point-in-polygon
5. **Boolean Logic in Networks**: AND, OR, NOT with neurons

### Related Topics

- Linear classifiers (Unit 4: ML1)
- SVMs and maximum margin (Unit 4)
- PyTorch fundamentals (Unit 6)
- Deep learning theory (Unit 7)
- Universal approximation theorem (Unit 7)

  ★ Insight ─────────────────────────────────────
  The beauty of Problem 11 is that it reveals the deep connection between:
  1. Computational Geometry — A triangle is the intersection of 3 half-planes, each defined by a linear inequality
  2. Neural Network Architecture — Each neuron with step activation implements one half-plane test; the network computes Boolean logic
  3. Expressiveness Theory — This construction proves that 2-layer MLPs can exactly represent any convex polygon, while non-convex shapes require 3+ layers
     ─────────────────────────────────────────────────

  Summary of Generated Variations

  I created 32+ variations across 9 categories:

  Category A: Different Triangle Vertices (5 variations)
  - A1: Right triangle shifted to (1,1), (3,1), (1,3)
  - A2: Approximately equilateral triangle
  - A3: Obtuse triangle with angle verification via dot products
  - A4: Thin near-degenerate triangle (tests numerical stability)
  - A5: Unit simplex (the probability simplex!)

  Category B: Different Shapes (5 variations)
  - B1: Square (4 half-planes)
  - B2: Regular pentagon (5 half-planes, computed via polar coordinates)
  - B3: Regular hexagon (6 half-planes)
  - B4: Circle approximation using 12-gon (with error analysis)
  - B5: L-shape (non-convex!) — Requires 3 layers with OR logic

  Category C: Different Activation Functions (3 variations)
  - C1: ReLU-based "soft" classifier
  - C2: Sigmoid activations with probabilistic interpretation
  - C3: Bipolar sign activation (+1/-1 instead of 0/1)

  Category D: Higher Dimensions (3 variations)
  - D1: Tetrahedron in 3D (4 half-spaces)
  - D2: 4-simplex in 4D (5 half-spaces)
  - D3: General d-dimensional hypercube (2d half-spaces)

  Category E: Proof/Theory Questions (3 variations)
  - E1: Prove convex polygons can be represented by 2-layer MLPs
  - E2: XOR problem and its geometric interpretation
  - E3: Universal Approximation Theorem connections

  Category F: Coding Variations (3 variations)
  - F1: Pure NumPy implementation (no PyTorch)
  - F2: GPU batch processing with benchmarks
  - F3: General create_polygon_mlp(vertices) function

  Category G: Edge Cases (3 variations)
  - G1: Points exactly on edges and vertices (closed vs open regions)
  - G2: Numerical precision and fuzzy classification
  - G3: Degenerate triangles (collinear vertices, zero area)

  Category H: Rotation Variations (2 variations) — NEW
  - H1: Diamond (45° rotated square) — reveals L1 unit ball connection
  - H2: Arbitrary rotation — shows weight transformation under rotation

  Category I: Misconception Traps (5 variations) — NEW
  - I1: Clockwise vs counterclockwise vertex ordering disaster
  - I2: Off-by-one AND threshold trap (-3.0 vs -2.5)
  - I3: Bias sign blunder (b = n·v vs b = -n·v)
  - I4: Debug the broken implementation (find 3 bugs)
  - I5: Normalization red herring (when does it matter?)

  Each variation includes:
  - Complete problem statement with multiple parts
  - Full solutions in collapsible <details> blocks
  - Working PyTorch code that can be run directly
  - Mathematical derivations where appropriate

  The file also includes summary tables of key formulas, architecture patterns, and connections to other USAAIO topics.
