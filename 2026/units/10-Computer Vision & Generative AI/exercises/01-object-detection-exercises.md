# Object Detection Exercises

**5 exercises** | Covers: IoU computation, NMS algorithm, anchor boxes, mAP, detection metrics

---

## Exercise 1: Compute IoU

**Target time**: 3 minutes

Two bounding boxes in $(x_1, y_1, x_2, y_2)$ format:
- Box A: $(2, 3, 8, 9)$
- Box B: $(5, 5, 11, 12)$

**Part 1**: Compute the intersection area.

**Part 2**: Compute the union area.

**Part 3**: Compute the IoU. Is this a "good" detection at IoU threshold 0.5?

<details>
<summary>Solution</summary>

**Part 1**: Intersection
- $x_1^I = \max(2, 5) = 5$, $y_1^I = \max(3, 5) = 5$
- $x_2^I = \min(8, 11) = 8$, $y_2^I = \min(9, 12) = 9$
- Intersection area $= (8-5) \times (9-5) = 3 \times 4 = 12$

**Part 2**: Union
- Area(A) $= (8-2) \times (9-3) = 6 \times 6 = 36$
- Area(B) $= (11-5) \times (12-5) = 6 \times 7 = 42$
- Union $= 36 + 42 - 12 = 66$

**Part 3**: IoU $= 12/66 = 2/11 \approx 0.182$

This is NOT a good detection at IoU threshold 0.5 (0.182 < 0.5).

</details>

---

## Exercise 2: Run NMS by Hand

**Target time**: 5 minutes

Given 5 detections (all same class) with bounding boxes and confidence scores:

| Detection | Box $(x_1, y_1, x_2, y_2)$ | Confidence |
|---|---|---|
| A | $(10, 10, 50, 50)$ | 0.9 |
| B | $(12, 12, 52, 52)$ | 0.85 |
| C | $(100, 100, 150, 150)$ | 0.7 |
| D | $(14, 14, 48, 48)$ | 0.6 |
| E | $(102, 98, 148, 148)$ | 0.5 |

IoU threshold for NMS: $\tau = 0.5$.

**Part 1**: Sort detections by confidence.

**Part 2**: Keep detection A (highest confidence). Compute IoU of A with each remaining detection. Which ones get suppressed?

**Part 3**: Continue NMS. What is the final set of kept detections?

<details>
<summary>Solution</summary>

**Part 1**: Sorted: A(0.9), B(0.85), C(0.7), D(0.6), E(0.5)

**Part 2**: Keep A = $(10, 10, 50, 50)$, area = 1600.

IoU(A, B): intersection = $(12,12,50,50)$ = $38 \times 38 = 1444$. Union = $1600 + 1600 - 1444 = 1756$. IoU = $1444/1756 \approx 0.822 > 0.5$. **Suppress B**.

IoU(A, C): No overlap (boxes are far apart). IoU = 0. **Keep C**.

IoU(A, D): intersection = $(14,14,48,48)$ = $34 \times 34 = 1156$. Union = $1600 + 1156 - 1156 = 1600$. IoU = $1156/1600 = 0.7225 > 0.5$. **Suppress D**.

IoU(A, E): No overlap. IoU = 0. **Keep E**.

Remaining after A: {C, E}

**Part 3**: Next highest: C(0.7). Keep C.

IoU(C, E): C = $(100,100,150,150)$, E = $(102,98,148,148)$. Intersection = $(102,100,148,148)$ = $46 \times 48 = 2208$. Area(C) = $2500$, Area(E) = $46 \times 50 = 2300$. Union = $2500 + 2300 - 2208 = 2592$. IoU = $2208/2592 \approx 0.852 > 0.5$. **Suppress E**.

**Final kept detections: {A, C}** — two objects detected.

</details>

---

## Exercise 3: Anchor Box Offsets

**Target time**: 3 minutes

An anchor box has center $(cx, cy) = (32, 32)$ and dimensions $(w, h) = (64, 64)$.

The network predicts offsets $(\Delta cx, \Delta cy, \Delta w, \Delta h) = (0.5, -0.3, 0.2, 0.1)$ using the encoding:

$$\hat{cx} = cx + \Delta cx \cdot w, \quad \hat{cy} = cy + \Delta cy \cdot h$$
$$\hat{w} = w \cdot e^{\Delta w}, \quad \hat{h} = h \cdot e^{\Delta h}$$

**Part 1**: Compute the predicted bounding box center $(\hat{cx}, \hat{cy})$.

**Part 2**: Compute the predicted dimensions $(\hat{w}, \hat{h})$.

**Part 3**: Convert to $(x_1, y_1, x_2, y_2)$ corner format.

<details>
<summary>Solution</summary>

**Part 1**:
- $\hat{cx} = 32 + 0.5 \times 64 = 32 + 32 = 64$
- $\hat{cy} = 32 + (-0.3) \times 64 = 32 - 19.2 = 12.8$

**Part 2**:
- $\hat{w} = 64 \times e^{0.2} = 64 \times 1.2214 \approx 78.17$
- $\hat{h} = 64 \times e^{0.1} = 64 \times 1.1052 \approx 70.73$

**Part 3**:
- $x_1 = 64 - 78.17/2 \approx 24.92$
- $y_1 = 12.8 - 70.73/2 \approx -22.57$
- $x_2 = 64 + 78.17/2 \approx 103.08$
- $y_2 = 12.8 + 70.73/2 \approx 48.17$

Box: $(24.92, -22.57, 103.08, 48.17)$

</details>

---

## Exercise 4: Precision-Recall Computation

**Target time**: 5 minutes

A detector produces 6 detections (sorted by confidence) for an image with 3 ground truth objects. IoU threshold = 0.5.

| Detection | Confidence | IoU with best GT | TP/FP? |
|---|---|---|---|
| D1 | 0.95 | 0.82 | ? |
| D2 | 0.88 | 0.71 | ? |
| D3 | 0.75 | 0.35 | ? |
| D4 | 0.60 | 0.65 | ? |
| D5 | 0.50 | 0.90 (same GT as D1) | ? |
| D6 | 0.30 | 0.55 | ? |

**Part 1**: Label each detection as TP or FP. (A GT object can only be matched once.)

**Part 2**: Compute precision and recall at each detection threshold.

**Part 3**: Compute the Average Precision (AP) using the precision-recall curve.

<details>
<summary>Solution</summary>

**Part 1**:
- D1: IoU = 0.82 > 0.5, unmatched GT → **TP** (matches GT_A)
- D2: IoU = 0.71 > 0.5, unmatched GT → **TP** (matches GT_B)
- D3: IoU = 0.35 < 0.5 → **FP**
- D4: IoU = 0.65 > 0.5, unmatched GT → **TP** (matches GT_C)
- D5: IoU = 0.90 > 0.5, but GT_A already matched → **FP**
- D6: IoU = 0.55 > 0.5, but all GTs matched → **FP**

**Part 2**: (cumulative TP, cumulative FP, P, R with 3 GT objects)

| Det | Cum TP | Cum FP | Precision | Recall |
|---|---|---|---|---|
| D1 | 1 | 0 | 1/1 = 1.00 | 1/3 = 0.33 |
| D2 | 2 | 0 | 2/2 = 1.00 | 2/3 = 0.67 |
| D3 | 2 | 1 | 2/3 = 0.67 | 2/3 = 0.67 |
| D4 | 3 | 1 | 3/4 = 0.75 | 3/3 = 1.00 |
| D5 | 3 | 2 | 3/5 = 0.60 | 3/3 = 1.00 |
| D6 | 3 | 3 | 3/6 = 0.50 | 3/3 = 1.00 |

**Part 3**: AP (11-point interpolation at recall = 0, 0.1, ..., 1.0):
- At R=0: max P = 1.00
- At R=0.1: max P = 1.00
- At R=0.2: max P = 1.00
- At R=0.33: max P = 1.00
- At R=0.67: max P = 1.00 (interpolated max for R≥0.67 is max(0.67, 0.75, 0.60, 0.50) = 0.75)
- At R=1.0: max P = 0.75

Using the all-point interpolation: AP = area under monotone-decreasing precision envelope.

Interpolated precision: At each recall level, take the maximum precision at any recall ≥ that level.
- R=0.33: P=max(1.00, 0.75) = 1.00
- R=0.67: P=max(0.75) = 0.75
- R=1.00: P=0.75

AP = $0.33 \times 1.00 + (0.67-0.33) \times 0.75 + (1.00-0.67) \times 0.75 = 0.33 + 0.255 + 0.2475 = 0.8325$

**AP ≈ 0.83**

</details>

---

## Exercise 5: IoU in Vectorized Form

**Target time**: 4 minutes

Write out the shapes at each step of computing pairwise IoU between $N$ predicted boxes and $M$ ground truth boxes.

Given:
- `pred_boxes`: shape $(N, 4)$ in $(x_1, y_1, x_2, y_2)$ format
- `gt_boxes`: shape $(M, 4)$ in $(x_1, y_1, x_2, y_2)$ format

**Part 1**: What broadcasting operation produces the intersection coordinates? Write the shapes.

**Part 2**: What is the shape of the resulting IoU matrix? What does entry $(i, j)$ represent?

**Part 3**: To match each prediction to the best ground truth, what operation do you perform on the IoU matrix?

<details>
<summary>Solution</summary>

**Part 1**:
```
pred_boxes[:, None, :2]  shape: (N, 1, 2)   ← x1, y1 of predictions
gt_boxes[None, :, :2]    shape: (1, M, 2)   ← x1, y1 of ground truths
torch.max(...)           shape: (N, M, 2)   ← intersection top-left

pred_boxes[:, None, 2:]  shape: (N, 1, 2)   ← x2, y2 of predictions
gt_boxes[None, :, 2:]    shape: (1, M, 2)   ← x2, y2 of ground truths
torch.min(...)           shape: (N, M, 2)   ← intersection bottom-right

intersection_wh = (bottom_right - top_left).clamp(min=0)  shape: (N, M, 2)
intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  shape: (N, M)
```

**Part 2**: IoU matrix shape: $(N, M)$. Entry $(i, j)$ = IoU between prediction $i$ and ground truth $j$.

**Part 3**: `best_gt_per_pred = iou_matrix.argmax(dim=1)` gives shape $(N,)$ — the index of the best-matching ground truth for each prediction.

`best_iou_per_pred = iou_matrix.max(dim=1).values` gives shape $(N,)$ — the best IoU for each prediction (used to determine TP/FP at a given threshold).

</details>

---
