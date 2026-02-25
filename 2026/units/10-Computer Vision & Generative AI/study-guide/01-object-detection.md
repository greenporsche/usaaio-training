# Object Detection

**Prerequisites**: CNNs, classification, bounding box representation, basic probability
**USAAIO Relevance**: IoU computation, NMS algorithm, and mAP calculation appear as algorithmic questions. Understanding detection as classification + localization is foundational.

---

## Discovery

### The Core Question

Image classification answers "what is in the image?" Object detection answers "what is in the image AND where is it?" This requires simultaneously predicting class labels and bounding box coordinates — turning a single-output problem into a multi-output, variable-count problem.

> Given an image, output a set of bounding boxes $\{(x_1, y_1, x_2, y_2, c, p)\}$ where $(x_1, y_1, x_2, y_2)$ are box coordinates, $c$ is the class, and $p$ is the confidence.

### Historical Context

- **R-CNN** (Girshick et al., 2014): Region proposals + CNN classification. Slow — each region processed independently.
- **Fast R-CNN** (2015): Share CNN computation across regions.
- **Faster R-CNN** (2015): Replace external region proposals with a learned Region Proposal Network (RPN).
- **YOLO** (Redmon et al., 2016): Single-shot detection — predict boxes and classes in one forward pass.
- **SSD** (Liu et al., 2016): Multi-scale single-shot detection with anchor boxes.

### Socratic Warm-Up

1. If you classify each pixel independently, what problems arise for detecting objects?
2. Why can't you just use a fixed-size grid of predictions? (Hint: objects come in different sizes.)
3. How do you decide if a predicted box "matches" a ground truth box?

### Misconception Traps

- **"IoU > 0.5 means a good detection."** — 0.5 is the *minimum* threshold for PASCAL VOC; COCO uses multiple thresholds (0.5, 0.55, ..., 0.95).
- **"NMS just removes duplicate boxes."** — NMS can also suppress valid detections of nearby objects of the same class.
- **"More anchor boxes = better."** — Too many anchors slow training and increase false positives.

---

## Intuition

### Bounding Box Representation

Two common formats:

```
(x1, y1, x2, y2) — corner format      (cx, cy, w, h) — center format
┌──────────┐
│          │ (x2, y2)                        (cx, cy) = center
│          │                                 w, h = width, height
│(x1, y1)  │
└──────────┘

Conversion:
cx = (x1+x2)/2,  cy = (y1+y2)/2,  w = x2-x1,  h = y2-y1
x1 = cx-w/2,     y1 = cy-h/2,     x2 = cx+w/2, y2 = cy+h/2
```

### IoU: Measuring Overlap

Intersection over Union quantifies how well two boxes overlap:

```
Box A:          Box B:          Intersection:      Union:
┌─────┐                        ┌──┐               ┌─────────┐
│     │   ┌─────┐              │  │               │         │
│   ┌─┼───┼─┐   │              └──┘               │         │
│   │ │   │ │   │                                  │         │
└───┼─┘   │ │                                      │         │
    └─────┘                                        └─────────┘

IoU = Area(Intersection) / Area(Union)
    = Area(Intersection) / (Area(A) + Area(B) - Area(Intersection))
```

- IoU = 0: No overlap at all
- IoU = 1: Perfect overlap (identical boxes)
- IoU = 0.5: Typical "good enough" threshold

### Non-Maximum Suppression (NMS)

After a detector produces many overlapping boxes, NMS keeps only the best:

```
Before NMS:                 After NMS:
┌───┐
│0.9│ ┌───┐                 ┌───┐
│   │ │0.7│                 │0.9│
└───┘ │   │                 └───┘
      └───┘
  ┌──┐                        ┌──┐
  │0.8│                       │0.8│
  └──┘                        └──┘
```

### Anchor Boxes

Instead of predicting boxes from scratch, the network predicts *offsets* from pre-defined anchor boxes:

```
Grid cell with 3 anchors:

 ┌─────────────┐
 │  ┌───┐      │  Anchor 1: tall (person)
 │  │   │      │
 │  └───┘      │
 │   ┌────┐    │  Anchor 2: wide (car)
 │   └────┘    │
 │  ┌──┐       │  Anchor 3: square (face)
 │  └──┘       │
 └─────────────┘

Network predicts: (dx, dy, dw, dh, obj_conf, class_probs) per anchor
```

---

## Math

### IoU Computation

Given two axis-aligned boxes $A = (x_1^A, y_1^A, x_2^A, y_2^A)$ and $B = (x_1^B, y_1^B, x_2^B, y_2^B)$:

**Intersection coordinates**:
$$x_1^I = \max(x_1^A, x_1^B), \quad y_1^I = \max(y_1^A, y_1^B)$$
$$x_2^I = \min(x_2^A, x_2^B), \quad y_2^I = \min(y_2^A, y_2^B)$$

**Intersection area**:
$$\text{Area}_I = \max(0, x_2^I - x_1^I) \cdot \max(0, y_2^I - y_1^I)$$

**Union area**:
$$\text{Area}_U = \text{Area}_A + \text{Area}_B - \text{Area}_I$$

**IoU**:
$$\text{IoU}(A, B) = \frac{\text{Area}_I}{\text{Area}_U}$$

### NMS Algorithm

```
Input:  boxes B, scores S, IoU threshold τ
Output: kept boxes D

D = []
while B is not empty:
    i = argmax(S)          # highest confidence box
    D.append(B[i])
    for j in B \ {i}:
        if IoU(B[i], B[j]) > τ:
            remove B[j] from B, S[j] from S
    remove B[i] from B, S[i] from S
return D
```

### Mean Average Precision (mAP)

For each class $c$:
1. Sort all detections by confidence (descending)
2. For each detection, mark as TP if IoU with unmatched ground truth > threshold, else FP
3. Compute precision and recall at each detection
4. $\text{AP}_c = \int_0^1 p(r) \, dr$ (area under the precision-recall curve)
5. $\text{mAP} = \frac{1}{C}\sum_{c=1}^{C} \text{AP}_c$

---

## Code

### IoU in PyTorch

```python
def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    box_a: (N, 4) in (x1, y1, x2, y2) format
    box_b: (M, 4) in (x1, y1, x2, y2) format
    Returns: (N, M) IoU matrix
    """
    # Intersection
    x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])  # (N, M)
    y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # (N,)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])  # (M,)

    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / (union + 1e-6)
```

### NMS in PyTorch

```python
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    boxes: (N, 4)
    scores: (N,)
    Returns: indices of kept boxes
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[i:i+1], boxes[remaining]).squeeze(0)  # (R,)
        mask = ious <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)
```
