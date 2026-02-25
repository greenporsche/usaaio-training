# Transfer Learning Exercises

**Topic**: Feature extraction, fine-tuning, data augmentation, pretrained models
**Difficulty**: Foundational → Intermediate

---

## Exercise 1: Freeze/Unfreeze Strategy

You are given a pretrained ResNet-18 for a 5-class flower classification task with only 200 training images. The ResNet-18 architecture has:

```
conv1 → bn1 → relu → maxpool →
layer1 (2 blocks) → layer2 (2 blocks) → layer3 (2 blocks) → layer4 (2 blocks) →
avgpool → fc(512, 1000)
```

1. How would you modify the model for 5 classes?
2. Which layers should you freeze?
3. What optimizer and learning rate would you use?
4. Why is feature extraction better than full fine-tuning here?

<details>
<summary>Solution</summary>

**1. Modify for 5 classes**:
```python
model.fc = nn.Linear(512, 5)  # Replace final FC layer
```

**2. Freeze all except the new FC layer**:
```python
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
```

**3. Optimizer**: Adam with `lr=1e-3` for the FC layer only.
```python
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**4. Why feature extraction?**:
- With only 200 images (40 per class), fine-tuning 11M parameters would massively overfit
- The pretrained features (edges, textures, shapes) from ImageNet are highly relevant to flower classification
- Training only 5 × 512 + 5 = 2,565 parameters is feasible with 200 samples
- The feature extraction approach acts like a linear classifier on top of rich pretrained features

**Key insight**: The rule of thumb is: small dataset + similar domain = feature extraction. With only 200 images, you cannot afford to update millions of parameters. The pretrained features are already excellent for natural images.
</details>

---

## Exercise 2: Trainable Parameter Count

For a pretrained ResNet-50 modified for 20-class classification, compute the number of trainable parameters under each strategy:

ResNet-50 has approximately 25.6M total parameters. The final FC layer is `Linear(2048, 1000)`.

| Strategy | What is trainable? | Trainable params |
|---|---|---|
| Feature extraction | ? | ? |
| Fine-tune layer4 + fc | ? | ? |
| Full fine-tuning | ? | ? |

<details>
<summary>Solution</summary>

**Feature extraction**: Only the new FC layer.
- $\text{fc}: 2048 \times 20 + 20 = 40{,}980$
- Trainable: **40,980** (0.16% of total)

**Fine-tune layer4 + fc**: Layer4 in ResNet-50 has 3 bottleneck blocks. Each bottleneck has:
- 3 Conv2d layers + 3 BN layers (+ projection in first block)
- Layer4 total: approximately $2{,}048 \times 512 + 512 \times 512 \times 9 + 512 \times 2048 + \text{BN} + \text{projection} \approx 7.1\text{M}$
- Trainable: **~7.1M + 40,980 ≈ 7.14M** (28% of total)

**Full fine-tuning**: Everything.
- Trainable: Replace fc with $2048 \times 20 + 20 = 40{,}980$
- New total: $25.6\text{M} - 2{,}049{,}000 + 40{,}980 \approx 23.6\text{M}$
- Trainable: **~23.6M** (100%)

| Strategy | Trainable | % of total |
|---|---|---|
| Feature extraction | 40,980 | 0.16% |
| Fine-tune layer4 + fc | ~7.14M | ~28% |
| Full fine-tuning | ~23.6M | 100% |

**Key insight**: Feature extraction trains < 0.2% of parameters — incredibly efficient. Fine-tuning layer4 is a good middle ground (28%). The choice depends on your dataset size and domain similarity.
</details>

---

## Exercise 3: Data Augmentation Analysis

For each augmentation technique, answer: (a) what invariance does it encourage? (b) when would it be harmful?

1. `RandomHorizontalFlip(p=0.5)`
2. `RandomRotation(30)`
3. `ColorJitter(brightness=0.3)`
4. `RandomResizedCrop(224, scale=(0.5, 1.0))`
5. `RandomVerticalFlip(p=0.5)`

<details>
<summary>Solution</summary>

**1. RandomHorizontalFlip**:
- (a) Left-right symmetry. A dog facing left should be classified the same as one facing right.
- (b) **Harmful for**: text recognition (mirrored text is different), medical imaging with laterality (left vs right lung), and any task where orientation is meaningful.

**2. RandomRotation(30)**:
- (a) Rotation invariance within ±30 degrees. Objects in natural images can appear slightly tilted.
- (b) **Harmful for**: digit recognition (6 vs 9 under rotation), satellite imagery where north orientation matters, or when rotation > 10° is unrealistic for the domain.

**3. ColorJitter**:
- (a) Illumination invariance. The same object under different lighting conditions should be recognized.
- (b) **Harmful for**: color-specific classification (red vs green traffic light), skin lesion diagnosis where color is a diagnostic feature, or any task where precise color information is discriminative.

**4. RandomResizedCrop**:
- (a) Scale invariance and translation invariance. Objects can appear at different sizes and positions.
- (b) **Harmful for**: tasks where the object fills the entire image and scale is fixed (e.g., standardized X-rays), or if `scale=0.5` might crop out the object of interest entirely.

**5. RandomVerticalFlip**:
- (a) Top-bottom symmetry.
- (b) **Harmful for**: nearly all natural image tasks! Animals, faces, text, buildings — almost nothing looks the same upside-down. Only useful for aerial/satellite imagery or microscopy where orientation is arbitrary.

**Key insight**: Every augmentation encodes an assumption about the task. The assumption is: "these transformations should not change the label." If that assumption is wrong, the augmentation teaches the model to ignore information that matters. Always think about whether the augmentation is label-preserving for your specific task.
</details>

---

## Exercise 4: ImageNet Normalization

When using pretrained ImageNet models, you must normalize inputs with:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

1. Why these specific values?
2. What happens if you skip this normalization?
3. If your dataset has different statistics (e.g., medical X-rays with mean=[0.5], std=[0.25]), should you use ImageNet stats or your dataset's stats?

<details>
<summary>Solution</summary>

**1.** These are the per-channel mean and standard deviation of the ImageNet training set (1.2M natural images). The pretrained model was trained with inputs normalized to these statistics, so its internal representations expect this distribution.

**2.** Without normalization, the model receives inputs with different statistics than it was trained on. The first layer's weights, biases, and batch norm parameters all expect inputs centered around 0 with unit variance. Feeding raw pixels (mean ~0.45, std ~0.23) is like feeding a different distribution — the model's features will be misaligned and accuracy will drop significantly, possibly to random.

**3.** **Use ImageNet stats**, not your dataset's stats. Here's why:

The pretrained model learned features assuming ImageNet normalization. The first conv layer's weights, and all subsequent batch norm layers, encode this expectation. Using different normalization statistics is equivalent to applying an affine transformation to the input, which shifts ALL features out of their learned distributions.

Even for medical images that look very different from ImageNet, the normalization step should match what the pretrained model expects. The model's internal layers will adapt during fine-tuning, but the input normalization must remain consistent.

**Exception**: If training from scratch (no pretrained weights), normalize with your dataset's own statistics.

**Key insight**: Input normalization is part of the model's "interface contract." Changing it is like passing Fahrenheit to a function that expects Celsius — the computation runs but the results are meaningless.
</details>

---

## Exercise 5: Transfer Learning Debugging

Your transfer learning pipeline achieves 95% training accuracy but only 52% validation accuracy (10 classes, so random is 10%). Identify the problem and propose fixes.

```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# Unfreeze everything
model.fc = nn.Linear(512, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(50):
    model.train()
    for X, y in train_loader:  # 500 training images
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

<details>
<summary>Solution</summary>

**Problem**: Severe overfitting. 95% train vs 52% val (42% gap) with only 500 images.

**Root causes**:
1. **Full fine-tuning with too little data**: 11M parameters with 500 images = massive overfitting risk
2. **Learning rate too high for pretrained layers**: `lr=1e-3` is appropriate for the new head but destroys pretrained features in earlier layers
3. **No augmentation**: With 500 images, the model quickly memorizes the training set
4. **No regularization**: No dropout, weight decay, or early stopping
5. **Training too long**: 50 epochs on 500 images with a powerful model

**Fixes** (apply all):

```python
# 1. Feature extraction first, then gradual unfreezing
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)

# 2. Use appropriate learning rate
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Add data augmentation
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 4. Add early stopping based on validation loss

# 5. Reduce epochs or use learning rate scheduler
```

With these changes, you might achieve 80-90% validation accuracy even with 500 images.

**Key insight**: Transfer learning with small datasets requires discipline: freeze most layers, use aggressive augmentation, add regularization, and monitor the train-val gap carefully. The pretrained features are valuable — don't destroy them with a high learning rate.
</details>
