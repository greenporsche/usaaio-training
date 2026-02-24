# Problem 12 Variations: Pretrained Models (EXHAUSTIVE)

> Original: ResNet34 — parameter counting, output shapes, layer analysis, transfer learning
> Core Skills: PyTorch model introspection, torchsummary, transfer learning, parameter freezing

---

## CATEGORY A: Different Pretrained Models

### Variation A1: VGG16 Analysis

Run the following code:

```python
from torchvision.models import vgg16
import torch

model = vgg16(weights='DEFAULT')
```

**Part 12.1**: What is the total number of learnable parameters in the model?

**Part 12.2**: Suppose the input has shape (B, 3, 224, 224). What is the shape of the output from `model.features`?

**Part 12.3**: How many convolutional layers are in `model.features`? How many max pooling layers?

**Part 12.4**: Build a model using VGG16 as backbone for 5-class classification. Freeze all parameters in the backbone.

<details>
<summary>Solution A1</summary>

**12.1**:
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: 138,357,544 (approximately 138M parameters)
```

**12.2**:
```python
x = torch.randn(1, 3, 224, 224)
out = model.features(x)
print(out.shape)
# Output: torch.Size([1, 512, 7, 7])
# After 5 max pools: 224 -> 112 -> 56 -> 28 -> 14 -> 7
```

**12.3**:
```python
conv_count = sum(1 for m in model.features if isinstance(m, torch.nn.Conv2d))
pool_count = sum(1 for m in model.features if isinstance(m, torch.nn.MaxPool2d))
print(f"Conv layers: {conv_count}")  # 13
print(f"MaxPool layers: {pool_count}")  # 5
```

**12.4**:
```python
import torch.nn as nn

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = vgg16(weights='DEFAULT')

        # Freeze backbone
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Replace classifier
        self.backbone.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = VGG16Classifier(num_classes=5)
```

</details>

### Variation A2: ResNet18 vs ResNet50

Run the following code:

```python
from torchvision.models import resnet18, resnet50
import torch

model18 = resnet18(weights='DEFAULT')
model50 = resnet50(weights='DEFAULT')
```

**Part 12.1**: Compare the total number of learnable parameters in resnet18 and resnet50.

**Part 12.2**: For input shape (B, 3, 224, 224), what are the output shapes from `layer3` in both models?

**Part 12.3**: What is the key architectural difference between a "BasicBlock" (used in ResNet18) and a "Bottleneck" (used in ResNet50)?

**Part 12.4**: Which model would you choose for a mobile application? Justify your answer.

<details>
<summary>Solution A2</summary>

**12.1**:
```python
params18 = sum(p.numel() for p in model18.parameters())
params50 = sum(p.numel() for p in model50.parameters())
print(f"ResNet18: {params18:,}")  # 11,689,512
print(f"ResNet50: {params50:,}")  # 25,557,032
print(f"Ratio: {params50/params18:.2f}x")  # ~2.2x more parameters
```

**12.2**:
```python
x = torch.randn(1, 3, 224, 224)

out18 = model18.layer3(model18.layer2(model18.layer1(model18.maxpool(model18.relu(model18.bn1(model18.conv1(x)))))))
out50 = model50.layer3(model50.layer2(model50.layer1(model50.maxpool(model50.relu(model50.bn1(model50.conv1(x)))))))

print(f"ResNet18 layer3 output: {out18.shape}")  # [1, 256, 14, 14]
print(f"ResNet50 layer3 output: {out50.shape}")  # [1, 1024, 14, 14]
```

**12.3**:
- **BasicBlock** (ResNet18/34): Two 3×3 convolutions
  - Input → 3×3 Conv → BN → ReLU → 3×3 Conv → BN → + Input → ReLU
  - Total: 2 conv layers per block

- **Bottleneck** (ResNet50/101/152): Three convolutions with 1×1 reducing and expanding
  - Input → 1×1 Conv (reduce channels) → BN → ReLU → 3×3 Conv → BN → ReLU → 1×1 Conv (expand) → BN → + Input → ReLU
  - Total: 3 conv layers per block
  - Uses 1×1 convolutions to reduce computational cost ("bottleneck")

**12.4**:
ResNet18 is better for mobile because:
- ~2.2x fewer parameters (11.7M vs 25.6M)
- Lower memory footprint
- Faster inference
- Still achieves good accuracy (69.8% vs 76.1% on ImageNet)

For extremely constrained devices, consider MobileNet or EfficientNet variants.

</details>

### Variation A3: MobileNetV2 Analysis

Run the following code:

```python
from torchvision.models import mobilenet_v2
import torch

model = mobilenet_v2(weights='DEFAULT')
```

**Part 12.1**: What is the total number of learnable parameters? Compare with ResNet34.

**Part 12.2**: What is the architectural innovation in MobileNetV2 that makes it efficient?

**Part 12.3**: For input (B, 3, 224, 224), what is the shape after `model.features`?

**Part 12.4**: Adapt MobileNetV2 for 100-class classification with frozen backbone.

<details>
<summary>Solution A3</summary>

**12.1**:
```python
params_mobile = sum(p.numel() for p in model.parameters())
print(f"MobileNetV2: {params_mobile:,}")  # 3,504,872

# Compare with ResNet34: 21,797,672
# MobileNetV2 is ~6.2x smaller!
```

**12.2**:
MobileNetV2 innovations:
1. **Depthwise Separable Convolutions**: Split into depthwise (spatial) + pointwise (channel) operations
   - Regular 3×3 conv: k² × C_in × C_out
   - Depthwise separable: k² × C_in + C_in × C_out
   - ~8-9x fewer parameters for 3×3 convs

2. **Inverted Residuals**:
   - Standard: wide → narrow → wide (bottleneck)
   - MobileNetV2: narrow → wide → narrow (inverted bottleneck)
   - Expands channels before depthwise conv

3. **Linear Bottlenecks**: No ReLU after the last pointwise conv (preserves information in narrow layers)

**12.3**:
```python
x = torch.randn(1, 3, 224, 224)
out = model.features(x)
print(out.shape)  # torch.Size([1, 1280, 7, 7])
```

**12.4**:
```python
import torch.nn as nn

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = mobilenet_v2(weights='DEFAULT')

        # Freeze backbone
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Replace classifier (MobileNetV2 uses Sequential classifier)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = MobileNetClassifier(num_classes=100)
```

</details>

### Variation A4: EfficientNet-B0 Analysis

Run the following code:

```python
from torchvision.models import efficientnet_b0
import torch

model = efficientnet_b0(weights='DEFAULT')
```

**Part 12.1**: What is the total number of learnable parameters?

**Part 12.2**: What is the key innovation in EfficientNet's compound scaling?

**Part 12.3**: Examine the first and last blocks of `model.features`. What is the difference in number of channels?

**Part 12.4**: Adapt for binary classification with frozen backbone.

<details>
<summary>Solution A4</summary>

**12.1**:
```python
params = sum(p.numel() for p in model.parameters())
print(f"EfficientNet-B0: {params:,}")  # 5,288,548
```

**12.2**:
EfficientNet compound scaling jointly scales:
- **Width** (w): Number of channels in each layer
- **Depth** (d): Number of layers
- **Resolution** (r): Input image size

Formula: depth = d^φ, width = w^φ, resolution = r^φ
Where φ is determined by grid search and satisfies d × w² × r² ≈ 2

This balanced scaling is more efficient than scaling only one dimension.

**12.3**:
```python
# First block (features[0] is Conv2dNormActivation)
first_conv = model.features[0][0]  # First Conv2d
print(f"First conv: in={first_conv.in_channels}, out={first_conv.out_channels}")
# in=3, out=32

# Last feature block
last_block = model.features[-1][0]  # Last Conv2d in features
print(f"Last conv: in={last_block.in_channels}, out={last_block.out_channels}")
# in=320, out=1280 (expansion to 1280 channels)
```

**12.4**:
```python
import torch.nn as nn

class EfficientNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights='DEFAULT')

        # Freeze backbone
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 1)  # Binary classification
        )

    def forward(self, x):
        return self.backbone(x)

model = EfficientNetBinary()
```

</details>

---

## CATEGORY B: Different Layer Analysis

### Variation B1: Analyzing Specific Layers in ResNet34

Run the following code:

```python
from torchvision.models import resnet34
import torch

model = resnet34(weights='DEFAULT')
```

**Part 12.1**: How many BasicBlock modules are in each of layer1, layer2, layer3, layer4?

**Part 12.2**: What is the total number of parameters in `model.layer1` only?

**Part 12.3**: Calculate the output shape at each layer for input (1, 3, 224, 224).

**Part 12.4**: What is the receptive field after layer2?

<details>
<summary>Solution B1</summary>

**12.1**:
```python
print(f"layer1: {len(model.layer1)} blocks")  # 3
print(f"layer2: {len(model.layer2)} blocks")  # 4
print(f"layer3: {len(model.layer3)} blocks")  # 6
print(f"layer4: {len(model.layer4)} blocks")  # 3
# Total: 3+4+6+3 = 16 blocks = 34 layers (+ initial conv and fc)
```

**12.2**:
```python
layer1_params = sum(p.numel() for p in model.layer1.parameters())
print(f"layer1 parameters: {layer1_params:,}")  # 147,968

# Breakdown: Each BasicBlock has 2 conv layers (64→64)
# Per block: 64*64*3*3 + 64*64*3*3 = 73,728 (with BN: ~74k)
# 3 blocks × ~49k per block ≈ 148k
```

**12.3**:
```python
x = torch.randn(1, 3, 224, 224)

# Initial layers
x1 = model.conv1(x)          # (1, 64, 112, 112) - stride 2, 7×7 conv
x2 = model.bn1(x1)
x3 = model.relu(x2)
x4 = model.maxpool(x3)       # (1, 64, 56, 56) - stride 2

# Main layers
l1 = model.layer1(x4)        # (1, 64, 56, 56) - same spatial
l2 = model.layer2(l1)        # (1, 128, 28, 28) - stride 2
l3 = model.layer3(l2)        # (1, 256, 14, 14) - stride 2
l4 = model.layer4(l3)        # (1, 512, 7, 7) - stride 2

print(f"After conv1: {x1.shape}")
print(f"After maxpool: {x4.shape}")
print(f"After layer1: {l1.shape}")
print(f"After layer2: {l2.shape}")
print(f"After layer3: {l3.shape}")
print(f"After layer4: {l4.shape}")
```

**12.4**:
Receptive field calculation:
- conv1 (7×7, stride 2): RF = 7
- maxpool (3×3, stride 2): RF = 7 + (3-1)×2 = 11
- layer1 (3 blocks, each 2×(3×3)): RF = 11 + 6×(3-1)×1 = 23
- layer2 (4 blocks, stride 2 on first):
  - First block: RF = 23 + 2×2 = 27, then RF = 27 + 2×2 = 31
  - Remaining 3 blocks: RF = 31 + 6×(3-1)×2 = 31 + 24 = 55

Approximate receptive field after layer2: **~55 pixels**

</details>

### Variation B2: Parameter Distribution Analysis

Using ResNet34:

**Part 12.1**: Calculate the percentage of parameters in each major component (conv1, layer1-4, fc).

**Part 12.2**: Which layer has the most parameters? Why?

**Part 12.3**: What percentage of parameters are in batch normalization layers?

<details>
<summary>Solution B2</summary>

**12.1**:
```python
from torchvision.models import resnet34
model = resnet34(weights='DEFAULT')

components = {
    'conv1': sum(p.numel() for p in model.conv1.parameters()),
    'bn1': sum(p.numel() for p in model.bn1.parameters()),
    'layer1': sum(p.numel() for p in model.layer1.parameters()),
    'layer2': sum(p.numel() for p in model.layer2.parameters()),
    'layer3': sum(p.numel() for p in model.layer3.parameters()),
    'layer4': sum(p.numel() for p in model.layer4.parameters()),
    'fc': sum(p.numel() for p in model.fc.parameters()),
}

total = sum(components.values())
for name, params in components.items():
    print(f"{name}: {params:,} ({100*params/total:.2f}%)")

# conv1: 9,408 (0.04%)
# bn1: 128 (0.00%)
# layer1: 147,968 (0.68%)
# layer2: 525,568 (2.41%)
# layer3: 2,099,712 (9.64%)
# layer4: 8,393,728 (38.52%)
# fc: 513,000 (2.35%)
```

**12.2**:
layer4 has the most parameters (~38.5%) because:
- Largest number of channels (512)
- Conv weight size = C_in × C_out × k² = 512 × 512 × 9 = 2,359,296 per conv
- 3 BasicBlocks × 2 convs = 6 convolutions
- Total: ~8.4M parameters

Parameters scale quadratically with channel count.

**12.3**:
```python
import torch.nn as nn

bn_params = sum(
    p.numel() for m in model.modules()
    if isinstance(m, nn.BatchNorm2d)
    for p in m.parameters()
)
total_params = sum(p.numel() for p in model.parameters())

print(f"BN parameters: {bn_params:,}")  # 10,048
print(f"Percentage: {100*bn_params/total_params:.2f}%")  # ~0.05%

# BN has 2 params per channel (gamma, beta), very small contribution
```

</details>

### Variation B3: Convolution Details

For ResNet34's `model.layer2[0]`:

**Part 12.1**: List all convolutional layers and their parameters (kernel size, stride, padding, in/out channels).

**Part 12.2**: Calculate the number of parameters manually for all convolutions.

**Part 12.3**: Why is there a `downsample` module in layer2[0]?

<details>
<summary>Solution B3</summary>

**12.1**:
```python
from torchvision.models import resnet34
model = resnet34(weights='DEFAULT')

block = model.layer2[0]
print(block)

# BasicBlock(
#   (conv1): Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
#   (bn1): BatchNorm2d(128)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#   (bn2): BatchNorm2d(128)
#   (downsample): Sequential(
#     (0): Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
#     (1): BatchNorm2d(128)
#   )
# )
```

| Layer | Kernel | Stride | Padding | In | Out |
|-------|--------|--------|---------|-----|-----|
| conv1 | 3×3 | 2 | 1 | 64 | 128 |
| conv2 | 3×3 | 1 | 1 | 128 | 128 |
| downsample[0] | 1×1 | 2 | 0 | 64 | 128 |

**12.2**:
```python
# Manual calculation (no bias in ResNet convolutions)
conv1_params = 64 * 128 * 3 * 3  # = 73,728
conv2_params = 128 * 128 * 3 * 3  # = 147,456
downsample_params = 64 * 128 * 1 * 1  # = 8,192

total_conv_params = conv1_params + conv2_params + downsample_params
print(f"Total conv params: {total_conv_params:,}")  # 229,376

# Verification
actual = sum(
    p.numel() for m in block.modules()
    if isinstance(m, torch.nn.Conv2d)
    for p in m.parameters()
)
print(f"Actual: {actual:,}")  # 229,376 ✓
```

**12.3**:
The `downsample` module exists because:
- Input to layer2[0] is (B, 64, 56, 56)
- Output must be (B, 128, 28, 28)
- The skip connection needs to match dimensions
- Without downsample: input + conv_output would have mismatched shapes

The downsample performs:
1. Channel expansion: 64 → 128 (via 1×1 conv)
2. Spatial reduction: 56×56 → 28×28 (via stride 2)

This allows: `output = conv_path(x) + downsample(x)`

</details>

---

## CATEGORY C: Different Transfer Learning Scenarios

### Variation C1: Feature Extraction Only

Build a model for 20-class classification using ResNet34 where:
- The backbone is completely frozen
- Only a new classification head is trained
- Use global average pooling before the classifier

<details>
<summary>Solution C1</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        # Load pretrained ResNet34
        backbone = resnet34(weights='DEFAULT')

        # Remove the original fc layer
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # Freeze ALL parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # New classification head
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Extract features (no gradients computed)
        with torch.no_grad():
            x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Verify frozen parameters
model = FeatureExtractor(num_classes=20)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Only classifier (512*20 + 20 = 10,260) is trainable
```

</details>

### Variation C2: Fine-tuning Later Layers

Build a model where:
- layer1 and layer2 are frozen
- layer3, layer4, and classifier are trainable

**Part 12.1**: Implement this selective freezing.

**Part 12.2**: How many parameters are trainable vs frozen?

<details>
<summary>Solution C2</summary>

**12.1**:
```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class SelectiveFineTuning(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet34(weights='DEFAULT')

        # Freeze early layers
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False

        # Replace fc for new task
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

model = SelectiveFineTuning(num_classes=10)
```

**12.2**:
```python
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = frozen + trainable

print(f"Frozen: {frozen:,} ({100*frozen/total:.2f}%)")
print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# Frozen: ~683,072 (conv1 + bn1 + layer1 + layer2)
# Trainable: ~21,114,890 (layer3 + layer4 + fc)
```

</details>

### Variation C3: Multi-Head Classification

Build a model with two classification heads:
- Head A: 10 classes (trained)
- Head B: 5 classes (trained)
- Shared backbone (frozen)

<details>
<summary>Solution C3</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class MultiHeadClassifier(nn.Module):
    def __init__(self, num_classes_a=10, num_classes_b=5):
        super().__init__()

        backbone = resnet34(weights='DEFAULT')

        # Extract feature layers
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Two classification heads
        self.head_a = nn.Linear(512, num_classes_a)
        self.head_b = nn.Linear(512, num_classes_b)

    def forward(self, x):
        # Shared feature extraction
        with torch.no_grad():
            features = self.features(x)

        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        # Two outputs
        out_a = self.head_a(features)
        out_b = self.head_b(features)

        return out_a, out_b

# Test
model = MultiHeadClassifier(num_classes_a=10, num_classes_b=5)
x = torch.randn(4, 3, 224, 224)
out_a, out_b = model(x)
print(f"Head A output: {out_a.shape}")  # [4, 10]
print(f"Head B output: {out_b.shape}")  # [4, 5]
```

</details>

### Variation C4: Gradual Unfreezing

Implement a training schedule that:
1. First epoch: only train classifier
2. Epochs 2-3: unfreeze layer4
3. Epochs 4-5: unfreeze layer3
4. Epochs 6+: unfreeze all

<details>
<summary>Solution C4</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class GradualUnfreezing:
    def __init__(self, num_classes=10):
        self.model = resnet34(weights='DEFAULT')
        self.model.fc = nn.Linear(512, num_classes)

        # Initially freeze everything except fc
        self._freeze_all_except_fc()

    def _freeze_all_except_fc(self):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_layer(self, layer_name):
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True

    def update_for_epoch(self, epoch):
        if epoch == 2:
            print("Unfreezing layer4")
            self.unfreeze_layer('layer4')
        elif epoch == 4:
            print("Unfreezing layer3")
            self.unfreeze_layer('layer3')
        elif epoch == 6:
            print("Unfreezing all layers")
            for param in self.model.parameters():
                param.requires_grad = True

    def count_trainable(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# Demonstration
trainer = GradualUnfreezing(num_classes=10)

for epoch in range(1, 8):
    trainer.update_for_epoch(epoch)
    trainable = trainer.count_trainable()
    print(f"Epoch {epoch}: {trainable:,} trainable parameters")

# Epoch 1: 5,130 (just fc)
# Epoch 2: 8,398,858 (layer4 + fc)
# Epoch 4: 10,498,570 (layer3 + layer4 + fc)
# Epoch 6: 21,797,672 (all)
```

</details>

---

## CATEGORY D: Output Shape Computations

### Variation D1: Intermediate Activations

For ResNet34 with input (B, 3, 224, 224):

**Part 12.1**: What is the shape after `model.conv1`?

**Part 12.2**: What is the shape after `model.maxpool`?

**Part 12.3**: What is the shape after `model.avgpool`?

**Part 12.4**: Trace all spatial dimension changes through the network.

<details>
<summary>Solution D1</summary>

```python
import torch
from torchvision.models import resnet34

model = resnet34(weights='DEFAULT')
x = torch.randn(2, 3, 224, 224)

# Trace through network
shapes = {}
shapes['input'] = x.shape

x = model.conv1(x)    # 7x7, stride 2, padding 3
shapes['conv1'] = x.shape  # [2, 64, 112, 112]

x = model.bn1(x)
x = model.relu(x)

x = model.maxpool(x)  # 3x3, stride 2, padding 1
shapes['maxpool'] = x.shape  # [2, 64, 56, 56]

x = model.layer1(x)   # No downsampling
shapes['layer1'] = x.shape  # [2, 64, 56, 56]

x = model.layer2(x)   # First block: stride 2
shapes['layer2'] = x.shape  # [2, 128, 28, 28]

x = model.layer3(x)   # First block: stride 2
shapes['layer3'] = x.shape  # [2, 256, 14, 14]

x = model.layer4(x)   # First block: stride 2
shapes['layer4'] = x.shape  # [2, 512, 7, 7]

x = model.avgpool(x)  # AdaptiveAvgPool2d(1,1)
shapes['avgpool'] = x.shape  # [2, 512, 1, 1]

x = torch.flatten(x, 1)
shapes['flatten'] = x.shape  # [2, 512]

x = model.fc(x)
shapes['fc'] = x.shape  # [2, 1000]

for name, shape in shapes.items():
    print(f"{name}: {list(shape)}")
```

**12.4**: Spatial dimension trace:
| Stage | Spatial Size | Operation |
|-------|--------------|-----------|
| Input | 224×224 | - |
| conv1 | 112×112 | 7×7, stride 2 |
| maxpool | 56×56 | 3×3, stride 2 |
| layer1 | 56×56 | No change |
| layer2 | 28×28 | stride 2 |
| layer3 | 14×14 | stride 2 |
| layer4 | 7×7 | stride 2 |
| avgpool | 1×1 | Global pool |

</details>

### Variation D2: Non-Standard Input Sizes

For ResNet34:

**Part 12.1**: What happens with input (B, 3, 128, 128)?

**Part 12.2**: What is the minimum valid input size?

**Part 12.3**: Can you use (B, 3, 300, 400) rectangular inputs?

<details>
<summary>Solution D2</summary>

**12.1**:
```python
import torch
from torchvision.models import resnet34

model = resnet34(weights='DEFAULT')
x = torch.randn(1, 3, 128, 128)
out = model(x)
print(f"Output shape: {out.shape}")  # [1, 1000] - works!

# Intermediate sizes:
# conv1: 64×64 (128/2)
# maxpool: 32×32
# layer1: 32×32
# layer2: 16×16
# layer3: 8×8
# layer4: 4×4
# avgpool: 1×1 (adaptive)
```

**12.2**:
```python
# Minimum size: must survive all downsampling
# 224 → 112 → 56 → 56 → 28 → 14 → 7
# Minimum to get non-zero output at layer4: at least 1×1
# Working backwards: 1 → 2 → 4 → 8 → 16 → 32 → 64
# Minimum practical: ~32×32 (to avoid degenerate feature maps)

x_min = torch.randn(1, 3, 32, 32)
try:
    out = model(x_min)
    print(f"32×32 works: {out.shape}")  # Works!
except:
    print("32×32 fails")

x_tiny = torch.randn(1, 3, 16, 16)
try:
    out = model(x_tiny)
    print(f"16×16 works: {out.shape}")  # Still works with adaptive pooling!
except:
    print("16×16 fails")
```

**12.3**:
```python
x_rect = torch.randn(1, 3, 300, 400)
out = model(x_rect)
print(f"Rectangular input: {out.shape}")  # [1, 1000]

# Intermediate shapes:
# conv1: [1, 64, 150, 200]
# maxpool: [1, 64, 75, 100]
# ...
# layer4: [1, 512, 10, 13] (approximately)
# avgpool: [1, 512, 1, 1] (adaptive handles non-square)
# Output: [1, 1000]

# Yes, rectangular inputs work because AdaptiveAvgPool2d(1,1)
# pools any spatial dimensions to 1×1
```

</details>

### Variation D3: Batch Size Effects

**Part 12.1**: Does batch size affect the number of parameters?

**Part 12.2**: How does batch size affect memory usage during training?

**Part 12.3**: Calculate approximate memory for batch sizes 1, 8, 32, 128.

<details>
<summary>Solution D3</summary>

**12.1**:
No, batch size does NOT affect the number of parameters.
```python
from torchvision.models import resnet34
model = resnet34()

params = sum(p.numel() for p in model.parameters())
print(f"Parameters (always): {params:,}")  # 21,797,672
```

**12.2**:
Batch size affects memory through:
1. **Input tensor**: B × C × H × W
2. **Intermediate activations**: scales with B (for backprop)
3. **Gradients**: same size as activations
4. **Optimizer states** (Adam): 2x parameter memory (doesn't scale with B)

**12.3**:
```python
import torch
from torchvision.models import resnet34

def estimate_memory_mb(batch_size, input_size=224):
    # Parameter memory (fixed)
    param_memory = 21_797_672 * 4  # float32 = 4 bytes

    # Rough activation memory per sample
    # Approximate: sum of all intermediate tensor sizes
    activation_per_sample = (
        3*224*224 +           # input
        64*112*112 +          # after conv1
        64*56*56 +            # after maxpool
        64*56*56 +            # layer1
        128*28*28 +           # layer2
        256*14*14 +           # layer3
        512*7*7               # layer4
    ) * 4  # float32

    # During training: activations + gradients ≈ 2x
    training_activations = batch_size * activation_per_sample * 2

    total_bytes = param_memory + training_activations
    return total_bytes / (1024**2)

for bs in [1, 8, 32, 128]:
    mem = estimate_memory_mb(bs)
    print(f"Batch size {bs:3d}: ~{mem:.0f} MB")

# Batch size   1: ~95 MB
# Batch size   8: ~152 MB
# Batch size  32: ~325 MB
# Batch size 128: ~1015 MB
```

Note: Actual memory usage varies with PyTorch version, CUDA operations, and other factors.

</details>

---

## CATEGORY E: Different Model Modifications

### Variation E1: Adding Dropout

Modify ResNet34 to add dropout after each layer:

**Part 12.1**: Add dropout (p=0.5) after layer3 and layer4.

**Part 12.2**: Why might this help with overfitting?

<details>
<summary>Solution E1</summary>

**12.1**:
```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class ResNet34WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_p=0.5):
        super().__init__()

        backbone = resnet34(weights='DEFAULT')

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.dropout3 = nn.Dropout2d(p=dropout_p)  # Spatial dropout
        self.layer4 = backbone.layer4
        self.dropout4 = nn.Dropout2d(p=dropout_p)

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout3(x)  # Dropout after layer3
        x = self.layer4(x)
        x = self.dropout4(x)  # Dropout after layer4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = ResNet34WithDropout(num_classes=10, dropout_p=0.3)
```

**12.2**:
Dropout helps with overfitting by:
1. **Regularization**: Forces network to learn redundant representations
2. **Ensemble effect**: Each forward pass uses a different "sub-network"
3. **Prevents co-adaptation**: Neurons can't rely on specific other neurons
4. **Spatial dropout (Dropout2d)**: Drops entire feature maps, more appropriate for CNNs

</details>

### Variation E2: Replacing Activation Functions

Modify ResNet34 to use GELU instead of ReLU:

<details>
<summary>Solution E2</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

def replace_relu_with_gelu(model):
    """Recursively replace all ReLU with GELU"""
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)
    return model

# Load and modify
model = resnet34(weights='DEFAULT')
model = replace_relu_with_gelu(model)

# Verify
relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
gelu_count = sum(1 for m in model.modules() if isinstance(m, nn.GELU))
print(f"ReLU: {relu_count}, GELU: {gelu_count}")  # ReLU: 0, GELU: 17

# Test
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(f"Output shape: {out.shape}")  # [1, 1000]
```

Note: GELU (Gaussian Error Linear Unit) is smoother than ReLU and used in models like BERT and GPT. It may improve performance but increases computation slightly.

</details>

### Variation E3: Adding Squeeze-and-Excitation (SE) Blocks

Add SE blocks to ResNet34's layer4:

<details>
<summary>Solution E3</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class ResNet34SE(nn.Module):
    """ResNet34 with SE blocks in layer4"""
    def __init__(self, num_classes=1000):
        super().__init__()

        backbone = resnet34(weights='DEFAULT')

        # Copy all layers except layer4 and fc
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # layer4 with SE blocks
        self.layer4 = nn.Sequential()
        for i, block in enumerate(backbone.layer4):
            self.layer4.add_module(f'block{i}', block)
            self.layer4.add_module(f'se{i}', SEBlock(512))

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Test
model = ResNet34SE(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"Output: {out.shape}")  # [2, 10]

# New parameters from SE blocks
se_params = sum(
    p.numel() for m in model.layer4.modules()
    if isinstance(m, SEBlock)
    for p in m.parameters()
)
print(f"SE block params: {se_params:,}")  # 4,128 (3 blocks × (512×32 + 32×512))
```

</details>

---

## CATEGORY F: Proof and Theory Questions

### Variation F1: Why Residual Connections Help

**Part 12.1**: Mathematically, what is the gradient flow through a residual block?

**Part 12.2**: Why does this help with the vanishing gradient problem?

**Part 12.3**: What is the identity mapping hypothesis?

<details>
<summary>Solution F1</summary>

**12.1**:
For a residual block: y = F(x) + x

During backpropagation:
```
∂L/∂x = ∂L/∂y · ∂y/∂x
      = ∂L/∂y · (∂F(x)/∂x + 1)
      = ∂L/∂y · ∂F(x)/∂x + ∂L/∂y
```

The gradient has two terms:
1. `∂L/∂y · ∂F(x)/∂x` - gradient through the residual function
2. `∂L/∂y` - **direct gradient path** (identity shortcut)

**12.2**:
Without residual connections (plain network):
- Gradient: ∂L/∂x = ∂L/∂y · ∂F(x)/∂x
- If |∂F(x)/∂x| < 1, gradients vanish exponentially with depth
- If |∂F(x)/∂x| > 1, gradients explode

With residual connections:
- The "+1" term ensures gradient ≥ ∂L/∂y regardless of F(x)
- Gradients can flow directly from loss to early layers
- Even if F(x) contributes little, the identity path preserves gradient magnitude

**12.3**:
Identity mapping hypothesis (He et al., 2016):
- It's easier to learn F(x) = 0 (identity) than to learn F(x) = x in a plain network
- If identity is optimal, the network just learns F(x) ≈ 0
- This is why ResNets can be very deep (100+ layers) without degradation
- Pre-activation ResNets make this even cleaner: y = x + F(BN(ReLU(x)))

</details>

### Variation F2: BatchNorm Analysis

**Part 12.1**: What does BatchNorm compute during training vs inference?

**Part 12.2**: Why does BatchNorm have learnable parameters?

**Part 12.3**: How does BatchNorm interact with frozen layers during transfer learning?

<details>
<summary>Solution F2</summary>

**12.1**:
During **training**:
```python
# For input x with shape (B, C, H, W)
mean = x.mean(dim=(0, 2, 3))  # (C,) - mean over batch and spatial
var = x.var(dim=(0, 2, 3))    # (C,) - variance over batch and spatial
x_norm = (x - mean) / sqrt(var + eps)
out = gamma * x_norm + beta   # gamma, beta are learnable (C,)

# Update running statistics
running_mean = momentum * running_mean + (1-momentum) * mean
running_var = momentum * running_var + (1-momentum) * var
```

During **inference**:
```python
# Use running statistics (fixed)
x_norm = (x - running_mean) / sqrt(running_var + eps)
out = gamma * x_norm + beta
```

**12.2**:
Learnable parameters (gamma, beta) allow the network to:
1. **Undo normalization** if needed: gamma=sqrt(var), beta=mean recovers original
2. **Learn optimal scale** for each channel
3. **Learn optimal shift** for each channel
4. Without them, every layer would have zero mean, unit variance (too restrictive)

**12.3**:
When freezing layers in transfer learning:
```python
# Freeze all parameters including BN's gamma/beta
for param in layer.parameters():
    param.requires_grad = False

# IMPORTANT: Also set BN to eval mode
layer.eval()  # Uses running_mean/running_var, not batch statistics
```

If you don't set eval mode, BN still updates running statistics based on new data, which can hurt performance on the original task.

</details>

### Variation F3: Effective Depth vs Actual Depth

**Part 12.1**: What is the effective depth of ResNet50 according to Veit et al.?

**Part 12.2**: Why do most gradient paths skip many layers?

<details>
<summary>Solution F3</summary>

**12.1**:
Veit et al. (2016) "Residual Networks Behave Like Ensembles of Relatively Shallow Networks":
- ResNet with n blocks has 2^n possible paths
- But most paths are short (most skip connections are taken)
- Effective depth ≈ O(√n) instead of O(n)
- ResNet50's effective depth is roughly 10-15 layers, not 50

**12.2**:
During forward pass, each residual block: y = F(x) + x

If F(x) ≈ 0 (easy to learn), the signal primarily flows through identity:
- Most paths take the shortcut (+x) more often than the residual (F(x))
- Gradient distribution: shorter paths contribute more to gradient flow
- Long paths through many F blocks have diminishing contribution

This is why:
1. Removing individual layers doesn't hurt ResNets much
2. ResNets behave like ensembles of shallow networks
3. You can train very deep ResNets without special techniques

</details>

---

## CATEGORY G: Coding Implementations

### Variation G1: Model Summary from Scratch

Implement a function that prints model summary without using torchsummary:

```python
def model_summary(model, input_shape=(3, 224, 224)):
    """
    Print model summary including:
    - Layer name
    - Layer type
    - Output shape
    - Number of parameters
    """
    pass
```

<details>
<summary>Solution G1</summary>

```python
import torch
import torch.nn as nn

def model_summary(model, input_shape=(3, 224, 224), batch_size=2):
    """Print model summary with layer details"""

    device = next(model.parameters()).device
    x = torch.randn(batch_size, *input_shape).to(device)

    summary = []
    hooks = []

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__

        if isinstance(output, (tuple, list)):
            output_shape = [list(o.shape) for o in output]
        else:
            output_shape = list(output.shape)

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

        summary.append({
            'name': class_name,
            'output_shape': output_shape,
            'params': params,
            'trainable': trainable
        })

    # Register hooks for all modules
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    model.eval()
    with torch.no_grad():
        model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print summary
    print("=" * 80)
    print(f"{'Layer':<25} {'Output Shape':<25} {'Params':>12} {'Trainable':>12}")
    print("=" * 80)

    total_params = 0
    total_trainable = 0

    for layer in summary:
        shape_str = str(layer['output_shape'])
        if len(shape_str) > 24:
            shape_str = shape_str[:21] + "..."

        print(f"{layer['name']:<25} {shape_str:<25} {layer['params']:>12,} {layer['trainable']:>12,}")
        total_params += layer['params']
        total_trainable += layer['trainable']

    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable:,}")
    print(f"Non-trainable params: {total_params - total_trainable:,}")

    return summary

# Test
from torchvision.models import resnet34
model = resnet34(weights='DEFAULT')
model_summary(model)
```

</details>

### Variation G2: Parameter Freezing Utility

Implement flexible parameter freezing:

```python
def freeze_parameters(model, patterns=None, layers=None, freeze=True):
    """
    Freeze/unfreeze parameters matching patterns or layer names.

    Args:
        model: PyTorch model
        patterns: List of regex patterns to match parameter names
        layers: List of layer names to freeze
        freeze: True to freeze, False to unfreeze
    """
    pass
```

<details>
<summary>Solution G2</summary>

```python
import re
import torch.nn as nn

def freeze_parameters(model, patterns=None, layers=None, freeze=True):
    """
    Freeze or unfreeze parameters by pattern or layer name.

    Examples:
        freeze_parameters(model, patterns=[r'layer[12]'])  # Freeze layer1, layer2
        freeze_parameters(model, layers=['conv1', 'bn1'])  # Freeze specific layers
        freeze_parameters(model, patterns=[r'.*'], freeze=False)  # Unfreeze all
    """
    frozen_count = 0
    total_params = 0

    # Collect layer names if specified
    layer_params = set()
    if layers:
        for name, module in model.named_modules():
            if name in layers:
                for param in module.parameters():
                    layer_params.add(id(param))

    for name, param in model.named_parameters():
        should_modify = False

        # Check patterns
        if patterns:
            for pattern in patterns:
                if re.search(pattern, name):
                    should_modify = True
                    break

        # Check layer membership
        if layers and id(param) in layer_params:
            should_modify = True

        if should_modify:
            param.requires_grad = not freeze
            if freeze:
                frozen_count += param.numel()

        total_params += param.numel()

    action = "Frozen" if freeze else "Unfrozen"
    print(f"{action} {frozen_count:,} parameters ({100*frozen_count/total_params:.1f}%)")

    return model

# Test
from torchvision.models import resnet34
model = resnet34(weights='DEFAULT')

# Freeze early layers
print("Freezing conv1 and layer1:")
freeze_parameters(model, patterns=[r'^conv1', r'^bn1', r'^layer1'])

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}")
```

</details>

### Variation G3: Feature Extraction at Multiple Scales

Implement a model that extracts features at multiple resolutions:

```python
class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features from multiple layers of a pretrained model.
    Returns features at different spatial scales.
    """
    pass
```

<details>
<summary>Solution G3</summary>

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple spatial scales from ResNet.
    Useful for:
    - Object detection (FPN-style)
    - Semantic segmentation
    - Dense prediction tasks
    """
    def __init__(self, pretrained=True, return_layers=None):
        super().__init__()

        backbone = resnet34(weights='DEFAULT' if pretrained else None)

        # Default: return features from each major layer
        if return_layers is None:
            return_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        self.return_layers = return_layers

        # Build feature extractor
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Channel dimensions for each layer
        self.out_channels = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }

    def forward(self, x):
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.return_layers:
            features['layer1'] = x

        x = self.layer2(x)
        if 'layer2' in self.return_layers:
            features['layer2'] = x

        x = self.layer3(x)
        if 'layer3' in self.return_layers:
            features['layer3'] = x

        x = self.layer4(x)
        if 'layer4' in self.return_layers:
            features['layer4'] = x

        return features

    def get_output_channels(self):
        return {k: self.out_channels[k] for k in self.return_layers}

# Test
extractor = MultiScaleFeatureExtractor(return_layers=['layer2', 'layer3', 'layer4'])
x = torch.randn(2, 3, 224, 224)
features = extractor(x)

print("Multi-scale features:")
for name, feat in features.items():
    print(f"  {name}: {list(feat.shape)}")

# layer2: [2, 128, 28, 28]
# layer3: [2, 256, 14, 14]
# layer4: [2, 512, 7, 7]
```

</details>

### Variation G4: Knowledge Distillation Setup

Implement teacher-student model setup:

```python
class DistillationTrainer:
    """
    Set up knowledge distillation from ResNet50 (teacher) to ResNet18 (student).
    """
    pass
```

<details>
<summary>Solution G4</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

class DistillationTrainer:
    """
    Knowledge distillation: transfer knowledge from large model to small model.

    Loss = α * CE(student_logits, labels) + (1-α) * KL(soft_student, soft_teacher)

    Where soft distributions use temperature T > 1 to soften probabilities.
    """
    def __init__(self, num_classes=1000, temperature=4.0, alpha=0.3):
        self.temperature = temperature
        self.alpha = alpha  # Weight for hard labels (CE loss)

        # Teacher: larger model, frozen
        self.teacher = resnet50(weights='DEFAULT')
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Student: smaller model, trainable
        self.student = resnet18(weights=None)

        # Adjust for custom number of classes
        if num_classes != 1000:
            self.teacher.fc = nn.Linear(2048, num_classes)
            self.student.fc = nn.Linear(512, num_classes)

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute combined distillation loss.
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft label loss (KL divergence with temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        # Scale soft loss by T^2 (standard practice)
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss

    def train_step(self, images, labels):
        """Single training step."""
        # Teacher inference (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(images)

        # Student forward
        student_logits = self.student(images)

        # Compute loss
        loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, labels
        )

        return loss, {
            'total_loss': loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item()
        }

# Test
trainer = DistillationTrainer(num_classes=100, temperature=4.0, alpha=0.3)

# Simulate a batch
images = torch.randn(8, 3, 224, 224)
labels = torch.randint(0, 100, (8,))

loss, metrics = trainer.train_step(images, labels)
print(f"Total loss: {metrics['total_loss']:.4f}")
print(f"Hard loss (CE): {metrics['hard_loss']:.4f}")
print(f"Soft loss (KL): {metrics['soft_loss']:.4f}")

# Parameter comparison
teacher_params = sum(p.numel() for p in trainer.teacher.parameters())
student_params = sum(p.numel() for p in trainer.student.parameters())
print(f"\nTeacher (ResNet50): {teacher_params:,}")
print(f"Student (ResNet18): {student_params:,}")
print(f"Compression: {teacher_params/student_params:.1f}x")
```

</details>

### Variation G5: Model Profiling

Implement inference profiling:

```python
def profile_model(model, input_shape, num_runs=100, warmup=10):
    """
    Profile model inference time and memory usage.
    """
    pass
```

<details>
<summary>Solution G5</summary>

```python
import torch
import time
import numpy as np

def profile_model(model, input_shape=(1, 3, 224, 224), num_runs=100, warmup=10, device='cpu'):
    """
    Profile model performance.

    Returns:
        dict with timing statistics and memory info
    """
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_shape).to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())

    # Memory usage (approximate)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    results = {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'median_time_ms': np.median(times),
        'throughput_fps': 1000 / np.mean(times) * input_shape[0],
        'total_params': total_params,
        'param_memory_mb': param_memory / (1024**2),
    }

    if device == 'cuda':
        results['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()

    return results

def compare_models(models_dict, input_shape=(1, 3, 224, 224), device='cpu'):
    """Compare multiple models"""
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'Time (ms)':>12} {'Throughput':>12}")
    print("=" * 70)

    for name, model in models_dict.items():
        results = profile_model(model, input_shape, device=device)
        print(f"{name:<20} {results['total_params']:>12,} {results['mean_time_ms']:>10.2f}ms {results['throughput_fps']:>9.1f} fps")

# Test
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2

models = {
    'ResNet18': resnet18(weights='DEFAULT'),
    'ResNet34': resnet34(weights='DEFAULT'),
    'ResNet50': resnet50(weights='DEFAULT'),
    'MobileNetV2': mobilenet_v2(weights='DEFAULT'),
}

compare_models(models, device='cpu')
```

</details>

---

## CATEGORY H: Edge Cases and Debugging

### Variation H1: Handling Pretrained Weight Mismatches

**Part 12.1**: What happens when you try to load pretrained weights for a model with a different number of classes?

**Part 12.2**: Implement safe weight loading that handles mismatches.

<details>
<summary>Solution H1</summary>

**12.1**:
```python
from torchvision.models import resnet34
import torch.nn as nn

# This will fail if trying to load 1000-class weights into 10-class model
model = resnet34(weights='DEFAULT')  # Has fc for 1000 classes
model.fc = nn.Linear(512, 10)  # Now expects 10 classes

# If you try to reload pretrained weights:
# state_dict = torch.load('resnet34_pretrained.pth')
# model.load_state_dict(state_dict)  # ERROR! fc shape mismatch
```

**12.2**:
```python
import torch
import torch.nn as nn

def load_pretrained_partial(model, pretrained_path, strict=False):
    """
    Load pretrained weights, ignoring mismatched layers.
    """
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()

    # Filter out mismatched keys
    matched_dict = {}
    skipped = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                skipped.append(f"{k}: pretrained {v.shape} vs model {model_dict[k].shape}")
        else:
            skipped.append(f"{k}: not in model")

    # Update model dict
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched_dict)}/{len(pretrained_dict)} layers")
    if skipped:
        print(f"Skipped layers:")
        for s in skipped:
            print(f"  {s}")

    return model

# Alternative: Using strict=False
def load_with_strict_false(model, state_dict):
    """
    Load with strict=False and report missing/unexpected keys.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"Missing keys ({len(incompatible.missing_keys)}):")
        for k in incompatible.missing_keys[:5]:
            print(f"  {k}")
        if len(incompatible.missing_keys) > 5:
            print(f"  ... and {len(incompatible.missing_keys)-5} more")

    if incompatible.unexpected_keys:
        print(f"Unexpected keys ({len(incompatible.unexpected_keys)}):")
        for k in incompatible.unexpected_keys[:5]:
            print(f"  {k}")

    return model
```

</details>

### Variation H2: BatchNorm Eval vs Train Mode

**Part 12.1**: Explain the difference in BatchNorm behavior between train and eval mode.

**Part 12.2**: When does this matter for transfer learning?

**Part 12.3**: How to properly freeze BatchNorm?

<details>
<summary>Solution H2</summary>

**12.1**:
```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(64)

# Train mode
bn.train()
x = torch.randn(32, 64, 14, 14)
out_train = bn(x)
# Uses batch statistics (mean, var computed from x)
# Updates running_mean, running_var

# Eval mode
bn.eval()
out_eval = bn(x)
# Uses running_mean, running_var (fixed)
# Does NOT update running statistics

print(f"Train mode output mean: {out_train.mean().item():.6f}")
print(f"Eval mode output mean: {out_eval.mean().item():.6f}")
# These will be different!
```

**12.2**:
This matters when:
1. **Fine-tuning with frozen BN**: If BN is in train mode, running stats get corrupted by new domain data
2. **Small batch sizes**: Batch statistics become noisy
3. **Domain shift**: New data statistics differ from pretrained statistics

**12.3**:
```python
def freeze_batchnorm(model):
    """
    Properly freeze BatchNorm layers:
    1. Set to eval mode (use running stats)
    2. Freeze gamma/beta parameters
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Use running statistics
            module.eval()
            # Freeze affine parameters
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            # Alternatively, disable updates to running stats
            module.track_running_stats = False

    return model

# IMPORTANT: Override train() to keep BN in eval
class ModelWithFrozenBN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        freeze_batchnorm(self.backbone)

    def train(self, mode=True):
        super().train(mode)
        # Keep BN in eval mode even when model is in train mode
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
```

</details>

### Variation H3: Debugging Feature Maps

**Part 12.1**: Write code to visualize intermediate feature maps.

**Part 12.2**: How do you identify if a model is "dead" (all activations are zero)?

<details>
<summary>Solution H3</summary>

**12.1**:
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class FeatureMapVisualizer:
    """
    Hook-based feature map visualization.
    """
    def __init__(self, model, layers):
        self.model = model
        self.features = {}
        self.hooks = []

        # Register hooks
        for name, module in model.named_modules():
            if name in layers:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def __call__(self, x):
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def visualize(self, layer_name, num_channels=16):
        """Visualize first N channels of a layer"""
        feat = self.features[layer_name]
        feat = feat[0].cpu().numpy()  # First sample in batch

        n_show = min(num_channels, feat.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i, ax in enumerate(axes.flat):
            if i < n_show:
                ax.imshow(feat[i], cmap='viridis')
                ax.set_title(f'Channel {i}')
            ax.axis('off')

        plt.suptitle(f'Feature maps: {layer_name}')
        plt.tight_layout()
        return fig

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

# Usage
from torchvision.models import resnet34
model = resnet34(weights='DEFAULT').eval()

vis = FeatureMapVisualizer(model, ['layer1', 'layer2', 'layer3'])
x = torch.randn(1, 3, 224, 224)
features = vis(x)

print("Feature shapes:")
for name, feat in features.items():
    print(f"  {name}: {list(feat.shape)}")
```

**12.2**:
```python
def check_dead_neurons(model, x, threshold=1e-6):
    """
    Check for dead ReLU neurons (always output 0).
    """
    activation_stats = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output.detach()
            activation_stats[name] = {
                'mean': out.mean().item(),
                'std': out.std().item(),
                'zero_pct': (out == 0).float().mean().item() * 100,
                'max': out.max().item(),
                'min': out.min().item()
            }
        return hook

    # Hook all ReLU layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Report
    print("Activation Statistics:")
    print("-" * 60)
    dead_layers = []

    for name, stats in activation_stats.items():
        is_dead = stats['max'] < threshold
        status = "DEAD!" if is_dead else "OK"
        print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"zero%={stats['zero_pct']:.1f}%, {status}")

        if is_dead:
            dead_layers.append(name)

    if dead_layers:
        print(f"\nWARNING: {len(dead_layers)} potentially dead layers!")
    else:
        print("\nAll layers have active neurons.")

    return activation_stats, dead_layers

# Test
model = resnet34(weights='DEFAULT').eval()
x = torch.randn(4, 3, 224, 224)
stats, dead = check_dead_neurons(model, x)
```

</details>

---

## KEY FORMULAS SUMMARY

| Concept | Formula/Definition |
|---------|-------------------|
| **Total Parameters** | Σ_layer numel(weights) + numel(biases) |
| **Conv2d Parameters** | C_out × (C_in × K × K + 1) if bias, else C_out × C_in × K × K |
| **BatchNorm Parameters** | 2 × num_features (gamma + beta) |
| **Linear Parameters** | out_features × (in_features + 1) if bias |
| **Output Size (Conv)** | (W - K + 2P) / S + 1 |
| **ResNet34 Architecture** | [3, 4, 6, 3] BasicBlocks with 64, 128, 256, 512 channels |
| **ResNet50 Architecture** | [3, 4, 6, 3] Bottleneck blocks with 64, 128, 256, 512 base channels |

---

## ATOMIC SKILLS CHECKLIST

- [ ] Count total parameters in a PyTorch model
- [ ] Count parameters for specific layers/modules
- [ ] Compute output shapes for different input sizes
- [ ] Freeze parameters selectively (patterns, layers)
- [ ] Replace classification head for transfer learning
- [ ] Use hooks for feature extraction
- [ ] Handle BatchNorm properly during transfer learning
- [ ] Implement gradual unfreezing
- [ ] Compare model architectures (params, speed, memory)
- [ ] Debug dead neurons and activation issues

---

## COMMON MISCONCEPTIONS

1. **Freezing ≠ eval()**: `param.requires_grad = False` freezes gradients, but `model.eval()` changes BatchNorm/Dropout behavior. For proper freezing, you need both.

2. **BatchNorm running stats**: In train mode, BatchNorm updates running_mean/running_var. If you freeze weights but keep train mode, statistics still change!

3. **Parameter counting with numel()**: This counts elements, not bytes. Memory = numel × dtype_size (4 for float32, 2 for float16).

4. **ResNet "layers"**: layer1, layer2, etc. contain multiple BasicBlock/Bottleneck modules, not single layers.

5. **Pretrained weights and num_classes**: Changing fc layer requires either:
   - Load pretrained first, then modify fc
   - Use strict=False when loading
   - Filter mismatched keys manually

6. **AdaptiveAvgPool2d**: Handles arbitrary input sizes, but very small inputs may lose too much information.

7. **Transfer learning from ImageNet**: ImageNet models expect normalized inputs (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
