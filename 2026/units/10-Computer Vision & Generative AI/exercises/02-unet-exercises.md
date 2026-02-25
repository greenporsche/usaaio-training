# UNet Exercises

**5 exercises** | Covers: encoder-decoder shapes, skip connections, transposed convolution, Dice loss, parameter counting

---

## Exercise 1: Trace UNet Shapes

**Target time**: 5 minutes

A UNet has input shape $(B, 1, 256, 256)$ with feature channels $[64, 128, 256, 512]$ and bottleneck channels $1024$.

**Part 1**: Fill in the shape at each stage of the encoder path (after each ConvBlock and after each MaxPool):

| Level | After ConvBlock | After MaxPool |
|---|---|---|
| Enc 1 | ? | ? |
| Enc 2 | ? | ? |
| Enc 3 | ? | ? |
| Enc 4 | ? | ? |
| Bottleneck | ? | — |

**Part 2**: Fill in the shape at each stage of the decoder path (after UpConv, after Concat, after ConvBlock):

| Level | After UpConv | After Concat | After ConvBlock |
|---|---|---|---|
| Dec 4 | ? | ? | ? |
| Dec 3 | ? | ? | ? |
| Dec 2 | ? | ? | ? |
| Dec 1 | ? | ? | ? |

<details>
<summary>Solution</summary>

**Part 1**:

| Level | After ConvBlock | After MaxPool |
|---|---|---|
| Enc 1 | $(B, 64, 256, 256)$ | $(B, 64, 128, 128)$ |
| Enc 2 | $(B, 128, 128, 128)$ | $(B, 128, 64, 64)$ |
| Enc 3 | $(B, 256, 64, 64)$ | $(B, 256, 32, 32)$ |
| Enc 4 | $(B, 512, 32, 32)$ | $(B, 512, 16, 16)$ |
| Bottleneck | $(B, 1024, 16, 16)$ | — |

**Part 2**:

| Level | After UpConv | After Concat | After ConvBlock |
|---|---|---|---|
| Dec 4 | $(B, 512, 32, 32)$ | $(B, 1024, 32, 32)$ | $(B, 512, 32, 32)$ |
| Dec 3 | $(B, 256, 64, 64)$ | $(B, 512, 64, 64)$ | $(B, 256, 64, 64)$ |
| Dec 2 | $(B, 128, 128, 128)$ | $(B, 256, 128, 128)$ | $(B, 128, 128, 128)$ |
| Dec 1 | $(B, 64, 256, 256)$ | $(B, 128, 256, 256)$ | $(B, 64, 256, 256)$ |

The concat doubles channels because skip features (from encoder) are concatenated with upsampled features.

</details>

---

## Exercise 2: Transposed Convolution Output Size

**Target time**: 3 minutes

The formula for transposed convolution output size is:
$$H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}$$

**Part 1**: Given `ConvTranspose2d(512, 256, kernel_size=2, stride=2)` with input $(B, 512, 16, 16)$, what is the output shape?

**Part 2**: Given `ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)` with input $(B, 256, 32, 32)$, what is the output shape?

**Part 3**: Why does UNet typically use `kernel_size=2, stride=2` for upsampling? What's the alternative?

<details>
<summary>Solution</summary>

**Part 1**:
$H_{out} = (16-1) \times 2 - 0 + 2 + 0 = 30 + 2 = 32$

Output: $(B, 256, 32, 32)$ — exactly doubles spatial dimensions.

**Part 2**:
$H_{out} = (32-1) \times 2 - 2 \times 1 + 4 + 0 = 62 - 2 + 4 = 64$

Output: $(B, 128, 64, 64)$ — also doubles.

**Part 3**: `kernel_size=2, stride=2` is the simplest 2x upsampling with no overlap (each output pixel comes from exactly one input pixel). Alternative: bilinear upsampling (`F.interpolate(x, scale_factor=2, mode='bilinear')`) followed by a regular conv layer. The alternative avoids "checkerboard artifacts" that transposed convolutions can produce.

</details>

---

## Exercise 3: Compute Dice Loss

**Target time**: 4 minutes

A binary segmentation model produces the following prediction (after sigmoid) for a 4x4 image:

**Predicted probabilities** $P$:
```
0.9  0.8  0.1  0.05
0.7  0.6  0.2  0.1
0.3  0.4  0.85 0.9
0.1  0.05 0.7  0.95
```

**Ground truth** $G$ (binary):
```
1  1  0  0
1  1  0  0
0  0  1  1
0  0  1  1
```

**Part 1**: Compute $\sum_i p_i g_i$ (element-wise product, then sum).

**Part 2**: Compute $\sum_i p_i$ and $\sum_i g_i$.

**Part 3**: Compute the Dice score and Dice loss using:
$$\text{Dice} = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}, \quad \mathcal{L}_\text{Dice} = 1 - \text{Dice}$$

<details>
<summary>Solution</summary>

**Part 1**: Element-wise product (only nonzero where $g_i = 1$):
$\sum p_i g_i = 0.9 + 0.8 + 0.7 + 0.6 + 0.85 + 0.9 + 0.7 + 0.95 = 6.4$

**Part 2**:
$\sum p_i = 0.9 + 0.8 + 0.1 + 0.05 + 0.7 + 0.6 + 0.2 + 0.1 + 0.3 + 0.4 + 0.85 + 0.9 + 0.1 + 0.05 + 0.7 + 0.95 = 7.7$

$\sum g_i = 8$ (eight 1s)

**Part 3**:
$\text{Dice} = \frac{2 \times 6.4}{7.7 + 8} = \frac{12.8}{15.7} \approx 0.815$

$\mathcal{L}_\text{Dice} = 1 - 0.815 = 0.185$

</details>

---

## Exercise 4: Skip Connection Analysis

**Target time**: 3 minutes

**Part 1**: In a UNet encoder, level 3 produces features of shape $(B, 256, 64, 64)$. In the decoder, after upsampling level 4, we get shape $(B, 256, 64, 64)$. After concatenation with the skip connection, what is the shape? What is the input channel count for the next ConvBlock?

**Part 2**: If we used additive skip connections (like ResNet) instead of concatenation, what would the shape be after combining? What would the input channel count be?

**Part 3**: What information does the skip connection provide that the upsampled decoder features don't have?

<details>
<summary>Solution</summary>

**Part 1**: After concatenation along the channel dimension:
$(B, 256+256, 64, 64) = (B, 512, 64, 64)$

The next ConvBlock takes 512 input channels.

**Part 2**: After addition:
$(B, 256, 64, 64)$ — same shape (element-wise addition doesn't change dimensions).

The next ConvBlock takes 256 input channels (half of the concatenation case).

**Part 3**: The skip connection provides high-resolution spatial details (edges, textures, exact boundaries) from the encoder. The decoder's upsampled features have semantic/contextual information but have lost fine spatial detail through the pooling operations. The skip connection restores the "where" information that the encoder's "what" information needs.

</details>

---

## Exercise 5: UNet Parameter Count

**Target time**: 5 minutes

A simplified UNet has:
- Input: 1 channel
- Encoder: one level with ConvBlock(1→32) + MaxPool
- Bottleneck: ConvBlock(32→64)
- Decoder: ConvTranspose2d(64, 32, 2, stride=2) + ConvBlock(64→32)
- Output: Conv2d(32, 2, 1) (2-class segmentation)

Each ConvBlock = Conv2d(in, out, 3) → Conv2d(out, out, 3) (no bias, no BN for simplicity).

**Part 1**: How many parameters in the encoder ConvBlock?

**Part 2**: How many parameters in the bottleneck ConvBlock?

**Part 3**: How many total parameters in the entire model?

<details>
<summary>Solution</summary>

**Part 1**: Encoder ConvBlock(1→32):
- Conv2d(1, 32, 3): $1 \times 32 \times 3 \times 3 = 288$ params
- Conv2d(32, 32, 3): $32 \times 32 \times 3 \times 3 = 9{,}216$ params
- Total: $288 + 9{,}216 = 9{,}504$

**Part 2**: Bottleneck ConvBlock(32→64):
- Conv2d(32, 64, 3): $32 \times 64 \times 3 \times 3 = 18{,}432$
- Conv2d(64, 64, 3): $64 \times 64 \times 3 \times 3 = 36{,}864$
- Total: $18{,}432 + 36{,}864 = 55{,}296$

**Part 3**: Full model:
- Encoder ConvBlock: $9{,}504$
- Bottleneck ConvBlock: $55{,}296$
- ConvTranspose2d(64, 32, 2, stride=2): $64 \times 32 \times 2 \times 2 = 8{,}192$
- Decoder ConvBlock(64→32): Conv(64,32,3) = $18{,}432$ + Conv(32,32,3) = $9{,}216$ = $27{,}648$
- Output Conv2d(32, 2, 1): $32 \times 2 \times 1 \times 1 = 64$

**Total: $9{,}504 + 55{,}296 + 8{,}192 + 27{,}648 + 64 = 100{,}704$**

Note: Decoder ConvBlock takes 64 channels (32 from upsample + 32 from skip = 64 after concat).

</details>

---
