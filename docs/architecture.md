# Model Architecture Details

This document provides detailed information about the neural network architectures used in this semantic segmentation project.

---

## Table of Contents

1. [Input Specification](#input-specification)
2. [Baseline CNN](#baseline-cnn)
3. [DeepLabV3-ResNet50](#deeplabv3-resnet50)
4. [SegFormer-B0](#segformer-b0)
5. [Architecture Comparison](#architecture-comparison)

---

## Input Specification

All models receive the same input format:

```
Input Shape: (Batch, 5, 224, 224)
├── Channel 0: Red (normalized to [0, 1])
├── Channel 1: Green (normalized to [0, 1])
├── Channel 2: Blue (normalized to [0, 1])
├── Channel 3: Near-Infrared (normalized to [0, 1])
└── Channel 4: Digital Surface Model (z-score normalized, clipped to [-5, 5])

Output Shape: (Batch, 6, 224, 224)
└── 6 class logits per pixel
```

---

## Baseline CNN

A simple 2-layer convolutional network used as a performance baseline.

### Architecture

```
DoubleConv(
  (conv): Sequential(
    (0): Conv2d(5, 32, kernel_size=3, padding=1, bias=False)
    (1): BatchNorm2d(32)
    (2): ReLU(inplace=True)
    (3): Conv2d(32, 6, kernel_size=3, padding=1)
  )
)
```

### Properties

| Property | Value |
|----------|-------|
| Total Parameters | ~10,000 |
| Trainable Parameters | ~10,000 |
| Receptive Field | 5x5 pixels |
| Training Strategy | From scratch |

### Limitations

- Very limited receptive field
- Cannot capture global context
- No multi-scale feature extraction

---

## DeepLabV3-ResNet50

State-of-the-art CNN architecture with Atrous Spatial Pyramid Pooling (ASPP).

### Architecture Overview

```
DeepLabV3-ResNet50
├── Backbone: ResNet-50 (modified for 5 input channels)
│   ├── conv1: Conv2d(5, 64, 7x7, stride=2) [MODIFIED]
│   ├── bn1 + relu + maxpool
│   ├── layer1: 3x Bottleneck blocks
│   ├── layer2: 4x Bottleneck blocks
│   ├── layer3: 6x Bottleneck blocks (dilated)
│   └── layer4: 3x Bottleneck blocks (dilated)
│
├── ASPP Module (Atrous Spatial Pyramid Pooling)
│   ├── 1x1 convolution
│   ├── 3x3 conv, dilation=6
│   ├── 3x3 conv, dilation=12
│   ├── 3x3 conv, dilation=18
│   └── Global average pooling
│
└── Decoder
    ├── 1x1 conv to 256 channels
    └── 1x1 conv to 6 classes [MODIFIED]
```

### Input Channel Modification

The original ResNet-50 expects 3 channels (RGB). We modify the first convolution:

```python
# Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
# Modified: Conv2d(5, 64, kernel_size=7, stride=2, padding=3)

# Weight initialization for extra channels:
# - Channels 0-2: Copy pretrained RGB weights
# - Channels 3-4: Initialize with mean of RGB weights
```

### Training Configuration

| Property | Value |
|----------|-------|
| Total Parameters | ~39M |
| Trainable Parameters | ~2M (decoder only) |
| Pretrained Weights | COCO + VOC |
| Encoder | Frozen during training |
| Optimizer | Adam, lr=5e-4 |

### ASPP Benefits

- Multi-scale context aggregation
- Captures both local and global information
- Robust to object scale variation

---

## SegFormer-B0

Transformer-based architecture designed for efficient semantic segmentation.

### Architecture Overview

```
SegFormer-B0
├── Hierarchical Transformer Encoder (MiT-B0)
│   ├── Stage 1: Patch Embed (5→32) + 2x Transformer Blocks
│   │   └── Output: H/4 x W/4 x 32
│   ├── Stage 2: Patch Embed (32→64) + 2x Transformer Blocks
│   │   └── Output: H/8 x W/8 x 64
│   ├── Stage 3: Patch Embed (64→160) + 2x Transformer Blocks
│   │   └── Output: H/16 x W/16 x 160
│   └── Stage 4: Patch Embed (160→256) + 2x Transformer Blocks
│       └── Output: H/32 x W/32 x 256
│
└── MLP Decoder
    ├── Upsample all stages to H/4 x W/4
    ├── Concatenate: 32 + 64 + 160 + 256 = 512 channels
    ├── MLP fusion: 512 → 256
    └── Classification: 256 → 6 classes [MODIFIED]
```

### Efficient Self-Attention

SegFormer uses Efficient Self-Attention to reduce computational complexity:

```
Standard Attention: O(N²)
Efficient Attention: O(N²/R)

where R is the reduction ratio (default: 8)
```

### Patch Embedding Modification

```python
# Original: Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
# Modified: Conv2d(5, 32, kernel_size=7, stride=4, padding=3)

# Weight initialization for extra channels:
# - Channels 0-2: Copy pretrained RGB weights
# - Channels 3-4: Initialize with mean of RGB weights
```

### Training Configuration

| Property | Value |
|----------|-------|
| Total Parameters | ~3.7M |
| Trainable Parameters | ~400K (decoder only) |
| Pretrained Weights | ADE20K |
| Encoder | Frozen during training |
| Optimizer | AdamW, lr=1e-4 |

### Key Innovations

- **No positional encoding**: Uses 3x3 depth-wise convolutions instead
- **Hierarchical features**: Multi-scale representations like CNNs
- **Lightweight decoder**: Simple MLP instead of complex decoder

---

## Architecture Comparison

### Visual Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   INPUT (5, 224, 224)               │
                    └─────────────────────────────────────────────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │  Baseline CNN │              │  DeepLabV3    │              │  SegFormer    │
    │    (2-layer)  │              │  (ResNet-50)  │              │    (MiT-B0)   │
    └───────────────┘              └───────────────┘              └───────────────┘
            │                               │                               │
            │                      ┌────────┴────────┐             ┌────────┴────────┐
            │                      │   ResNet-50    │             │  Hierarchical   │
            │                      │   Backbone     │             │  Transformer    │
            │                      │   (FROZEN)     │             │   (FROZEN)      │
            │                      └────────┬────────┘             └────────┬────────┘
            │                               │                               │
            │                      ┌────────┴────────┐             ┌────────┴────────┐
            │                      │      ASPP      │             │   MLP Decoder   │
            │                      │    Module      │             │                 │
            │                      └────────┬────────┘             └────────┬────────┘
            │                               │                               │
            ▼                               ▼                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │                         OUTPUT (6, 224, 224) - Class Logits                   │
    └───────────────────────────────────────────────────────────────────────────────┘
```

### Quantitative Comparison

| Metric | Baseline | DeepLabV3 | SegFormer |
|--------|----------|-----------|-----------|
| Parameters | ~10K | ~39M | ~3.7M |
| Trainable | ~10K | ~2M | ~400K |
| GFLOPs* | ~0.1 | ~50 | ~4 |
| mIoU | 0.37 | **0.47** | 0.43 |
| Inference Speed | Fastest | Slowest | Fast |

*Approximate values for 224x224 input

### When to Use Each

| Model | Best For |
|-------|----------|
| **Baseline** | Quick experiments, sanity checks |
| **DeepLabV3** | Best accuracy, when compute is available |
| **SegFormer** | Balance of accuracy and efficiency |

---

## Loss Function

All models use Weighted Cross-Entropy Loss to handle class imbalance:

```python
weights = 1.0 / class_frequencies  # Inverse frequency weighting
criterion = nn.CrossEntropyLoss(weight=weights)
```

### Class Weights (Example)

| Class | Frequency | Weight |
|-------|-----------|--------|
| Impervious | High | Low |
| Building | Medium | Medium |
| Low Vegetation | Medium | Medium |
| Tree | Medium | Medium |
| Car | Low | **High** |
| Clutter | Low | **High** |

---

## References

1. **DeepLabV3**: Chen, L.C., et al. "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
2. **SegFormer**: Xie, E., et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (2021)
3. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition" (2016)
