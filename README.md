# Semantic Segmentation on Potsdam Aerial Imagery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A comprehensive deep learning project for **semantic segmentation** of high-resolution aerial imagery from the ISPRS Potsdam dataset. This project compares multiple architectures including CNN-based and Transformer-based approaches, with transfer learning techniques.

<p align="center">
  <img src="assets/sample_prediction.png" alt="Sample Segmentation Result" width="800"/>
  <br>
  <em>Example: RGB input, ground truth mask, and model prediction</em>
</p>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visual Examples](#visual-examples)
- [Report](#report)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Problem Statement

Semantic segmentation of aerial/satellite imagery is a critical task in remote sensing, urban planning, and environmental monitoring. The challenge involves:

- **Pixel-wise classification** of high-resolution aerial images into multiple land cover categories
- Handling **class imbalance** (e.g., cars are significantly underrepresented compared to buildings)
- Leveraging **multi-spectral data** (RGB + Near-Infrared + Digital Surface Model)
- Achieving accurate boundary delineation between adjacent classes

This project addresses these challenges by implementing and comparing multiple deep learning architectures with various training strategies.

---

## Dataset

**ISPRS Potsdam 2D Semantic Labeling Dataset**

| Property | Value |
|----------|-------|
| **Total Tiles** | 15,048 |
| **Tile Size** | 224 x 224 pixels |
| **Input Channels** | 5 (R, G, B, NIR, DSM) |
| **Classes** | 6 |
| **Format** | GeoTIFF |
| **Split** | 70% Train / 15% Val / 15% Test |

### Class Distribution

| Class ID | Class Name | Color | Description |
|----------|------------|-------|-------------|
| 0 | Impervious Surface | Brown | Roads, pavements, parking lots |
| 1 | Building | Yellow | Residential and commercial buildings |
| 2 | Low Vegetation | Light Blue | Grass, shrubs |
| 3 | Tree | Green | Trees and forests |
| 4 | Car | Orange | Vehicles |
| 5 | Clutter/Background | Gray | Other objects |

### Input Channels

- **RGB (Bands 1-3)**: True color aerial imagery
- **NIR (Band 4)**: Near-Infrared for vegetation analysis (NDVI computation)
- **DSM (Band 5)**: Digital Surface Model for height information

---

## Methodology

### 1. Data Preprocessing

- **Normalization**: RGB+NIR scaled to [0,1]; DSM z-score normalized and clipped
- **Data Augmentation** (training only):
  - Horizontal/Vertical flips
  - Random 90-degree rotations
  - Shift-Scale-Rotate transformations
  - Photometric augmentations (brightness, contrast, gamma)
  - Gaussian noise injection

### 2. Model Architectures

| Model | Type | Backbone | Trainable Params | Strategy |
|-------|------|----------|------------------|----------|
| **Baseline CNN** | CNN | Custom 2-layer | ~10K | Train from scratch |
| **Deeper Model** | CNN | Custom multi-layer | ~100K | Train from scratch |
| **DeepLabV3** | CNN | ResNet-50 | Decoder only | Transfer learning (frozen encoder) |
| **SegFormer-B0** | Transformer | MiT-B0 | Decoder only | Transfer learning (frozen encoder) |

### 3. Training Strategy

- **Loss Function**: Weighted Cross-Entropy (inverse frequency weighting for class imbalance)
- **Optimizer**: Adam / AdamW with weight decay
- **Learning Rate**: 1e-3 (baseline), 1e-4 to 5e-4 (transfer learning)
- **Epochs**: 10 per model
- **Batch Size**: 32 (training), 128 (validation)
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU

### 4. Evaluation Metrics

- **Pixel Accuracy**: Overall classification accuracy
- **Mean IoU (mIoU)**: Intersection over Union averaged across classes
- **Per-Class IoU**: Individual class performance
- **Macro F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

---

## Project Structure

```
.
├── data_analys.ipynb                                    # Exploratory data analysis
├── data_preprocessing_baseline_U_net_model.ipynb        # Data pipeline & baseline models
├── Fine_Tunning.ipynb                                   # Fine-tuning experiments
├── transformLearning_cnnBased_transformersBased.ipynb   # Transfer learning comparison
│
├── docs/
│   ├── report.pdf                   # Detailed project report
│   └── architecture.md              # Model architecture details
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── CONTRIBUTING.md                  # Contribution guidelines
└── README.md                        # This file
```

### Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| `data_analys.ipynb` | Dataset exploration, class distribution analysis, band statistics, correlation heatmaps, NDVI/DSM visualizations |
| `data_preprocessing_baseline_U_net_model.ipynb` | Data loading pipeline, augmentation setup, baseline CNN training, class weighting experiments |
| `Fine_Tunning.ipynb` | End-to-end fine-tuning of DeepLabV3 and SegFormer models |
| `transformLearning_cnnBased_transformersBased.ipynb` | Transfer learning with frozen encoders, model comparison, final evaluation |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/semantic-segmentation-potsdam.git
cd semantic-segmentation-potsdam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the Potsdam dataset from [ISPRS website](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) and update the `data_dir` path in the notebooks.

---

## Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook data_analys.ipynb
```

Explore class distributions, visualize samples, and analyze band statistics.

### 2. Train Baseline Models

```bash
jupyter notebook data_preprocessing_baseline_U_net_model.ipynb
```

Run cells sequentially to:
- Load and preprocess data
- Train baseline CNN models
- Compare weighted vs. unweighted loss

### 3. Transfer Learning Experiments

```bash
jupyter notebook transformLearning_cnnBased_transformersBased.ipynb
```

Train DeepLabV3 and SegFormer with frozen encoders.

### 4. Fine-Tuning

```bash
jupyter notebook Fine_Tunning.ipynb
```

End-to-end fine-tuning for best results.

### Quick Inference Example

```python
import torch
from model import build_deeplabv3_resnet50_5ch_6cls_frozen

# Load model
model = build_deeplabv3_resnet50_5ch_6cls_frozen(pretrained=True)
model.load_state_dict(torch.load('deepLab_model.ckpt')['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    prediction = model(input_tensor)  # (B, 5, 224, 224) -> (B, 6, 224, 224)
    segmentation_map = prediction.argmax(dim=1)
```

---

## Results

### Model Comparison (Validation Set, Average of Last 3 Epochs)

| Model | mIoU | Macro F1 | Pixel Accuracy |
|-------|------|----------|----------------|
| Baseline CNN | 0.372 | 0.527 | ~60% |
| Deeper Model | 0.444 | 0.601 | ~65% |
| **DeepLabV3-ResNet50** | **0.470** | **0.626** | **~69%** |
| SegFormer-B0 | 0.427 | 0.585 | ~65% |

### Per-Class IoU (DeepLabV3 - Best Model)

| Class | IoU |
|-------|-----|
| Impervious Surface | 0.516 |
| Building | 0.729 |
| Low Vegetation | 0.440 |
| Tree | 0.536 |
| Car | 0.223 |
| Clutter/Background | 0.376 |

### Key Findings

1. **Transfer learning significantly outperforms** training from scratch
2. **DeepLabV3 achieves best overall performance** with mIoU of 0.47
3. **Class imbalance remains challenging** - Car class has lowest IoU due to small object size and underrepresentation
4. **Multi-spectral input (RGB+NIR+DSM)** provides richer features than RGB alone
5. **Weighted loss function** improves minority class performance

---

## Visual Examples

<p align="center">
  <img src="assets/results_visualization.png" alt="Segmentation Results" width="800"/>
  <br>
  <em>Left to right: RGB Input, Ground Truth, DeepLabV3 Prediction, SegFormer Prediction</em>
</p>

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
  <br>
  <em>Normalized confusion matrix showing per-class accuracy</em>
</p>

<p align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="600"/>
  <br>
  <em>Training and validation loss/mIoU over epochs</em>
</p>

> **Note**: Add your generated plots from the notebooks to the `assets/` folder

---

## Report

For detailed methodology, experiments, and analysis, see the full project report:

[Project Report (PDF)](docs/report.pdf)

---

## Technologies Used

### Deep Learning Frameworks
- **PyTorch 2.6** - Primary framework for model implementation
- **TensorFlow** - Data pipeline utilities
- **Hugging Face Transformers** - SegFormer implementation

### Computer Vision
- **torchvision** - DeepLabV3 pretrained models
- **Albumentations** - Advanced image augmentation
- **OpenCV** - Image processing utilities
- **rasterio** - GeoTIFF file handling

### Data Science
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **scikit-learn** - Train/val/test splitting, metrics

### Development Tools
- **Jupyter Notebook** - Interactive development
- **CUDA 12.4** - GPU acceleration

---

## Future Work

- [ ] Experiment with larger SegFormer variants (B2, B4, B5)
- [ ] Implement attention-based fusion for multi-spectral inputs
- [ ] Add test-time augmentation (TTA) for improved inference
- [ ] Explore boundary-aware loss functions
- [ ] Deploy model as a REST API or web application
- [ ] Extend to other remote sensing datasets (Vaihingen, LoveDA)

---

## Author

**[Your Name]**

- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- Email: your.email@example.com
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [ISPRS](https://www.isprs.org/) for the Potsdam benchmark dataset
- [NVIDIA](https://huggingface.co/nvidia) for SegFormer pretrained weights
- [PyTorch](https://pytorch.org/) team for the deep learning framework

---

<p align="center">
  <i>If you find this project useful, please consider giving it a star!</i>
</p>
