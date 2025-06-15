# 🛣️ UNet Road Segmentation with Raycasting

A computer vision project combining U-Net neural network for road segmentation with raycasting algorithms for autonomous vehicle perception.

## 📋 Overview

This project implements a complete pipeline for road detection using:
- **U-Net architecture** for semantic segmentation of road surfaces
- **Raycasting algorithm** for distance measurement and obstacle detection
- **PyTorch framework** for deep learning training and inference

## 🚀 Features

- ✅ **Road Segmentation**: Binary segmentation of road vs non-road areas
- ✅ **Raycasting**: Distance measurement using ray-based collision detection
- ✅ **Real-time Inference**: Single image prediction capabilities
- ✅ **Visualization**: Built-in plotting for results analysis
- ✅ **GPU Support**: CUDA acceleration when available

## 🏗️ Project Structure

```
├── main.py           # Training script
├── inference.py      # Inference and visualization
├── raycast.py        # Raycasting implementation
├── unet.py          # U-Net model architecture
├── carvana_dataset.py # Dataset loading utilities
└── dataset/         # Training data directory
```

## 🔧 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Required Dependencies

#### Core Libraries
```bash
# PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Computer Vision & Image Processing
pip install opencv-python pillow

# Data Science & Visualization
pip install matplotlib numpy scipy

# Utilities
pip install tqdm

# Albumentations (Data Augmentation)
pip install albumentations
```

#### Complete Installation (One Command)
```bash
pip install torch torchvision opencv-python matplotlib pillow numpy scipy tqdm
```

### Conda Environment (Recommended)
If you prefer using conda (based on your environment):

```bash
# Create new environment
conda create -n maskgen python=3.9

# Activate environment
conda activate maskgen

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install opencv matplotlib pillow numpy scipy tqdm -c conda-forge
```

### Verify Installation
```python
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
```

## 📚 Usage

### 1. Training the Model

Train the U-Net model on your road segmentation dataset:

```python
python main.py
```

**Training Configuration:**
- Learning Rate: `3e-4`
- Batch Size: `4`
- Epochs: `100`
- Optimizer: AdamW
- Loss Function: BCEWithLogitsLoss

### 2. Single Image Inference

Perform segmentation on a single image:

```python
from inference import single_image_inference

single_image_inference(
    image_path="path/to/your/image.png",
    model_path="path/to/trained/model.pth",
    device="cuda"  # or "cpu"
)
```

### 3. Raycasting on Segmented Roads

Apply raycasting to detect road boundaries and measure distances:

```python
from raycast import raycast
import cv2

# Load segmentation mask
mask = cv2.imread("path/to/mask.png", cv2.IMREAD_GRAYSCALE)

# Perform raycasting
distances, hit_points = raycast(
    mask, 
    num_rays=40, 
    fov_degrees=120, 
    max_length=200
)

# Visualize results
import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
for pt in hit_points:
    plt.plot([mask.shape[1] // 2, pt[0]], [mask.shape[0] - 1, pt[1]], 'r-')
plt.show()
```

## 🔍 Raycasting Algorithm

The raycasting implementation provides:

### Parameters
- `num_rays`: Number of rays to cast (default: 30)
- `fov_degrees`: Field of view in degrees (default: 90°)
- `max_length`: Maximum ray distance (default: 200 pixels)

### How it Works
1. **Origin Point**: Rays start from bottom-center of the image
2. **Ray Direction**: Evenly distributed across the specified FOV
3. **Collision Detection**: Rays stop when hitting white pixels (road boundaries)
4. **Distance Measurement**: Returns distance to collision point for each ray

### Output
- `distances`: List of distances for each ray
- `hit_points`: List of (x, y) coordinates where rays hit obstacles

## 🎯 Model Architecture

**U-Net Features:**
- Input: 3-channel RGB images (512×512)
- Output: Single-channel binary mask
- Architecture: Encoder-decoder with skip connections
- Activation: Sigmoid for binary classification

## 📊 Training Details

The model is trained with:
- **Dataset Split**: 80% training, 20% validation
- **Data Augmentation**: Resize to 512×512
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: AdamW with learning rate 3e-4
- **Device**: Automatic GPU/CPU detection

## 🔧 Configuration

### Dataset Structure
```
dataset/
├── images/          # RGB road images
├── masks/           # Binary segmentation masks
└── CapturedImages/  # Additional test images
```

### Model Paths
- Training output: `unet.pth`
- Inference input: Specify your trained model path

## 📈 Results Visualization

The project includes comprehensive visualization tools:

### Training Progress
- Real-time loss monitoring
- Epoch-by-epoch performance tracking

### Inference Results  
- Side-by-side original and predicted masks
- Ray visualization on segmented roads
- Distance measurements overlay

## 🚗 Applications

This system is designed for:
- **Autonomous Vehicles**: Road boundary detection
- **ADAS Systems**: Lane keeping assistance  
- **Robotics**: Path planning and navigation
- **Computer Vision Research**: Segmentation + geometric analysis

## 🤝 Contributing

Feel free to contribute by:
1. Adding new raycasting algorithms
2. Improving the U-Net architecture
3. Enhancing visualization features
4. Optimizing performance

## 📝 License

This project is open-source. Please check individual dependencies for their respective licenses.

## 🔗 References

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- PyTorch Deep Learning Framework
- Computer Vision raycasting algorithms

---

**Made with ❤️ for autonomous vehicle perception**