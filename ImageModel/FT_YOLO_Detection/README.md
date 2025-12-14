# 🍄 YOLO Mushroom Detection Project

## Overview

**Stage 1: Object Detection**
- Detect mushroom locations (bounding boxes)
- No edible/poisonous classification
- Single-class detection: `mushroom`

Used with a Stage-2 ResNet classifier to form a two-stage detect + classify system.

## Files

```
FT_YOLO_Detection/
├── data_detection.yaml      # Dataset config (single class)
├── train_detection.py       # Training script ⭐
├── predict_detection.py     # Inference script ⭐
├── README.md                # This document
└── yolo_detector/           # Training outputs (after training)
    ├── weights/
    │   ├── best.pt          # Best model
    │   └── last.pt          # Last epoch model
    ├── results.csv          # Training metrics CSV
    ├── plots/               # Visualizations
    │   ├── loss_curves.png
    │   ├── map_curves.png
    │   └── precision_recall.png
    └── training_report.txt  # Training report
```

## Dataset

Uses `merged_mushroom_dataset` configured as single-class detection:
- **Original**: 2 classes (edible_mushroom, poisonous_mushroom)
- **This project**: 1 class (mushroom)
- YOLO treats all labels as the same class

| Split | Images | Annotations |
|------|---------|---------|
| Train | 3,362 | ~4,500 |
| Val   | 959  | ~1,300 |
| Test  | 458  | ~600  |

Dataset path: `../../Dataset/merged_mushroom_dataset/`

## Quick Start

### 1. Train the detector

```bash
cd ImageModel/FT_YOLO_Detection
python train_detection.py
```

**Training params** (edit in script):
- Model: YOLOv8n (nano, fastest)
- Epochs: 100
- Batch size: 16
- Image size: 640
- Early-stop patience: 20

**Estimated time**:
- GPU (RTX 4070): ~20–30 minutes
- CPU: ~2–4 hours

### 2. Run detection

**Single image**:
```bash
python predict_detection.py --source path/to/image.jpg --mode detect
```

**Detect and crop mushrooms**:
```bash
python predict_detection.py --source path/to/image.jpg --mode crop
```

**Batch a directory**:
```bash
python predict_detection.py --source path/to/images/ --mode batch
```

### 3. Use in Python

```python
from predict_detection import MushroomDetector

# Initialize detector
detector = MushroomDetector('yolo_detector/weights/best.pt')

# Detect mushrooms
detections = detector.detect('mushroom.jpg', conf_threshold=0.25)
for det in detections:
    print(f"bbox: {det['bbox']}, confidence: {det['confidence']}")

# Detect and crop
cropped = detector.detect_and_crop('mushroom.jpg')
print(f"Cropped {len(cropped)} mushrooms")
```

## Model Architecture

### YOLOv8n (Nano)

```
Input (640x640x3)
    ↓
Backbone (CSPDarknet)
    ↓
Neck (FPN + PAN)
    ↓
Detection Head
    ↓
Output:
  - Bounding Box (x, y, w, h)
  - Confidence Score
  - Class: mushroom (single class)
```

**Params**: ~3.2M  
**Inference speed**: ~30 FPS (GPU)

## Difference vs Classification YOLO

| Feature | FT_YOLO (classification) | FT_YOLO_Detection (this project)|
|------|----------------|---------------------------|
| Task | Detect + classify | **Detect only** |
| Output classes | 2 (edible/poisonous) | **1 (mushroom)** |
| Goal | End-to-end recognition | **Pair with ResNet classifier** |
| mAP | Higher (class split) | Higher (single-class simpler) |
| Use | Standalone | **Two-stage pipeline** |

## Training Tips

### 1. Data Augmentation

```python
hsv_h=0.015      # Hue
hsv_s=0.7        # Saturation
hsv_v=0.4        # Value
degrees=10.0     # Rotation
translate=0.1    # Translation
scale=0.5        # Scale
fliplr=0.5       # Horizontal flip
mosaic=1.0       # Mosaic
```

### 2. Regularization

- Weight Decay: 0.0005
- Early Stopping: patience=20
- Learning Rate Scheduling: cosine annealing

### 3. Model Choices

| Model | Params | Speed | Accuracy | Recommended |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | Fastest | Good | **Default** |
| YOLOv8s | 11.2M | Fast | Great | Balanced |
| YOLOv8m | 25.9M | Medium | Higher | Accuracy focus |
| YOLOv8l | 43.7M | Slow | Highest | Max accuracy |

Change model:
```python
config = {
    'model': 'yolov8n.pt',  # switch to yolov8s/m/l/x.pt
    ...
}
```

## Metrics to Track

After training, check:

- **mAP@0.5**: AP at IoU 0.5
- **mAP@0.5:0.95**: AP averaged 0.5–0.95
- **Precision**: correct detections / all detections
- **Recall**: detected mushrooms / all mushrooms
- **Losses**: Box / Class (single class, should be low) / DFL

## Two-Stage Pipeline Integration

### Workflow

```
Input image
    ↓
[Stage 1] YOLO detector (this project)
    ↓
Detect N mushroom locations and bounding boxes
    ↓
Crop N mushroom regions
    ↓
[Stage 2] ResNet classifier
    ↓
For each mushroom: edible or poisonous
    ↓
Output final result
```

