# 🚀 Quickstart - YOLO Mushroom Detection

## Get started in 5 minutes

### Step 1: Train the model (20–30 min)

```bash
cd ImageModel/FT_YOLO_Detection
python train_detection.py
```

Model will be saved to `yolo_detector/weights/best.pt`.

---

### Step 2: Test detection

**Single image**
```bash
python predict_detection.py --source ../../Dataset/merged_mushroom_dataset/test/images/sample.jpg --mode detect
```

**Detect and crop mushrooms**
```bash
python predict_detection.py --source ../../Dataset/merged_mushroom_dataset/test/images/sample.jpg --mode crop
```

Crops are saved to `cropped_mushrooms/`.

---

### Step 3: Batch process

```bash
python predict_detection.py --source ../../Dataset/merged_mushroom_dataset/test/images/ --mode batch
```

---

## View training results

### 1) Curves
- `yolo_detector/plots/loss_curves.png` - loss curves
- `yolo_detector/plots/map_curves.png` - mAP curves
- `yolo_detector/plots/precision_recall.png` - precision/recall

### 2) Training data
- `yolo_detector/results.csv` - per-epoch data
- `yolo_detector/training_report.txt` - summary

### 3) Model files
- `yolo_detector/weights/best.pt` - best model (highest mAP)
- `yolo_detector/weights/last.pt` - last epoch

---

## Python examples

### Basic detection

```python
from predict_detection import MushroomDetector

detector = MushroomDetector('yolo_detector/weights/best.pt')
detections = detector.detect('test.jpg')

for i, det in enumerate(detections):
    print(f"Mushroom {i+1}:")
    print(f"  bbox: {det['bbox']}")
    print(f"  confidence: {det['confidence']:.2%}")
```

### Detect and crop

```python
cropped = detector.detect_and_crop('test.jpg', output_dir='my_mushrooms')

print(f"Detected {len(cropped)} mushrooms")
for i, crop in enumerate(cropped):
    print(f"Mushroom {i+1} saved to: {crop['file_path']}")
```

### With ResNet classifier

```python
from predict_detection import MushroomDetector
import sys
sys.path.append('../FT_ResNet')
from predict import MushroomPredictor

detector = MushroomDetector('yolo_detector/weights/best.pt')
classifier = MushroomPredictor('../FT_ResNet/resnet_classifier_no_dropout/best_model.pth')

cropped = detector.detect_and_crop('field_photo.jpg')

for i, mushroom in enumerate(cropped):
    class_name, conf, probs = classifier.predict(mushroom['file_path'])
    
    print(f"\nMushroom {i+1}:")
    print(f"  detection confidence: {mushroom['confidence']:.2%}")
    print(f"  class: {class_name}")
    print(f"  class confidence: {conf:.2%}")
    
    if class_name == 'poisonous':
        print("  ⚠️  Warning: poisonous!")
    else:
        print("  ✅ Edible")
```

---

## Command cheatsheet

| Task | Command |
|------|---------|
| Train | `python train_detection.py` |
| Detect one image | `python predict_detection.py --source image.jpg --mode detect` |
| Crop mushrooms | `python predict_detection.py --source image.jpg --mode crop` |
| Batch detect | `python predict_detection.py --source images/ --mode batch` |
| Adjust confidence | `python predict_detection.py --source image.jpg --conf 0.5` |

---

## Tuning parameters

### Training (edit `train_detection.py`)

```python
config = {
    'model': 'yolov8n.pt',   # n/s/m/l/x
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'patience': 20,
    'device': '0'            # GPU id or 'cpu'
}
```

### Inference

```python
# Confidence threshold (higher = stricter)
conf_threshold = 0.25  # default
conf_threshold = 0.5   # stricter
conf_threshold = 0.1   # more permissive
```

---

## Troubleshooting

### Issue 1: Out of GPU memory
```python
# Reduce batch size
'batch_size': 8  # or smaller
```

### Issue 2: Training is slow
```python
# Use a smaller model
'model': 'yolov8n.pt'
```

### Issue 3: Detection quality is poor
```python
# Lower confidence threshold
conf_threshold = 0.1

# Or train more epochs
'epochs': 150
```

---

## Next

1. ✅ Train YOLO detector
2. ✅ Test detection
3. 🔄 Integrate ResNet classifier
4. 📊 Evaluate and compare

See full docs: `README.md`

