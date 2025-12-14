"""
YOLOv8 mushroom detection training script.
Stage 1: detect mushroom locations only; no toxicity classification.
"""

import os
# Resolve OpenMP duplicate library warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def train_yolo_detector(
    data_yaml='data_detection.yaml',
    model='yolov8n.pt',
    epochs=100,
    batch_size=16,
    img_size=640,
    patience=20,
    device='0'
):
    """
    Train YOLO detector
    
    Args:
        data_yaml: dataset config file
        model: pretrained model (yolov8n/s/m/l/x)
        epochs: number of epochs
        batch_size: batch size
        img_size: image size
        patience: early-stop patience
        device: training device ('0'=GPU, 'cpu'=CPU)
    """
    
    print("="*60)
    print("YOLOv8 Mushroom Detection Training")
    print("Single-class: locate mushrooms only, no toxicity classification")
    print("="*60)
    
    print(f"\nTraining config:")
    print(f"  Model: {model}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Early-stop patience: {patience}")
    print(f"  Device: {'GPU' if device == '0' else 'CPU'}")
    
    # Check dataset config
    if not Path(data_yaml).exists():
        print(f"\n[ERROR] Dataset config file not found: {data_yaml}")
        return
    
    # Load YOLO model
    print(f"\nLoading model: {model}")
    yolo_model = YOLO(model)
    
    # Start training
    print("\n" + "="*60)
    print("Training...")
    print("="*60 + "\n")
    
    results = yolo_model.train(
        data=data_yaml,           # dataset config
        epochs=epochs,            # epochs
        batch=batch_size,         # batch size
        imgsz=img_size,           # image size
        device=device,            # training device
        project='.',              # project dir
        name='yolo_detector',     # experiment name
        exist_ok=True,            # overwrite existing dir
        patience=patience,        # early-stop patience
        save=True,                # save checkpoints
        save_period=10,           # save every N epochs
        pretrained=True,          # use pretrained weights
        optimizer='Adam',         # optimizer
        lr0=0.001,                # initial LR
        weight_decay=0.0005,      # weight decay
        warmup_epochs=3,          # warmup epochs
        cos_lr=True,              # cosine LR schedule
        plots=True,               # save training plots
        verbose=True,             # verbose output
        
        # Augmentation
        hsv_h=0.015,              # hue
        hsv_s=0.7,                # saturation
        hsv_v=0.4,                # value
        degrees=10.0,             # rotation
        translate=0.1,            # translation
        scale=0.5,                # scaling
        shear=0.0,                # shear
        perspective=0.0,          # perspective
        flipud=0.0,               # vertical flip
        fliplr=0.5,               # horizontal flip
        mosaic=1.0,               # mosaic
        mixup=0.0,                # mixup
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Generate visualizations
    print("\nGenerating training metric visualizations...")
    plot_training_metrics('yolo_detector')
    
    print(f"\nModel saved to: yolo_detector/weights/best.pt")
    print(f"Training CSV: yolo_detector/results.csv")
    print(f"Plots: yolo_detector/plots/")
    
    return results


def plot_training_metrics(exp_dir):
    """Plot training metrics"""
    
    results_csv = Path(exp_dir) / 'results.csv'
    
    if not results_csv.exists():
        print(f"[WARNING] results file not found: {results_csv}")
        return
    
    # Load training data
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    plots_dir = Path(exp_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(df) + 1)
    
    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box Loss
    axes[0, 0].plot(epochs, df['train/box_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, df['val/box_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Box Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class Loss
    axes[0, 1].plot(epochs, df['train/cls_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, df['val/cls_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Class Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # DFL Loss
    axes[1, 0].plot(epochs, df['train/dfl_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, df['val/dfl_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('DFL Loss')
    axes[1, 0].set_title('DFL Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total Loss
    train_total = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_total = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    axes[1, 1].plot(epochs, train_total, 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, val_total, 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Loss curves: {plots_dir / 'loss_curves.png'}")
    
    # 2. mAP curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(epochs, df['metrics/mAP50(B)'], 'g-', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('mAP@0.5')
    axes[0].set_title('mAP@0.5')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, df['metrics/mAP50-95(B)'], 'b-', linewidth=2, marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP@0.5:0.95')
    axes[1].set_title('mAP@0.5:0.95')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'map_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] mAP curves: {plots_dir / 'map_curves.png'}")
    
    # 3. Precision & Recall
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(epochs, df['metrics/precision(B)'], 'purple', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, df['metrics/recall(B)'], 'orange', linewidth=2, marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Precision & Recall: {plots_dir / 'precision_recall.png'}")
    
    # 4. Generate report
    best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
    report = f"""
Mushroom Detection Training Report
{'='*60}

Training config:
  Total epochs: {len(df)}
  Best epoch: {best_epoch}

Best metrics (Epoch {best_epoch}):
  mAP@0.5: {df.loc[best_epoch-1, 'metrics/mAP50(B)']:.4f}
  mAP@0.5:0.95: {df.loc[best_epoch-1, 'metrics/mAP50-95(B)']:.4f}
  Precision: {df.loc[best_epoch-1, 'metrics/precision(B)']:.4f}
  Recall: {df.loc[best_epoch-1, 'metrics/recall(B)']:.4f}

Final metrics (Epoch {len(df)}):
  mAP@0.5: {df.iloc[-1]['metrics/mAP50(B)']:.4f}
  mAP@0.5:0.95: {df.iloc[-1]['metrics/mAP50-95(B)']:.4f}
  Precision: {df.iloc[-1]['metrics/precision(B)']:.4f}
  Recall: {df.iloc[-1]['metrics/recall(B)']:.4f}

Losses:
  Box Loss: {df.iloc[-1]['val/box_loss']:.4f}
  Class Loss: {df.iloc[-1]['val/cls_loss']:.4f}
  DFL Loss: {df.iloc[-1]['val/dfl_loss']:.4f}

Model file: yolo_detector/weights/best.pt
{'='*60}
"""
    
    with open(Path(exp_dir) / 'training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  [OK] Training report: {Path(exp_dir) / 'training_report.txt'}")
    print(report)


def main():
    """Entry point"""
    
    config = {
        'data_yaml': 'data_detection.yaml',
        'model': 'yolov8n.pt',  # yolov8n/s/m/l/x
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'patience': 20,
        'device': '0'  # '0' for GPU, 'cpu' for CPU
    }
    
    # Start training
    train_yolo_detector(**config)
    
    print("\n" + "="*60)
    print("Detection model training finished!")
    print("="*60)
    print("\nNext steps:")
    print("1. View training results: yolo_detector/plots/")
    print("2. Run inference: python predict_detection.py")
    print("3. Integrate into two-stage pipeline")


if __name__ == "__main__":
    main()

