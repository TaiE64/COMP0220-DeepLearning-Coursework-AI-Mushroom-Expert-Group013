"""
ViT Training with Strong Anti-Overfitting Regularization
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np

# Load species-toxicity mapping
mapping_file = Path('../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json')
with open(mapping_file, 'r') as f:
    SPECIES_TOXICITY = json.load(f)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy to reduce overfitting"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_pred = torch.log_softmax(pred, dim=1)
        loss = -log_pred.sum(dim=1).mean()
        nll = torch.nn.functional.nll_loss(log_pred, target, reduction='mean')
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll


class Cutout:
    """Cutout augmentation - randomly masks image regions"""
    def __init__(self, n_holes=1, length=56):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


def train_vit_antioverfitting(
    data_dir='../../Dataset/mushroom_species_dataset',
    output_dir='vit_antioverfit',
    model_name='vit_b_16',
    epochs=100,
    batch_size=32,
    lr=0.0001,           # lower LR
    weight_decay=0.01,   # stronger L2
    dropout_rate=0.3,    # add dropout
    label_smoothing=0.1, # label smoothing
    patience=15,         # higher patience
    resume=False
):
    """Train ViT with strong anti-overfitting setup"""
    
    print("\n" + "="*70)
    print("🛡️  ViT Training - ANTI-OVERFITTING MODE")
    print("="*70)
    print("🔧 Anti-Overfitting Techniques Applied:")
    print(f"   ✅ Stronger Data Augmentation (RandAugment, Cutout)")
    print(f"   ✅ Higher Weight Decay: {weight_decay}")
    print(f"   ✅ Dropout Rate: {dropout_rate}")
    print(f"   ✅ Label Smoothing: {label_smoothing}")
    print(f"   ✅ Lower Learning Rate: {lr}")
    print(f"   ✅ Early Stopping: patience={patience}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 🔥 Stronger data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # more aggressive crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # vertical flip
            transforms.RandomRotation(30),         # larger rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # translate
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # stronger jitter
            transforms.RandomGrayscale(p=0.1),          # random grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Cutout(n_holes=1, length=56),               # Cutout
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load dataset
    data_dir = Path(data_dir)
    image_datasets = {
        x: datasets.ImageFolder(data_dir / x, data_transforms[x])
        for x in ['train', 'valid', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                     shuffle=(x=='train'), num_workers=4,
                     drop_last=(x=='train'))  # drop last incomplete batch for train
        for x in ['train', 'valid', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"\nDataset Info:")
    print(f"  Train: {dataset_sizes['train']} images")
    print(f"  Valid: {dataset_sizes['valid']} images")
    print(f"  Test:  {dataset_sizes['test']} images")
    print(f"  Classes: {num_classes}")
    print(f"  Samples per class: ~{dataset_sizes['train']//num_classes}")
    
    # Build ViT model
    print(f"\nCreating Vision Transformer model: {model_name}")
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        hidden_dim = model.hidden_dim
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights='IMAGENET1K_V1')
        hidden_dim = model.hidden_dim
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights='IMAGENET1K_V1')
        hidden_dim = model.hidden_dim
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 🔥 Replace head with Dropout
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    )
    
    model = model.to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  With dropout: {dropout_rate}")
    
    # 🔥 Label smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    # 🔥 AdamW optimizer + higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # strong L2 regularization
        betas=(0.9, 0.999)
    )
    
    # 🔥 Cosine annealing restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,   # restart every 10 epochs
        T_mult=2, # double cycle
        eta_min=1e-6
    )
    
    # Resume training
    start_epoch = 0
    best_acc = 0.0
    history = []
    
    if resume:
        last_model_path = output_dir / 'last_model.pth'
        if last_model_path.exists():
            print(f"\nResuming from {last_model_path}")
            checkpoint = torch.load(last_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            
            history_file = output_dir / 'history.csv'
            if history_file.exists():
                history_df = pd.read_csv(history_file)
                history = history_df.to_dict('records')
                print(f"  📊 Loaded training history from: {history_file.absolute()}")
            else:
                print(f"  ⚠️  History file not found: {history_file.absolute()}")
            
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best accuracy so far: {best_acc:.2%}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay} (Strong!)")
    print(f"  Dropout: {dropout_rate}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: CosineAnnealingWarmRestarts")
    print(f"  Early stopping patience: {patience}")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            
            # 🔥 Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / dataset_sizes['train']
        train_acc = running_corrects.double() / dataset_sizes['train']
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / dataset_sizes['valid']
        val_acc = val_running_corrects.double() / dataset_sizes['valid']
        
        scheduler.step()
        
        # Record history
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'train_loss': train_loss,
            'train_acc': train_acc.item() * 100,
            'val_loss': val_loss,
            'val_acc': val_acc.item() * 100,
            'lr': current_lr,
                'gap': (train_acc.item() - val_acc.item()) * 100  # overfitting gap
        })
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Gap: {(train_acc-val_acc)*100:.2f}% | LR: {current_lr:.6f}")
        print(f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_names': class_names,
                'model_name': model_name
            }, best_model_path)
            print(f"✅ New best model saved! Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save last model
        last_model_path = output_dir / 'last_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'class_names': class_names,
            'model_name': model_name
        }, last_model_path)
        
        # Save history
        history_df = pd.DataFrame(history)
        history_file = output_dir / 'history.csv'
        history_df.to_csv(history_file, index=False)
        print(f"📊 Training history saved: {history_file.absolute()}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Plot curves
    history_df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history_df['train_loss'], label='Train Loss')
    axes[0, 0].plot(history_df['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history_df['train_acc'], label='Train Acc')
    axes[0, 1].plot(history_df['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Overfitting gap curve
    axes[1, 0].plot(history_df['gap'], label='Overfitting Gap', color='red')
    axes[1, 0].axhline(y=0, color='green', linestyle='--', label='No Gap')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train Acc - Val Acc (%)')
    axes[1, 0].set_title('Overfitting Gap (Lower is Better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate curve
    axes[1, 1].plot(history_df['lr'], label='Learning Rate', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = plots_dir / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved: {plot_file}")
    plt.close()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train ViT with anti-overfitting')
    parser.add_argument('--model', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_b_32', 'vit_l_16'],
                       help='ViT model variant')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001, lower than standard)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01, stronger)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='vit_antioverfit',
                       help='Output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    train_vit_antioverfitting(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        output_dir=args.output_dir,
        resume=args.resume
    )


if __name__ == '__main__':
    main()




