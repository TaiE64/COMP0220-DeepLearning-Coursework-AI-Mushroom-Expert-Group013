"""
16-Class Mushroom Species Classification with Vision Transformer (ViT)
Train ViT models for mushroom species classification.
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

# Load species-toxicity mapping
mapping_file = Path('../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json')
with open(mapping_file, 'r') as f:
    SPECIES_TOXICITY = json.load(f)

print("Species-Toxicity Mapping Loaded:")
for species, toxicity in SPECIES_TOXICITY.items():
    print(f"  {species:40s} -> {toxicity}")


def train_vit_classifier(
    data_dir='../../Dataset/mushroom_species_dataset',
    output_dir='vit_classifier',
    model_name='vit_b_16',  # ViT-B/16
    epochs=50,
    batch_size=32,
    lr=0.0003,  # ViT typically uses smaller LR
    patience=10,
    resume=False
):
    """Train 16-class species classifier using Vision Transformer"""
    
    print("\n" + "="*70)
    print("16-Class Mushroom Species Classification with ViT")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # ViT preprocessing (224x224 standard input)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    
    # Load datasets
    data_dir = Path(data_dir)
    image_datasets = {
        x: datasets.ImageFolder(data_dir / x, data_transforms[x])
        for x in ['train', 'valid', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
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
    print(f"  Class names: {class_names}")
    
    # Build ViT model
    print(f"\nCreating Vision Transformer model: {model_name}")
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        # ViT classification head is model.heads.head
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights='IMAGENET1K_V1')
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights='IMAGENET1K_V1')
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # ViT typically uses AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0001,  # L2 regularization
        betas=(0.9, 0.999)
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
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
            
            # Load history
            history_file = output_dir / 'history.csv'
            if history_file.exists():
                history_df = pd.read_csv(history_file)
                history = history_df.to_dict('records')
            
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best accuracy so far: {best_acc:.2%}")
    
    # Training config
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: 0.0001")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: ReduceLROnPlateau")
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
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Log history
        epoch_time = time.time() - epoch_start_time
        history.append({
            'train_loss': train_loss,
            'train_acc': train_acc.item() * 100,
            'val_loss': val_loss,
            'val_acc': val_acc.item() * 100
        })
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
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
            print(f"*** New best model saved! Acc: {best_acc:.4f} ***")
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
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_file = output_dir / 'history.csv'
    history_df.to_csv(history_file, index=False)
    print(f"\nTraining history saved: {history_file}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['train_acc'], label='Train Acc')
    plt.plot(history_df['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_file = plots_dir / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved: {plot_file}")
    plt.close()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on Test Set")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_corrects.double() / dataset_sizes['test']
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Save results
    results = {
        'test_acc': test_acc.item() * 100,
        'best_val_acc': best_acc.item() * 100,
        'num_classes': num_classes,
        'class_names': class_names,
        'model': model_name
    }
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_file}")
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train ViT for 16-class mushroom species classification')
    parser.add_argument('--model', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_b_32', 'vit_l_16'],
                       help='ViT model variant')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='vit_classifier',
                       help='Output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    train_vit_classifier(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        resume=args.resume
    )


if __name__ == '__main__':
    main()

