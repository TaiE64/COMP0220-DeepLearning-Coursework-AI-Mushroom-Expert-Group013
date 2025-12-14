#!/usr/bin/env python3
"""
Compare Baseline ViT vs Fine-tuned ViT Model
Baseline: ImageNet pretrained ViT (no fine-tuning)
Fine-tuned: ViT fine-tuned on mushroom species dataset
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Load species-toxicity mapping
mapping_file = Path('../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json')
with open(mapping_file, 'r') as f:
    SPECIES_TOXICITY = json.load(f)

NUM_CLASSES = len(SPECIES_TOXICITY)


def load_baseline_vit(model_name='vit_b_16', device='cuda'):
    """Load Baseline ViT (ImageNet pretrained, no fine-tuning)"""
    print(f"\nLoading Baseline ViT ({model_name})...")
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')  # ImageNet pretrained weights
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights='IMAGENET1K_V1')
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Replace classification head for 16 classes
    model.heads.head = nn.Linear(model.hidden_dim, NUM_CLASSES)
    
    model = model.to(device)
    model.eval()
    
    print("  ✅ Baseline ViT loaded (ImageNet pretrained, no fine-tuning)")
    return model


def load_finetuned_vit(model_path, device='cuda'):
    """Load Fine-tuned ViT model"""
    print(f"\nLoading Fine-tuned ViT: {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'vit_b_16')
    
    # Create model architecture
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Check if it's anti-overfitting version
    state_dict = checkpoint['model_state_dict']
    has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
    
    if has_dropout:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.hidden_dim, NUM_CLASSES)
        )
    else:
        model.heads.head = nn.Linear(model.hidden_dim, NUM_CLASSES)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_acc = checkpoint.get('best_acc', 0)
    print(f"  ✅ Fine-tuned ViT loaded")
    print(f"  Best validation accuracy during training: {best_acc:.2%}")
    
    return model, checkpoint


def evaluate_model(model, test_loader, class_names, device, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    overall_acc = (all_preds == all_labels).mean()
    
    # Calculate accuracy by toxicity
    edible_correct = 0
    edible_total = 0
    poisonous_correct = 0
    poisonous_total = 0
    
    for i, label in enumerate(all_labels):
        species_name = class_names[label]
        toxicity = SPECIES_TOXICITY.get(species_name, 'unknown')
        
        if toxicity == 'edible':
            edible_total += 1
            if all_preds[i] == label:
                edible_correct += 1
        elif toxicity == 'poisonous':
            poisonous_total += 1
            if all_preds[i] == label:
                poisonous_correct += 1
    
    edible_acc = edible_correct / edible_total if edible_total > 0 else 0
    poisonous_acc = poisonous_correct / poisonous_total if poisonous_total > 0 else 0
    
    # Per-species accuracy
    species_stats = []
    for i, species_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            species_acc = (all_preds[mask] == all_labels[mask]).mean()
            toxicity = SPECIES_TOXICITY.get(species_name, 'unknown')
            count = mask.sum()
            species_stats.append({
                'species': species_name,
                'toxicity': toxicity,
                'accuracy': float(species_acc),
                'count': int(count)
            })
    
    results = {
        'overall_accuracy': float(overall_acc),
        'edible_accuracy': float(edible_acc),
        'poisonous_accuracy': float(poisonous_acc),
        'species_stats': species_stats,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }
    
    return results


def compare_results(baseline_results, finetuned_results, class_names, output_dir):
    """Compare and visualize results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 Baseline vs Fine-tuned ViT Comparison Results")
    print("="*80)
    
    # Overall metrics comparison
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Fine-tuned':<20} {'Improvement':<15}")
    print("-" * 85)
    
    metrics = [
        ('Overall Accuracy', 'overall_accuracy', 'higher'),
        ('Edible Accuracy', 'edible_accuracy', 'higher'),
        ('Poisonous Accuracy', 'poisonous_accuracy', 'higher'),
    ]
    
    for metric_name, metric_key, better in metrics:
        baseline_val = baseline_results[metric_key]
        finetuned_val = finetuned_results[metric_key]
        
        if better == 'higher':
            improvement = ((finetuned_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvement_str = f"+{improvement:.2f}%"
        else:
            improvement = ((baseline_val - finetuned_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvement_str = f"{improvement:.2f}%"
        
        print(f"{metric_name:<30} {baseline_val:<20.2%} {finetuned_val:<20.2%} {improvement_str:<15}")
    
    # Per-species comparison
    print(f"\n{'Species':<40} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 85)
    
    baseline_stats = {s['species']: s['accuracy'] for s in baseline_results['species_stats']}
    finetuned_stats = {s['species']: s['accuracy'] for s in finetuned_results['species_stats']}
    
    # 使用两个模型的物种并集，确保所有物种都被包含
    all_species = set(baseline_stats.keys()) | set(finetuned_stats.keys())
    
    improvements = []
    for species in sorted(all_species):  # 排序确保一致性
        baseline_acc = baseline_stats.get(species, 0)  # 如果没有则默认为 0
        finetuned_acc = finetuned_stats.get(species, 0)
        
        # 计算改进百分比，避免除以零
        if baseline_acc > 0:
            improvement = ((finetuned_acc - baseline_acc) / baseline_acc * 100)
        elif finetuned_acc > 0:
            improvement = float('inf')  # baseline 为 0，fine-tuned 有值，表示无穷大改进
        else:
            improvement = 0  # 两者都为 0
        
        improvements.append({
            'species': species,
            'baseline': baseline_acc,
            'finetuned': finetuned_acc,
            'improvement': improvement if improvement != float('inf') else 9999  # 用大数值代替 inf
        })
        
        improvement_str = f"+∞%" if improvement == float('inf') else f"{improvement:+.2f}%"
        print(f"{species:<40} {baseline_acc:<15.2%} {finetuned_acc:<15.2%} {improvement_str}")
    
    # Save comparison results
    comparison_results = {
        'baseline': baseline_results,
        'finetuned': finetuned_results,
        'improvements': improvements,
        'summary': {
            'overall_improvement': ((finetuned_results['overall_accuracy'] - baseline_results['overall_accuracy']) / baseline_results['overall_accuracy'] * 100) if baseline_results['overall_accuracy'] > 0 else 0,
            'edible_improvement': ((finetuned_results['edible_accuracy'] - baseline_results['edible_accuracy']) / baseline_results['edible_accuracy'] * 100) if baseline_results['edible_accuracy'] > 0 else 0,
            'poisonous_improvement': ((finetuned_results['poisonous_accuracy'] - baseline_results['poisonous_accuracy']) / baseline_results['poisonous_accuracy'] * 100) if baseline_results['poisonous_accuracy'] > 0 else 0,
        }
    }
    
    results_file = output_dir / 'baseline_vs_finetuned_comparison.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Comparison results saved to: {results_file}")
    
    # Plot comparison charts
    plot_comparison(improvements, output_dir)
    
    return comparison_results


def plot_comparison(improvements, output_dir):
    """Plot comparison charts"""
    print("\nGenerating comparison charts...")
    
    df = pd.DataFrame(improvements)
    df = df.sort_values('improvement', ascending=False)
    
    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['baseline'], width, label='Baseline ViT', alpha=0.7, color='lightblue')
    bars2 = ax.bar(x + width/2, df['finetuned'], width, label='Fine-tuned ViT', alpha=0.7, color='lightgreen')
    
    ax.set_xlabel('Mushroom Species', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Baseline vs Fine-tuned ViT: Per-Species Accuracy Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['species'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_file = output_dir / 'accuracy_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved accuracy comparison chart: {plot_file}")
    plt.close()
    
    # 2. Improvement magnitude bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['green' if imp > 0 else 'red' for imp in df['improvement']]
    bars = ax.bar(range(len(df)), df['improvement'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mushroom Species', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Fine-tuned ViT Improvement over Baseline (by Species)', fontsize=14, pad=20)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['species'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    improvement_file = output_dir / 'improvement_comparison.png'
    plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved improvement chart: {improvement_file}")
    plt.close()
    
    # 3. Confusion matrix comparison (optional, may be slow for large datasets)
    print("  💡 Tip: For confusion matrix comparison, run evaluation script separately")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare Baseline vs Fine-tuned ViT')
    parser.add_argument('--finetuned-model', type=str, default='vit_classifier/best_model.pth',
                       help='Path to fine-tuned model')
    parser.add_argument('--baseline-model', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_b_32', 'vit_l_16'],
                       help='Baseline model architecture')
    parser.add_argument('--data-dir', type=str, default='../../Dataset/mushroom_species_dataset',
                       help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='vit_classifier/comparison',
                       help='Output directory for comparison results')
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 Baseline vs Fine-tuned ViT Comparison Test")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data preprocessing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_dataset = datasets.ImageFolder(
        Path(args.data_dir) / 'test',
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    print(f"\nTest set: {len(test_dataset)} samples")
    print(f"Number of classes: {len(class_names)}")
    
    # Evaluate Baseline model
    print("\n" + "="*80)
    print("1️⃣  Evaluating Baseline ViT (ImageNet Pretrained)")
    print("="*80)
    baseline_model = load_baseline_vit(args.baseline_model, device)
    baseline_results = evaluate_model(
        baseline_model, test_loader, class_names, device, "Baseline ViT"
    )
    
    print(f"\nBaseline ViT Results:")
    print(f"  Overall Accuracy: {baseline_results['overall_accuracy']:.2%}")
    print(f"  Edible Accuracy: {baseline_results['edible_accuracy']:.2%}")
    print(f"  Poisonous Accuracy: {baseline_results['poisonous_accuracy']:.2%}")
    
    # Clear GPU memory
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate Fine-tuned model
    print("\n" + "="*80)
    print("2️⃣  Evaluating Fine-tuned ViT")
    print("="*80)
    finetuned_model, checkpoint = load_finetuned_vit(args.finetuned_model, device)
    finetuned_results = evaluate_model(
        finetuned_model, test_loader, class_names, device, "Fine-tuned ViT"
    )
    
    print(f"\nFine-tuned ViT Results:")
    print(f"  Overall Accuracy: {finetuned_results['overall_accuracy']:.2%}")
    print(f"  Edible Accuracy: {finetuned_results['edible_accuracy']:.2%}")
    print(f"  Poisonous Accuracy: {finetuned_results['poisonous_accuracy']:.2%}")
    
    # Compare results
    print("\n" + "="*80)
    print("3️⃣  Generating Comparison Report")
    print("="*80)
    comparison_results = compare_results(
        baseline_results, finetuned_results, class_names, args.output_dir
    )
    
    print("\n" + "="*80)
    print("✅ Comparison test completed!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nKey Improvements:")
    summary = comparison_results['summary']
    print(f"  Overall accuracy improvement: {summary['overall_improvement']:+.2f}%")
    print(f"  Edible accuracy improvement: {summary['edible_improvement']:+.2f}%")
    print(f"  Poisonous accuracy improvement: {summary['poisonous_improvement']:+.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()

