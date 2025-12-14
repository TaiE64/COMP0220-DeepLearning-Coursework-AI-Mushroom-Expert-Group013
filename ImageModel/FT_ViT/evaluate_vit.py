"""
评估ViT蘑菇品种识别模型
Evaluate ViT mushroom species classifier
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

# 加载品种-毒性映射
mapping_file = Path('../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json')
with open(mapping_file, 'r') as f:
    SPECIES_TOXICITY = json.load(f)


def load_vit_model(model_path, num_classes=16, device='cuda'):
    """加载训练好的ViT模型（支持标准版和抗过拟合版）"""
    
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'vit_b_16')
    
    # Build model architecture
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Inspect classifier head structure
    # Anti-overfit uses Sequential(Dropout, Linear)
    # Standard uses Linear
    state_dict = checkpoint['model_state_dict']
    has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
    
    if has_dropout:
        # Anti-overfit version (with Dropout)
        print("   Detected anti-overfitting model (with Dropout)")
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),  # dropout inactive during eval
            nn.Linear(model.hidden_dim, num_classes)
        )
    else:
        # Standard version
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_acc = checkpoint.get('best_acc', 0)
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Model loaded: {model_path}")
    if isinstance(best_acc, (int, float)):
        print(f"   Best val accuracy: {best_acc:.2%}")
    else:
        print(f"   Best val accuracy: N/A")
    print(f"   Epoch: {epoch}")
    print(f"   Model: {model_name}")
    
    return model, checkpoint


def evaluate_model(model_path='vit_classifier/best_model.pth',
                   data_dir='../../Dataset/mushroom_species_dataset',
                   output_dir='vit_classifier/evaluation'):
    """完整评估ViT模型性能"""
    
    print("\n" + "="*70)
    print("ViT 16-Class Mushroom Species Classifier Evaluation")
    print("="*70)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 加载模型
    model, checkpoint = load_vit_model(model_path, device=device)
    
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    test_dataset = datasets.ImageFolder(
        Path(data_dir) / 'test',
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nDataset Info:")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Number of classes: {num_classes}")
    
    # 评估
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算整体准确率
    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall Accuracy: {overall_acc:.2%}")
    
    # 按毒性分类准确率
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
    
    print(f"\nBy Toxicity:")
    print(f"   Edible mushrooms:    {edible_acc:.2%} ({edible_correct}/{edible_total})")
    print(f"   Poisonous mushrooms: {poisonous_acc:.2%} ({poisonous_correct}/{poisonous_total})")
    
    # 每个品种的准确率
    print(f"\nPer-Species Accuracy:")
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
            print(f"   {species_name:40s} ({toxicity:10s}): {species_acc:6.2%} ({count:3d} samples)")
    
    # 保存统计结果
    stats_df = pd.DataFrame(species_stats)
    stats_file = output_dir / 'species_accuracy.csv'
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved species accuracy: {stats_file}")
    
    # 生成分类报告
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    report_file = output_dir / 'classification_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved classification report: {report_file}")
    
    # 绘制混淆矩阵
    print(f"\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ViT 16 Mushroom Species', fontsize=16, pad=20)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('True Species', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_file = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {cm_file}")
    plt.close()
    
    # 绘制归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix - ViT 16 Mushroom Species', fontsize=16, pad=20)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('True Species', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_file = output_dir / 'confusion_matrix_normalized.png'
    plt.savefig(cm_norm_file, dpi=300, bbox_inches='tight')
    print(f"Saved normalized confusion matrix: {cm_norm_file}")
    plt.close()
    
    # 绘制准确率柱状图
    plt.figure(figsize=(14, 8))
    
    # 按毒性排序
    stats_df_sorted = stats_df.sort_values(['toxicity', 'accuracy'], ascending=[False, False])
    colors = ['green' if t == 'edible' else 'red' for t in stats_df_sorted['toxicity']]
    
    bars = plt.bar(range(len(stats_df_sorted)), stats_df_sorted['accuracy'], color=colors, alpha=0.7)
    plt.axhline(y=overall_acc, color='blue', linestyle='--', linewidth=2, label=f'Overall: {overall_acc:.2%}')
    
    plt.xlabel('Mushroom Species', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('ViT Per-Species Classification Accuracy', fontsize=14, pad=20)
    plt.xticks(range(len(stats_df_sorted)), stats_df_sorted['species'], rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Edible'),
        Patch(facecolor='red', alpha=0.7, label='Poisonous'),
        plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label=f'Overall: {overall_acc:.2%}')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    acc_file = output_dir / 'species_accuracy.png'
    plt.savefig(acc_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy bar chart: {acc_file}")
    plt.close()
    
    # 保存完整结果
    results = {
        'overall_accuracy': float(overall_acc),
        'edible_accuracy': float(edible_acc),
        'poisonous_accuracy': float(poisonous_acc),
        'num_test_samples': len(test_dataset),
        'num_classes': num_classes,
        'species_stats': species_stats,
        'model_path': str(model_path),
        'model_name': checkpoint.get('model_name', 'vit_b_16'),
        'best_val_acc': float(checkpoint.get('best_acc', 0))
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results: {results_file}")
    
    # 打印总结
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Overall Accuracy:    {overall_acc:.2%}")
    print(f"Edible Accuracy:     {edible_acc:.2%}")
    print(f"Poisonous Accuracy:  {poisonous_acc:.2%}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ViT mushroom classifier')
    parser.add_argument('--model', type=str, default='vit_classifier/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='../../Dataset/mushroom_species_dataset',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as model directory + /evaluation)')
    args = parser.parse_args()
    
    # 自动设置输出目录
    if args.output_dir is None:
        model_dir = Path(args.model).parent
        output_dir = model_dir / 'evaluation'
    else:
        output_dir = args.output_dir
    
    evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=output_dir
    )

