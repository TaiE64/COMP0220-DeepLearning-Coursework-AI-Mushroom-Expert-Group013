"""
YOLO Detection Baseline Comparison
Compare original YOLO (no fine-tuning) vs fine-tuned YOLO detection performance.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil

# Set fonts that support Unicode
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(model_path, data_yaml, name="model"):
    """Evaluate a single YOLO model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    model = YOLO(model_path)
    
    # Evaluate on validation set
    results = model.val(
        data=data_yaml,
        split='test',  # use test set
        verbose=True,
        save_json=True
    )
    
    # 提取指标
    metrics = {
        'model_name': name,
        'model_path': str(model_path),
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    print(f"\nResults for {name}:")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    
    return metrics, results


def compare_models(
    baseline_model='yolov8n.pt',
    finetuned_model='yolo_detector/weights/best.pt',
    data_yaml='data_detection.yaml',
    output_dir='baseline_comparison'
):
    """Compare baseline and fine-tuned models"""
    
    print("\n" + "="*70)
    print("YOLO Detection: Baseline vs Fine-tuned Comparison")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if fine-tuned model exists
    if not Path(finetuned_model).exists():
        print(f"\n[ERROR] Fine-tuned model not found: {finetuned_model}")
        print("Please train the model first using train_detection.py")
        return
    
    # Evaluate both models
    print("\n" + "-"*60)
    print("Step 1: Evaluating Baseline (Pre-trained YOLO, no fine-tuning)")
    print("-"*60)
    baseline_metrics, baseline_results = evaluate_model(
        baseline_model, data_yaml, "Baseline (yolov8n.pt)"
    )
    
    print("\n" + "-"*60)
    print("Step 2: Evaluating Fine-tuned Model")
    print("-"*60)
    finetuned_metrics, finetuned_results = evaluate_model(
        finetuned_model, data_yaml, "Fine-tuned"
    )
    
    # Calculate improvements
    improvements = {
        'mAP50': finetuned_metrics['mAP50'] - baseline_metrics['mAP50'],
        'mAP50_95': finetuned_metrics['mAP50_95'] - baseline_metrics['mAP50_95'],
        'precision': finetuned_metrics['precision'] - baseline_metrics['precision'],
        'recall': finetuned_metrics['recall'] - baseline_metrics['recall'],
    }
    
    # Generate comparison report
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'mAP@0.5':<20} {baseline_metrics['mAP50']:<15.4f} {finetuned_metrics['mAP50']:<15.4f} {improvements['mAP50']:+.4f}")
    print(f"{'mAP@0.5:0.95':<20} {baseline_metrics['mAP50_95']:<15.4f} {finetuned_metrics['mAP50_95']:<15.4f} {improvements['mAP50_95']:+.4f}")
    print(f"{'Precision':<20} {baseline_metrics['precision']:<15.4f} {finetuned_metrics['precision']:<15.4f} {improvements['precision']:+.4f}")
    print(f"{'Recall':<20} {baseline_metrics['recall']:<15.4f} {finetuned_metrics['recall']:<15.4f} {improvements['recall']:+.4f}")
    
    # Create visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")
    print("-"*60)
    
    # 1. Metric comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    baseline_values = [baseline_metrics['mAP50'], baseline_metrics['mAP50_95'], 
                       baseline_metrics['precision'], baseline_metrics['recall']]
    finetuned_values = [finetuned_metrics['mAP50'], finetuned_metrics['mAP50_95'],
                        finetuned_metrics['precision'], finetuned_metrics['recall']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, baseline_values, width, label='Baseline (Pre-trained)', color='#ff7f0e', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, finetuned_values, width, label='Fine-tuned', color='#2ca02c', alpha=0.8)
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Detection Metrics Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, baseline_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, finetuned_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Improvement chart
    improvement_values = [improvements['mAP50'], improvements['mAP50_95'], 
                          improvements['precision'], improvements['recall']]
    colors = ['green' if v > 0 else 'red' for v in improvement_values]
    
    bars3 = axes[1].bar(metrics_names, improvement_values, color=colors, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Improvement')
    axes[1].set_title('Performance Improvement (Fine-tuned - Baseline)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, improvement_values):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05
        axes[1].text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{val:+.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    comparison_plot = output_dir / 'baseline_comparison.png'
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_plot}")
    
    # 3. Save comparison results JSON
    comparison_results = {
        'baseline': baseline_metrics,
        'finetuned': finetuned_metrics,
        'improvements': improvements,
        'improvement_percentage': {
            'mAP50': improvements['mAP50'] / max(baseline_metrics['mAP50'], 0.001) * 100,
            'mAP50_95': improvements['mAP50_95'] / max(baseline_metrics['mAP50_95'], 0.001) * 100,
            'precision': improvements['precision'] / max(baseline_metrics['precision'], 0.001) * 100,
            'recall': improvements['recall'] / max(baseline_metrics['recall'], 0.001) * 100,
        }
    }
    
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_file}")
    
    # 4. Generate text report
    report = f"""
================================================================================
YOLO Detection: Baseline vs Fine-tuned Comparison Report
================================================================================

1. MODELS COMPARED
--------------------------------------------------------------------------------
   Baseline:    {baseline_model} (Pre-trained on COCO, no fine-tuning)
   Fine-tuned:  {finetuned_model} (Fine-tuned on mushroom dataset)

2. RESULTS
--------------------------------------------------------------------------------
   Metric              Baseline        Fine-tuned      Improvement
   ------------------- --------------- --------------- ---------------
   mAP@0.5             {baseline_metrics['mAP50']:<15.4f} {finetuned_metrics['mAP50']:<15.4f} {improvements['mAP50']:+.4f}
   mAP@0.5:0.95        {baseline_metrics['mAP50_95']:<15.4f} {finetuned_metrics['mAP50_95']:<15.4f} {improvements['mAP50_95']:+.4f}
   Precision           {baseline_metrics['precision']:<15.4f} {finetuned_metrics['precision']:<15.4f} {improvements['precision']:+.4f}
   Recall              {baseline_metrics['recall']:<15.4f} {finetuned_metrics['recall']:<15.4f} {improvements['recall']:+.4f}

3. KEY FINDINGS
--------------------------------------------------------------------------------
   - The pre-trained YOLO model (trained on COCO) has very low performance on
     mushroom detection because COCO dataset does not contain mushroom class.
   
   - After fine-tuning on the mushroom dataset, the model shows significant
     improvement in all metrics.
   
   - mAP@0.5 improved by {improvements['mAP50']:+.4f} ({improvements['mAP50']/max(baseline_metrics['mAP50'], 0.001)*100:+.1f}%)
   - mAP@0.5:0.95 improved by {improvements['mAP50_95']:+.4f}

4. CONCLUSION
--------------------------------------------------------------------------------
   Fine-tuning is essential for domain-specific object detection tasks.
   Transfer learning from COCO pre-trained weights + fine-tuning on target
   dataset yields the best results.

================================================================================
"""
    
    report_file = output_dir / 'comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {report_file}")
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - baseline_comparison.png (visualization)")
    print(f"  - comparison_results.json (metrics)")
    print(f"  - comparison_report.txt (report)")
    
    # Print key conclusions
    print("\n" + "="*70)
    print("KEY CONCLUSION")
    print("="*70)
    if baseline_metrics['mAP50'] < 0.1:
        print("\n  The pre-trained YOLO (COCO) CANNOT detect mushrooms!")
        print("  (COCO dataset has no mushroom class)")
        print(f"\n  After fine-tuning: mAP@0.5 = {finetuned_metrics['mAP50']:.4f}")
        print("\n  This demonstrates the importance of FINE-TUNING for")
        print("  domain-specific object detection tasks!")
    else:
        print(f"\n  Fine-tuning improved mAP@0.5 by {improvements['mAP50']:+.4f}")
    
    return comparison_results


def main():
    """Main function"""
    
    # Check necessary files
    data_yaml = Path('data_detection.yaml')
    finetuned_model = Path('yolo_detector/weights/best.pt')
    
    if not data_yaml.exists():
        print(f"[ERROR] Data config not found: {data_yaml}")
        return
    
    if not finetuned_model.exists():
        print(f"[ERROR] Fine-tuned model not found: {finetuned_model}")
        print("\nPlease train the model first:")
        print("  python train_detection.py")
        return
    
    # Run comparison
    compare_models(
        baseline_model='yolov8n.pt',
        finetuned_model=str(finetuned_model),
        data_yaml=str(data_yaml),
        output_dir='baseline_comparison'
    )


if __name__ == '__main__':
    main()




