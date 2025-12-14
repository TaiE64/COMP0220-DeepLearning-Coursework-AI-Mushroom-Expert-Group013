#!/usr/bin/env python3
"""
Benchmark Results Comparison and Visualization
Compare baseline and fine-tuned model performance
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# File paths
BASELINE_RESULTS = "benchmark_results.json"
FINETUNED_RESULTS = "benchmark_chat_results.json"
OUTPUT_DIR = Path("benchmark_comparison")

def load_results(baseline_file, finetuned_file):
    """Load both result files"""
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    with open(finetuned_file, 'r', encoding='utf-8') as f:
        finetuned_data = json.load(f)
    
    baseline = baseline_data['baseline']
    finetuned = finetuned_data['results']
    
    return baseline, finetuned

def calculate_improvements(baseline, finetuned):
    """Calculate improvement percentages"""
    improvements = {}
    
    # BLEU score improvement
    if baseline['avg_bleu'] > 0:
        improvements['bleu'] = ((finetuned['avg_bleu'] - baseline['avg_bleu']) / baseline['avg_bleu']) * 100
    else:
        improvements['bleu'] = float('inf') if finetuned['avg_bleu'] > 0 else 0
    
    # ROUGE-L score improvement
    if baseline['avg_rouge_l'] > 0:
        improvements['rouge_l'] = ((finetuned['avg_rouge_l'] - baseline['avg_rouge_l']) / baseline['avg_rouge_l']) * 100
    else:
        improvements['rouge_l'] = 0
    
    # Perplexity improvement (lower is better)
    if baseline.get('avg_perplexity') and finetuned.get('avg_perplexity'):
        improvements['perplexity'] = ((baseline['avg_perplexity'] - finetuned['avg_perplexity']) / baseline['avg_perplexity']) * 100
    else:
        improvements['perplexity'] = None
    
    # Generation time improvement (lower is better)
    if baseline['avg_generation_time'] > 0:
        improvements['generation_time'] = ((baseline['avg_generation_time'] - finetuned['avg_generation_time']) / baseline['avg_generation_time']) * 100
    else:
        improvements['generation_time'] = 0
    
    return improvements

def plot_metric_comparison(baseline, finetuned, improvements, output_dir):
    """Create bar chart comparing metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison: Baseline vs Fine-tuned', fontsize=16, fontweight='bold')
    
    models = ['Baseline', 'Fine-tuned']
    colors = ['#3498db', '#2ecc71']
    
    # 1. BLEU Score
    ax1 = axes[0, 0]
    bleu_values = [baseline['avg_bleu'], finetuned['avg_bleu']]
    bars1 = ax1.bar(models, bleu_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'BLEU Score Comparison\n(Improvement: {improvements["bleu"]:.1f}%)', 
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(bleu_values) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, bleu_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. ROUGE-L Score
    ax2 = axes[0, 1]
    rouge_values = [baseline['avg_rouge_l'], finetuned['avg_rouge_l']]
    bars2 = ax2.bar(models, rouge_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'ROUGE-L Score Comparison\n(Improvement: {improvements["rouge_l"]:.1f}%)', 
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(rouge_values) * 1.3)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, rouge_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Perplexity (if available)
    ax3 = axes[1, 0]
    if baseline.get('avg_perplexity') and finetuned.get('avg_perplexity'):
        ppl_values = [baseline['avg_perplexity'], finetuned['avg_perplexity']]
        bars3 = ax3.bar(models, ppl_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
        ax3.set_title(f'Perplexity Comparison (Lower is Better)\n(Improvement: {improvements["perplexity"]:.1f}%)', 
                      fontsize=11, fontweight='bold')
        ax3.set_ylim(0, max(ppl_values) * 1.2)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, ppl_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Perplexity data not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Perplexity Comparison', fontsize=11, fontweight='bold')
    
    # 4. Generation Time
    ax4 = axes[1, 1]
    time_values = [baseline['avg_generation_time'], finetuned['avg_generation_time']]
    bars4 = ax4.bar(models, time_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Generation Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Generation Time Comparison (Lower is Better)\n(Improvement: {improvements["generation_time"]:.1f}%)', 
                  fontsize=11, fontweight='bold')
    ax4.set_ylim(0, max(time_values) * 1.2)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars4, time_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: metric_comparison.png")

def plot_score_distribution(baseline, finetuned, output_dir):
    """Plot distribution of BLEU and ROUGE-L scores across samples"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Score Distribution Across Samples', fontsize=16, fontweight='bold')
    
    # BLEU scores
    ax1 = axes[0]
    baseline_bleu = [s for s in baseline['bleu_scores'] if s > 0]
    finetuned_bleu = [s for s in finetuned['bleu_scores'] if s > 0]
    
    ax1.hist(baseline_bleu, bins=10, alpha=0.6, label='Baseline', color='#3498db', edgecolor='black')
    ax1.hist(finetuned_bleu, bins=10, alpha=0.6, label='Fine-tuned', color='#2ecc71', edgecolor='black')
    ax1.set_xlabel('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('BLEU Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # ROUGE-L scores
    ax2 = axes[1]
    ax2.hist(baseline['rouge_l_scores'], bins=10, alpha=0.6, label='Baseline', color='#3498db', edgecolor='black')
    ax2.hist(finetuned['rouge_l_scores'], bins=10, alpha=0.6, label='Fine-tuned', color='#2ecc71', edgecolor='black')
    ax2.set_xlabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('ROUGE-L Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: score_distribution.png")

def plot_generation_time_comparison(baseline, finetuned, output_dir):
    """Plot generation time comparison across samples"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    samples = range(1, len(baseline['generation_times']) + 1)
    
    ax.plot(samples, baseline['generation_times'], 'o-', label='Baseline', 
            color='#3498db', linewidth=2, markersize=8, alpha=0.8)
    ax.plot(samples, finetuned['generation_times'], 's-', label='Fine-tuned', 
            color='#2ecc71', linewidth=2, markersize=8, alpha=0.8)
    
    ax.axhline(y=baseline['avg_generation_time'], color='#3498db', linestyle='--', 
               alpha=0.5, label=f'Baseline Avg: {baseline["avg_generation_time"]:.2f}s')
    ax.axhline(y=finetuned['avg_generation_time'], color='#2ecc71', linestyle='--', 
               alpha=0.5, label=f'Fine-tuned Avg: {finetuned["avg_generation_time"]:.2f}s')
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Generation Time Comparison Across Samples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: generation_time_comparison.png")

def plot_improvement_summary(improvements, output_dir):
    """Plot improvement summary"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['BLEU', 'ROUGE-L', 'Generation Time']
    values = [
        improvements['bleu'],
        improvements['rouge_l'],
        improvements['generation_time']
    ]
    
    # Add perplexity if available
    if improvements['perplexity'] is not None:
        metrics.append('Perplexity')
        values.append(improvements['perplexity'])
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Summary', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%',
                ha='left' if width > 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: improvement_summary.png")

def generate_report(baseline, finetuned, improvements, output_dir):
    """Generate text report"""
    report = f"""
# Benchmark Results Comparison Report

## Overview
This report compares the performance of the Baseline Qwen model and the Fine-tuned Qwen model on mushroom-related QA tasks.

## Dataset
- Number of samples: {baseline['num_samples']}
- Test data: final_training_data.jsonl

## Performance Metrics

### 1. BLEU Score
- **Baseline**: {baseline['avg_bleu']:.4f}
- **Fine-tuned**: {finetuned['avg_bleu']:.4f}
- **Improvement**: {improvements['bleu']:.2f}%

### 2. ROUGE-L Score
- **Baseline**: {baseline['avg_rouge_l']:.4f}
- **Fine-tuned**: {finetuned['avg_rouge_l']:.4f}
- **Improvement**: {improvements['rouge_l']:.2f}%

### 3. Perplexity
"""
    
    if baseline.get('avg_perplexity') and finetuned.get('avg_perplexity'):
        report += f"""- **Baseline**: {baseline['avg_perplexity']:.4f}
- **Fine-tuned**: {finetuned['avg_perplexity']:.4f}
- **Improvement**: {improvements['perplexity']:.2f}% (lower is better)
"""
    else:
        report += "- Data not available\n"
    
    report += f"""
### 4. Generation Time
- **Baseline**: {baseline['avg_generation_time']:.2f} seconds
- **Fine-tuned**: {finetuned['avg_generation_time']:.2f} seconds
- **Improvement**: {improvements['generation_time']:.2f}% (faster)

## Key Findings

1. **BLEU Score**: The fine-tuned model shows {'significant' if improvements['bleu'] > 50 else 'moderate' if improvements['bleu'] > 0 else 'no'} improvement in BLEU score.
   - Baseline average: {baseline['avg_bleu']:.4f}
   - Fine-tuned average: {finetuned['avg_bleu']:.4f}
   - This indicates {'better' if improvements['bleu'] > 0 else 'similar'} n-gram overlap with reference answers.

2. **ROUGE-L Score**: The fine-tuned model demonstrates {'significant' if improvements['rouge_l'] > 50 else 'moderate' if improvements['rouge_l'] > 0 else 'no'} improvement in ROUGE-L score.
   - Baseline average: {baseline['avg_rouge_l']:.4f}
   - Fine-tuned average: {finetuned['avg_rouge_l']:.4f}
   - This suggests {'better' if improvements['rouge_l'] > 0 else 'similar'} longest common subsequence matching.

3. **Generation Speed**: The fine-tuned model is {'significantly faster' if improvements['generation_time'] > 30 else 'faster' if improvements['generation_time'] > 0 else 'similar in speed'}.
   - Baseline average: {baseline['avg_generation_time']:.2f}s
   - Fine-tuned average: {finetuned['avg_generation_time']:.2f}s
   - Speedup: {improvements['generation_time']:.1f}%

## Conclusion

The fine-tuned model shows {'significant improvements' if improvements['rouge_l'] > 30 and improvements['bleu'] > 30 else 'moderate improvements' if improvements['rouge_l'] > 0 or improvements['bleu'] > 0 else 'mixed results'} across multiple metrics:
- Better text quality (BLEU and ROUGE-L scores)
- {'Faster' if improvements['generation_time'] > 0 else 'Similar'} generation speed
- {'Lower' if improvements.get('perplexity') and improvements['perplexity'] > 0 else 'Similar'} perplexity (if available)

The fine-tuning process has {'successfully improved' if improvements['rouge_l'] > 20 else 'moderately improved' if improvements['rouge_l'] > 0 else 'shown limited improvement in'} the model's ability to generate accurate and relevant responses to mushroom-related questions.
"""
    
    with open(output_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  [OK] Saved: comparison_report.md")

def main():
    print("="*80)
    print("Benchmark Results Comparison and Visualization")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n[INFO] Output directory: {OUTPUT_DIR}")
    
    # Load results
    print("\n[1] Loading results...")
    baseline, finetuned = load_results(BASELINE_RESULTS, FINETUNED_RESULTS)
    print(f"  [OK] Baseline model: {baseline['model_name']}")
    print(f"  [OK] Fine-tuned model: {finetuned['model_name']}")
    
    # Calculate improvements
    print("\n[2] Calculating improvements...")
    improvements = calculate_improvements(baseline, finetuned)
    print(f"  [OK] BLEU improvement: {improvements['bleu']:.2f}%")
    print(f"  [OK] ROUGE-L improvement: {improvements['rouge_l']:.2f}%")
    if improvements['perplexity'] is not None:
        print(f"  [OK] Perplexity improvement: {improvements['perplexity']:.2f}%")
    print(f"  [OK] Generation time improvement: {improvements['generation_time']:.2f}%")
    
    # Generate visualizations
    print("\n[3] Generating visualizations...")
    plot_metric_comparison(baseline, finetuned, improvements, OUTPUT_DIR)
    plot_score_distribution(baseline, finetuned, OUTPUT_DIR)
    plot_generation_time_comparison(baseline, finetuned, OUTPUT_DIR)
    plot_improvement_summary(improvements, OUTPUT_DIR)
    
    # Generate report
    print("\n[4] Generating report...")
    generate_report(baseline, finetuned, improvements, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("[OK] Comparison complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()

