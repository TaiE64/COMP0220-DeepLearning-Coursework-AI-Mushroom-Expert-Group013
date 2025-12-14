# 🛡️ Fixing ViT Overfitting

## 📊 Current Issue
Train/val gap is large:

| Metric   | Train | Val | Gap |
|----------|-------|-----|-----|
| Accuracy | 100%  | 83% | 17% |
| Loss     | ~0.02 | ~0.85 | huge |

Severe overfitting. 🚨

## 🔍 Root Causes
1) Capacity vs. data: ViT-B/16 ~86M params vs ~7.5k samples → 11,467:1 params/sample (target < 100:1).  
2) Weak regularization: weight_decay 1e-4, no dropout, no label smoothing, basic aug.

## 💊 Base vs Anti-Overfit

| Technique         | Base     | Anti-Overfit      |
|-------------------|----------|-------------------|
| Learning Rate     | 0.0003   | **0.0001**        |
| Weight Decay      | 0.0001   | **0.01**          |
| Dropout           | None     | **0.3**           |
| Label Smoothing   | None     | **0.1**           |
| Data Augmentation | Basic    | **Stronger**      |
| Scheduler         | ReduceLROnPlateau | **CosineAnnealing** |
| Grad Clipping     | None     | **max_norm=1.0**  |
| Patience          | 10       | **15**            |

## 🚀 How to Use
- Default: `python train_vit_antioverfit.py`
- Stronger reg: `--weight-decay 0.02 --dropout 0.4`
- Tighter early stop: `--patience 10`
- Smaller LR: `--lr 0.00005`
- Resume: `--resume`

## 🔧 Key Changes
- Stronger aug (vertical flip, affine, stronger jitter, grayscale, Cutout, larger rotation).
- Dropout head: `Dropout(0.3) + Linear`.
- Label smoothing: `smoothing=0.1`.
- Weight decay: 0.01 (100× base).
- Scheduler: CosineAnnealingWarmRestarts (periodic restarts, smoother LR).

## 📊 Expected Results
- Before: Train 100%, Val 83%, Gap 17%.
- After anti-overfit: Train 90–92%, Val 85–88%, Gap 2–4%.

## 🎯 Training Tips
- Monitor Gap < 5%; don’t chase 100% train acc—aim highest val acc with small gap.
- Train longer if needed: `--epochs 120`.
- Small data: freeze lower layers `--freeze-layers 8`.
- Multiple seeds:
  ```bash
  for seed in 1 2 3; do
    python train_vit_antioverfit.py --seed $seed
  done
  ```

## 📦 Outputs
- Anti-overfit weights: `ImageModel/FT_ViT/vit_antioverfit/best_model.pth`
- Comparisons/plots: `ImageModel/FT_ViT/vit_antioverfit/evaluation/`
