# 🍄 Dataset Fusion Summary

## Data Sources

The final training dataset `mushroom_species_dataset` is a fusion of the following two Roboflow datasets:

> ⚠️ **Note**: There is also a `kaggle_edible&poison_mushroom` dataset in the project, but it is **not used**. It contains:
> - Train: 2,256 images (1,200 edible + 1,056 poisonous)
> - Val: 282 images (150 edible + 132 poisonous)  
> - Test: 282 images (150 edible + 132 poisonous)
> 
> **Why it is unused**:
> - The Kaggle dataset is classification-formatted (edible/poisonous folders), while the fusion pipeline expects YOLO detection data
> - Its species may not match the Roboflow datasets
> - The current fusion pipeline only merges the two Roboflow YOLO datasets

### 1. Edible Mushroom Dataset
- **Source**: Roboflow Universe
- **Link**: https://universe.roboflow.com/mushroom5/edible-mushroom-5
- **Dataset name**: `edible mushroom`
- **Format**: YOLO (detection)
- **Category**: Edible mushrooms
- **Size**:
  - Train: 1,835 images
  - Valid: 524 images
  - Test: 263 images
- **License**: CC BY 4.0

### 2. Non-edible Mushroom Dataset
- **Source**: Roboflow Universe
- **Link**: https://universe.roboflow.com/mushroom-7/non-edible-mushroom-2
- **Dataset name**: `Non-edible Mushroom 2.v1i.yolov8`
- **Format**: YOLO (detection)
- **Category**: Poisonous/non-edible mushrooms
- **Size**:
  - Train: 1,511 images
  - Valid: 429 images
  - Test: 217 images
- **License**: CC BY 4.0

---

## Fusion Pipeline

### Stage 1: Merge YOLO Detection Datasets
**Script**: `ImageModel/FT_YOLO/merge_dataset.py`

Merge the two YOLO datasets into one unified detection dataset:

- **Input**:
  - `edible mushroom/` (class 0: edible)
  - `Non-edible Mushroom 2.v1i.yolov8/` (class 1: poisonous)

- **Output**: `merged_mushroom_dataset/`
  - 2 classes (edible_mushroom, poisonous_mushroom)
  - Train: 3,346 images
  - Valid: 953 images
  - Test: 480 images

- **Process**:
  - Merge images and label files
  - Standardize class IDs (edible=0, poisonous=1)
  - Keep original train/valid/test splits

---

### Stage 2: Detection → Classification Conversion
**Script**: `Dataset/create_classification_dataset.py`

Convert the YOLO detection dataset into a classification dataset by cropping bounding boxes:

- **Input**: `merged_mushroom_dataset/` (YOLO format with bounding boxes)

- **Output**: `mushroom_classification_dataset/`
  - 2 classes (edible, poisonous)
  - Single-mushroom crops from YOLO bounding boxes
  - Train: 6,752 images (3,307 edible + 3,445 poisonous)
  - Valid: 1,984 images (965 edible + 1,019 poisonous)
  - Test: 1,000 images (441 edible + 559 poisonous)

- **Process**:
  - Read YOLO-format bounding boxes
  - Crop each bounding-box region
  - Save in classification layout (per-class folders)

---

### Stage 3: 2-Class → 16-Species Reorganization
**Script**: `Dataset/reorganize_by_species.py`

Reorganize the 2-class classification dataset into a 16-species classification dataset:

- **Input**: `mushroom_classification_dataset/` (2 classes: edible/poisonous)

- **Output**: `mushroom_species_dataset/` ⭐ **Final training dataset**
  - **16 mushroom species**:
    
    **Edible species (8)**:
    1. Amanita-calyptroderma
    2. Armillaria-mellea
    3. Armillaria-tabescens
    4. Artomyces-pyxidatus
    5. Bolbitius-titubans
    6. Boletus-pallidus
    7. Boletus-rex-veris
    8. Cantharellus-californicus
    
    **Poisonous species (8)**:
    9. Ganoderma-tsugae
    10. Leucoagaricus-leucothites
    11. Lycogala-epidendrum
    12. Trametes-gibbosa
    13. Trametes-versicolor
    14. Trichaptum-biforme
    15. Tylopilus-felleus
    16. Tylopilus-rubrobrunneus

- **Counts** (Train/Valid/Test):
  - **Train**: 6,752 images
    - Max: Lycogala-epidendrum (1,305 images)
    - Min: Tylopilus-felleus (238 images)
  
  - **Valid**: 1,984 images
    - Max: Lycogala-epidendrum (397 images)
    - Min: Tylopilus-felleus (74 images)
  
  - **Test**: 1,000 images
    - Max: Lycogala-epidendrum (256 images)
    - Min: Trametes-gibbosa (32 images)

- **Process**:
  - Extract species names from filenames (format: `edible_Species-name_number.jpg`)
  - Reorganize folder structure by species
  - Generate `species_toxicity_mapping.json`

---

## Final Dataset Structure

```
mushroom_species_dataset/
├── species_toxicity_mapping.json  # species-toxicity mapping
├── data_info.txt                  # dataset info
├── train/
│   ├── Amanita-calyptroderma/
│   ├── Armillaria-mellea/
│   ├── ... (16 species folders)
├── valid/
│   ├── Amanita-calyptroderma/
│   ├── ... (16 species folders)
└── test/
    ├── Amanita-calyptroderma/
    ├── ... (16 species folders)
```

---

## Dataset Characteristics

### ✅ Strengths
1. **Multi-source fusion**: Combines strengths of two Roboflow datasets
2. **Fine-grained classes**: Expanded from 2 classes to 16 species
3. **Balanced toxicity groups**: 8 edible vs 8 poisonous species
4. **Standard splits**: Pre-split into train/valid/test

### ⚠️ Challenges
1. **Class imbalance**: 
   - Largest species (Lycogala-epidendrum): 1,305 training images
   - Smallest species (Tylopilus-felleus): 238 training images
   - Ratio about 5.5:1

2. **Limited sources**: 
   - Mainly from Roboflow community datasets
   - Potential lack of diversity (angles, lighting, etc.)
   - ⚠️ **Unused Kaggle dataset**: `kaggle_edible&poison_mushroom` exists but is not merged

---

## Usage

### Train ViT model
```bash
cd ImageModel/FT_ViT
python train_vit_antiovertfit.py --data-dir ../../Dataset/mushroom_species_dataset
```

### Evaluate model
```bash
python evaluate_vit.py --model-path vit_antioverfit/best_model.pth
```

### Compare against baseline
```bash
python compare_baseline_vit.py
```

---

## File Descriptions

| File | Description |
|------|------|
| `merge_dataset.py` | Merge two YOLO datasets |
| `create_classification_dataset.py` | Convert detection dataset to classification |
| `reorganize_by_species.py` | Reorganize 2 classes into 16 species |
| `analyze_species.py` | Analyze species distribution |

---

## Unused Datasets

### Kaggle Mushroom Dataset (`kaggle_edible&poison_mushroom`)

**Status**: ❌ **Not used**

**Size**:
- Train: 2,256 images (1,200 edible + 1,056 poisonous)
- Val: 282 images (150 edible + 132 poisonous)
- Test: 282 images (150 edible + 132 poisonous)

**Format**: Classification layout (edible/poisonous folders)

**Why unused**:
1. **Format mismatch**: Kaggle set is classification-formatted, while the pipeline expects YOLO detection with boxes
2. **Species mismatch risk**: Species may not align with the 16 Roboflow species
3. **Pipeline scope**: Current `merge_dataset.py` only merges the two Roboflow YOLO datasets

**Potential uses**:
- Extra augmentation source
- Needs new fusion script to integrate Kaggle data
- Must verify species alignment with the 16-class setup

---

## License

All datasets use **CC BY 4.0**.

---

**Last updated**: 2025-01-XX

