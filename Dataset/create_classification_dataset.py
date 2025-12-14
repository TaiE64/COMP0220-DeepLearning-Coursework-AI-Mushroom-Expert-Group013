"""
Convert YOLO detection dataset to classification dataset
Read bounding boxes from merged_mushroom_dataset and crop mushroom regions
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# Source dataset (YOLO format)
source_dataset = "merged_mushroom_dataset"

# Target dataset (classification format)
target_dataset = "mushroom_classification_dataset"

# Class mapping
class_names = {
    0: "edible",      # edible mushrooms
    1: "poisonous"    # poisonous mushrooms
}

def create_directory_structure():
    """Create classification dataset directory structure"""
    print("="*60)
    print("Creating classification dataset structure...")
    print("="*60)
    
    splits = ['train', 'valid', 'test']
    for split in splits:
        for class_name in class_names.values():
            dir_path = Path(target_dataset) / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] Created directory: {target_dataset}")

def crop_and_save_mushrooms(split_name):
    """
    Crop and save mushroom images
    
    Args:
        split_name: Dataset split name (train/valid/test)
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} set")
    print(f"{'='*60}")
    
    # Source paths
    images_dir = Path(source_dataset) / split_name / 'images'
    labels_dir = Path(source_dataset) / split_name / 'labels'
    
    if not images_dir.exists():
        print(f"[WARN] Directory not found: {images_dir}")
        return {}
    
    # Get all images
    image_files = list(images_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    stats = {class_name: 0 for class_name in class_names.values()}
    total_crops = 0
    skipped = 0
    
    # Process each image
    for img_file in tqdm(image_files, desc=f"  Cropping {split_name}"):
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            skipped += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Read label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            skipped += 1
            continue
        
        # Read annotations (one image may have multiple mushrooms)
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Process each bounding box
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Parse YOLO format
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to pixel coordinates
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            # Calculate bounding box
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Check bounding box validity
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop mushroom region
            cropped = img[y1:y2, x1:x2]
            
            # Ensure cropped image is valid
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
            
            # Determine save path
            class_name = class_names[class_id]
            output_dir = Path(target_dataset) / split_name / class_name
            
            # Generate filename
            if len(lines) > 1:
                output_file = output_dir / f"{img_file.stem}_crop{idx}.jpg"
            else:
                output_file = output_dir / f"{img_file.stem}.jpg"
            
            # Save cropped image
            cv2.imwrite(str(output_file), cropped)
            
            # Update statistics
            stats[class_name] += 1
            total_crops += 1
    
    # Print statistics
    print(f"\n{split_name.upper()} statistics:")
    print(f"  Total crops: {total_crops}")
    print(f"  Edible mushrooms: {stats['edible']}")
    print(f"  Poisonous mushrooms: {stats['poisonous']}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")
    
    return stats

def print_summary(all_stats):
    """Print overall statistics"""
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    
    print("\nOverall statistics:")
    print("-"*60)
    
    total_edible = sum(stats.get('edible', 0) for stats in all_stats.values())
    total_poisonous = sum(stats.get('poisonous', 0) for stats in all_stats.values())
    
    print(f"Total images: {total_edible + total_poisonous}")
    print(f"  Edible mushrooms: {total_edible}")
    print(f"  Poisonous mushrooms: {total_poisonous}")
    
    print(f"\nDataset location: {os.path.abspath(target_dataset)}")
    
    print("\nSplit statistics:")
    print("-"*60)
    for split, stats in all_stats.items():
        print(f"{split.upper()}:")
        print(f"  Edible: {stats.get('edible', 0)}")
        print(f"  Poisonous: {stats.get('poisonous', 0)}")
        print(f"  Subtotal: {stats.get('edible', 0) + stats.get('poisonous', 0)}")

def main():
    """Main function"""
    print("="*60)
    print("YOLO Detection -> Classification Dataset Converter")
    print("="*60)
    print(f"\nSource dataset: {source_dataset}")
    print(f"Target dataset: {target_dataset}\n")
    
    # Check if source dataset exists
    if not Path(source_dataset).exists():
        print(f"[ERROR] Source dataset not found: {source_dataset}")
        print("Please run merge_dataset.py first")
        return
    
    # 1. Create directory structure
    create_directory_structure()
    
    # 2. Process each split
    all_stats = {}
    for split in ['train', 'valid', 'test']:
        stats = crop_and_save_mushrooms(split)
        all_stats[split] = stats
    
    # 3. Print summary
    print_summary(all_stats)
    
    print("\n" + "="*60)
    print("[OK] Conversion complete! Ready to train classification model")
    print("="*60)

if __name__ == "__main__":
    main()
