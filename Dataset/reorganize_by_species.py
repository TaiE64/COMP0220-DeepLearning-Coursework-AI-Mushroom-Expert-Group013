"""
Reorganize mushroom dataset by species (16-class classification)
From: edible/poisonous folders
To: 16 species folders
"""

import os
import shutil
from pathlib import Path
from collections import Counter

def extract_species(filename):
    """Extract species name from filename"""
    # Format: edible_Species-name_number_xxx.jpg
    parts = filename.split('_')
    if len(parts) >= 2:
        species = parts[1]
        if not species.isdigit():
            return species
    return None

def reorganize_dataset():
    """Reorganize dataset by species"""
    
    source_dir = Path('mushroom_classification_dataset')
    target_dir = Path('mushroom_species_dataset')
    
    print("="*60)
    print("Reorganizing Dataset by Species")
    print("="*60)
    
    # Species to toxicity mapping
    edible_species = [
        'Armillaria-mellea',
        'Bolbitius-titubans',
        'Artomyces-pyxidatus',
        'Armillaria-tabescens',
        'Amanita-calyptroderma',
        'Boletus-rex-veris',
        'Cantharellus-californicus',
        'Boletus-pallidus'
    ]
    
    poisonous_species = [
        'Lycogala-epidendrum',
        'Trametes-versicolor',
        'Trichaptum-biforme',
        'Tylopilus-rubrobrunneus',
        'Ganoderma-tsugae',
        'Leucoagaricus-leucothites',
        'Trametes-gibbosa',
        'Tylopilus-felleus'
    ]
    
    all_species = edible_species + poisonous_species
    
    # Create toxicity mapping file
    toxicity_map = {}
    for species in edible_species:
        toxicity_map[species] = 'edible'
    for species in poisonous_species:
        toxicity_map[species] = 'poisonous'
    
    import json
    mapping_file = target_dir / 'species_toxicity_mapping.json'
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n Processing {split} set...")
        
        # Create target directories for each species
        for species in all_species:
            target_species_dir = target_dir / split / species
            target_species_dir.mkdir(parents=True, exist_ok=True)
        
        # Process edible mushrooms
        edible_dir = source_dir / split / 'edible'
        if edible_dir.exists():
            for img_file in edible_dir.glob('*.jpg'):
                species = extract_species(img_file.stem)
                if species and species in all_species:
                    target_file = target_dir / split / species / img_file.name
                    shutil.copy2(img_file, target_file)
        
        # Process poisonous mushrooms
        poisonous_dir = source_dir / split / 'poisonous'
        if poisonous_dir.exists():
            for img_file in poisonous_dir.glob('*.jpg'):
                species = extract_species(img_file.stem)
                if species and species in all_species:
                    target_file = target_dir / split / species / img_file.name
                    shutil.copy2(img_file, target_file)
        
        # Count images per species
        print(f"\n{split} set statistics:")
        for species in all_species:
            species_dir = target_dir / split / species
            count = len(list(species_dir.glob('*.jpg')))
            toxicity = toxicity_map[species]
            print(f"  {species:40s} : {count:4d} images ({toxicity})")
    
    # Save toxicity mapping
    target_dir.mkdir(exist_ok=True)
    with open(mapping_file, 'w') as f:
        json.dump(toxicity_map, f, indent=2)
    print(f"\nToxicity mapping saved to: {mapping_file}")
    
    # Create data.yaml for reference
    data_yaml = target_dir / 'data_info.txt'
    with open(data_yaml, 'w') as f:
        f.write("Multi-Class Mushroom Species Dataset\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of classes: {len(all_species)}\n")
        f.write(f"Edible species: {len(edible_species)}\n")
        f.write(f"Poisonous species: {len(poisonous_species)}\n\n")
        f.write("Class Index Mapping:\n")
        for i, species in enumerate(sorted(all_species)):
            toxicity = toxicity_map[species]
            f.write(f"  {i:2d}: {species:40s} ({toxicity})\n")
    
    print(f"\nDataset info saved to: {data_yaml}")
    
    print("\n" + "="*60)
    print("Dataset reorganization completed!")
    print("="*60)
    print(f"\nNew dataset location: {target_dir}")
    print(f"Total classes: {len(all_species)}")


if __name__ == "__main__":
    reorganize_dataset()

