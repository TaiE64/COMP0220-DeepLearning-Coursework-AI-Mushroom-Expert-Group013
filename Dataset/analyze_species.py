"""Analyze mushroom species distribution in the dataset."""

from pathlib import Path
from collections import Counter
import re

def extract_species(filename):
    """Extract species name from filename.

    Expected format: edible_Species-name_number_xxx.jpg
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        # First part is edible/nonedible, second part is species name
        species = parts[1]
        # Species name usually contains hyphen, e.g., Amanita-calyptroderma
        # Skip pure numeric parts
        if not species.isdigit():
            return species
    return None

# Analyze edible mushrooms
edible_dir = Path('mushroom_classification_dataset/train/edible')
edible_species = Counter()
for f in edible_dir.glob('*.jpg'):
    species = extract_species(f.stem)
    if species:
        edible_species[species] += 1

print("="*60)
print("Edible Mushroom Species")
print("="*60)
for species, count in edible_species.most_common():
    print(f"{species:40s} : {count:4d} images")

print(f"\nTotal: {len(edible_species)} species, {sum(edible_species.values())} images")

# Analyze poisonous mushrooms
poisonous_dir = Path('mushroom_classification_dataset/train/poisonous')
poisonous_species = Counter()
for f in poisonous_dir.glob('*.jpg'):
    species = extract_species(f.stem)
    if species:
        poisonous_species[species] += 1

print("\n" + "="*60)
print("Poisonous Mushroom Species")
print("="*60)
for species, count in poisonous_species.most_common():
    print(f"{species:40s} : {count:4d} images")

print(f"\nTotal: {len(poisonous_species)} species, {sum(poisonous_species.values())} images")

# Summary
print("\n" + "="*60)
print("Dataset Summary")
print("="*60)
print(f"Total species: {len(edible_species) + len(poisonous_species)}")
print(f"Edible species: {len(edible_species)}")
print(f"Poisonous species: {len(poisonous_species)}")
print(f"Total images: {sum(edible_species.values()) + sum(poisonous_species.values())}")

