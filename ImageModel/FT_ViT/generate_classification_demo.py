"""
Generate ViT classification demo images.
Shows classification results with species name and confidence.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import cv2
import json
import numpy as np
from PIL import Image


def load_vit_model(model_path, num_classes=16, device='cuda'):
    """Load trained ViT model (supports anti-overfitting version)"""
    
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
    state_dict = checkpoint['model_state_dict']
    has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
    
    if has_dropout:
        print("   Detected anti-overfitting model (with Dropout)")
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.hidden_dim, num_classes)
        )
    else:
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def classify_and_draw(model, image_path, class_names, species_toxicity, transform, device, output_dir):
    """
    Classify mushroom and draw result on image.
    """
    # Read and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()
    
    pred_species = class_names[pred_idx]
    toxicity = species_toxicity.get(pred_species, 'unknown')
    
    # Read image for drawing
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Set colors and labels
    if toxicity == 'edible':
        color = (0, 180, 0)  # Green
        toxicity_label = "EDIBLE"
    elif toxicity == 'poisonous':
        color = (0, 0, 220)  # Red
        toxicity_label = "POISONOUS"
    else:
        color = (128, 128, 128)
        toxicity_label = "UNKNOWN"
    
    # Draw background box at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Species name
    cv2.putText(img, pred_species, (10, 35), font, 0.9, (255, 255, 255), 2)
    
    # Toxicity and confidence
    label2 = f"{toxicity_label} | Confidence: {conf:.1%}"
    cv2.putText(img, label2, (10, 70), font, 0.7, (255, 255, 255), 2)
    
    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    output_file = output_path / Path(image_path).name
    cv2.imwrite(str(output_file), img)
    print(f"[OK] {Path(image_path).name}: {pred_species} ({toxicity_label}) - {conf:.1%}")
    
    return str(output_file)


def main():
    # Paths
    model_path = 'vit_antioverfit/best_model.pth'
    data_dir = Path('../../Dataset/mushroom_species_dataset')
    output_dir = 'vit_antioverfit/classification_demo'
    
    # Load species-toxicity mapping
    mapping_file = data_dir / 'species_toxicity_mapping.json'
    with open(mapping_file, 'r') as f:
        species_toxicity = json.load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model, checkpoint = load_vit_model(model_path, num_classes=16, device=device)
    print(f"[OK] Model loaded successfully")
    
    # Get class names from test directory
    test_dir = data_dir / 'test'
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"Classes: {len(class_names)}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Select sample images (2 edible + 2 poisonous)
    print(f"\n{'='*60}")
    print("Generating ViT Classification Demo Images")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")
    
    sample_images = []
    
    # Find sample images from different species
    for species in class_names:
        species_dir = test_dir / species
        if species_dir.exists():
            images = list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.png"))
            if images:
                sample_images.append(images[0])  # Take first image
                if len(sample_images) >= 6:  # Get 6 samples
                    break
    
    # Generate demo images
    for img_path in sample_images:
        classify_and_draw(model, img_path, class_names, species_toxicity, transform, device, output_dir)
    
    print(f"\n[DONE] Classification demo images saved to: {output_dir}/")


if __name__ == "__main__":
    main()
