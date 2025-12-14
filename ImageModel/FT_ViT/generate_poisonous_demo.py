"""
Generate ViT classification demo images for POISONOUS mushrooms.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import cv2
import json
from PIL import Image


def load_vit_model(model_path, num_classes=16, device='cuda'):
    """Load trained ViT model"""
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'vit_b_16')
    
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    else:
        model = models.vit_b_16(weights=None)
    
    state_dict = checkpoint['model_state_dict']
    has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
    
    if has_dropout:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.hidden_dim, num_classes)
        )
    else:
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def classify_and_draw(model, image_path, class_names, species_toxicity, transform, device, output_dir):
    """Classify and draw result"""
    img_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()
    
    pred_species = class_names[pred_idx]
    toxicity = species_toxicity.get(pred_species, 'unknown')
    
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    if toxicity == 'edible':
        color = (0, 180, 0)
        toxicity_label = "EDIBLE"
    elif toxicity == 'poisonous':
        color = (0, 0, 220)
        toxicity_label = "POISONOUS"
    else:
        color = (128, 128, 128)
        toxicity_label = "UNKNOWN"
    
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, pred_species, (10, 35), font, 0.9, (255, 255, 255), 2)
    label2 = f"{toxicity_label} | Confidence: {conf:.1%}"
    cv2.putText(img, label2, (10, 70), font, 0.7, (255, 255, 255), 2)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    output_file = output_path / Path(image_path).name
    cv2.imwrite(str(output_file), img)
    print(f"[OK] {Path(image_path).name}: {pred_species} ({toxicity_label}) - {conf:.1%}")
    return str(output_file)


def main():
    model_path = 'vit_antioverfit/best_model.pth'
    data_dir = Path('../../Dataset/mushroom_species_dataset')
    output_dir = 'vit_antioverfit/classification_demo'
    
    mapping_file = data_dir / 'species_toxicity_mapping.json'
    with open(mapping_file, 'r') as f:
        species_toxicity = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"\nLoading model: {model_path}")
    model = load_vit_model(model_path, num_classes=16, device=device)
    print(f"[OK] Model loaded")
    
    test_dir = data_dir / 'test'
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Poisonous mushroom images
    poisonous_images = [
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\mushroom_species_dataset\test\Tylopilus-felleus\nonedible_Tylopilus-felleus_150_jpg.rf.b63dd3404d4e5f53b7a15f8087639e9e.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\mushroom_species_dataset\test\Trametes-versicolor\nonedible_Trametes-versicolor_117_jpg.rf.33aaf59f5ab216ed6606e72a2bd76096.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\mushroom_species_dataset\test\Ganoderma-tsugae\nonedible_Ganoderma-tsugae_233_jpg.rf.bf5ec6c126ddb373741c7904b607f457.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\mushroom_species_dataset\test\Ganoderma-tsugae\nonedible_Ganoderma-tsugae_85_jpg.rf.f8d08fc231d8c93ad2930f946e6dbeb5.jpg",
    ]
    
    print(f"\n{'='*60}")
    print("Generating ViT Classification Demo - POISONOUS Mushrooms")
    print(f"{'='*60}\n")
    
    for img_path in poisonous_images:
        if Path(img_path).exists():
            classify_and_draw(model, img_path, class_names, species_toxicity, transform, device, output_dir)


if __name__ == "__main__":
    main()
