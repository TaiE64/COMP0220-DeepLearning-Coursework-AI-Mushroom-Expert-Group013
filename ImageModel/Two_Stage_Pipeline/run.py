"""
Two-stage mushroom detection & species classification pipeline (ViT)
Stage 1: YOLO detects mushroom locations
Stage 2: ViT classifies species and toxicity (16 species)

ViT version achieves higher accuracy than the ResNet version (92.4% vs 81%)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import time


class MushroomViTPipeline:
    """Two-stage detection + ViT species classification pipeline"""
    
    def __init__(self, 
                 detector_path='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                 classifier_path='../FT_ViT/vit_antioverfit/best_model.pth',
                 mapping_file='../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json'):
        """
        Initialize pipeline.
        
        Args:
            detector_path: YOLO detector weights path
            classifier_path: ViT species classifier path (16 classes)
            mapping_file: species-toxicity mapping file
        """
        print("="*70)
        print("Two-Stage Mushroom Detection & Classification Pipeline")
        print("Stage 1: YOLO Detection | Stage 2: ViT Classification")
        print("="*70)
        
        # Load species-toxicity mapping
        print(f"\n[0/3] Loading species-toxicity mapping: {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.species_toxicity = json.load(f)
        print(f"  OK - Loaded {len(self.species_toxicity)} species mappings")
        
        # Load YOLO detector
        print(f"\n[1/3] Loading YOLO detector: {detector_path}")
        if not Path(detector_path).exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        self.detector = YOLO(detector_path)
        print("  OK - YOLO detector loaded")
        
        # Load ViT classifier (16 classes)
        print(f"\n[2/3] Loading ViT species classifier: {classifier_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(classifier_path, map_location=self.device)
        
        # Get model config
        model_name = checkpoint.get('model_name', 'vit_b_16')
        num_classes = len(self.species_toxicity)  # 16 species
        
        # Create ViT model
        print(f"  Model architecture: {model_name}")
        if model_name == 'vit_b_16':
            self.classifier = models.vit_b_16(weights=None)
        elif model_name == 'vit_b_32':
            self.classifier = models.vit_b_32(weights=None)
        elif model_name == 'vit_l_16':
            self.classifier = models.vit_l_16(weights=None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Check anti-overfitting variant (with Dropout)
        state_dict = checkpoint['model_state_dict']
        has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
        
        if has_dropout:
            print("  Detected anti-overfitting model (with Dropout)")
            self.classifier.heads.head = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(self.classifier.hidden_dim, num_classes)
            )
        else:
            self.classifier.heads.head = nn.Linear(self.classifier.hidden_dim, num_classes)
        
        # Load weights
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        self.class_names = checkpoint.get('class_names', list(self.species_toxicity.keys()))
        
        # Show performance info
        best_acc = checkpoint.get('best_acc', 'N/A')
        if isinstance(best_acc, (int, float)):
            print(f"  Best validation accuracy: {best_acc:.2%}")
        
        print(f"  OK - ViT classifier loaded")
        print(f"  Supported species: {len(self.class_names)}")
        
        # Preprocess (ViT uses 224x224)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("\n" + "="*70)
        print("Pipeline initialized successfully!")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
    
    def detect_mushrooms(self, image_path, conf_threshold=0.25):
        """
        Stage 1: Detect mushroom locations.
        
        Args:
            image_path: Image path
            conf_threshold: Confidence threshold
            
        Returns:
            detections: List of detection results
        """
        results = self.detector.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        return detections
    
    def classify_mushroom(self, image, bbox):
        """
        Stage 2: Classify mushroom species and toxicity.
        
        Args:
            image: original image (numpy array, BGR)
            bbox: bounding box [x1, y1, x2, y2]
            
        Returns:
            dict with species, toxicity, confidence, top3
        """
        x1, y1, x2, y2 = bbox
        
        # Crop mushroom region
        mushroom_crop = image[y1:y2, x1:x2]
        
        # Convert to PIL image
        mushroom_pil = Image.fromarray(cv2.cvtColor(mushroom_crop, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        mushroom_tensor = self.transform(mushroom_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier(mushroom_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            # Top-3 predictions
            top3_probs, top3_indices = torch.topk(probs, k=min(3, len(self.class_names)))
            top3_probs = top3_probs.cpu().numpy()
            top3_indices = top3_indices.cpu().numpy()
            
            # Best prediction
            best_idx = int(top3_indices[0])
            best_species = self.class_names[best_idx]
            best_confidence = float(top3_probs[0])
            toxicity = self.species_toxicity.get(best_species, 'unknown')
            
            # Top-3 results
            top3_species = [self.class_names[int(idx)] for idx in top3_indices]
            top3_confidences = [float(prob) for prob in top3_probs]
        
        return {
            'species': best_species,
            'toxicity': toxicity,
            'confidence': best_confidence,
            'top3_species': top3_species,
            'top3_confidences': top3_confidences
        }
    
    def process_image(self, image_path, conf_threshold=0.25):
        """
        Process single image: detection + classification.
        
        Args:
            image_path: image path
            conf_threshold: detection confidence threshold
            
        Returns:
            dict of processing results
        """
        start_time = time.time()
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Stage 1: Detect mushrooms
        detections = self.detect_mushrooms(image_path, conf_threshold)
        
        # Stage 2: Classify each detected mushroom
        mushrooms = []
        for det in detections:
            classification = self.classify_mushroom(image, det['bbox'])
            
            mushrooms.append({
                'bbox': det['bbox'],
                'detection_conf': det['confidence'],
                'species': classification['species'],
                'toxicity': classification['toxicity'],
                'classification_conf': classification['confidence'],
                'top3_species': classification['top3_species'],
                'top3_confidences': classification['top3_confidences']
            })
        
        processing_time = time.time() - start_time
        
        return {
            'image_path': str(image_path),
            'num_mushrooms': len(mushrooms),
            'mushrooms': mushrooms,
            'processing_time': processing_time
        }
    
    def visualize_result(self, image_path, result, output_path=None, show_top3=False):
        """
        Visualize detection and classification results
        
        Args:
            image_path: Original image path
            result: Result from process_image
            output_path: Output image path (optional)
            show_top3: Whether to show top-3 predictions
        """
        image = cv2.imread(str(image_path))
        
        for i, mushroom in enumerate(result['mushrooms'], 1):
            x1, y1, x2, y2 = mushroom['bbox']
            species = mushroom['species']
            toxicity = mushroom['toxicity']
            det_conf = mushroom['detection_conf']
            cls_conf = mushroom['classification_conf']
            
            # Choose color based on toxicity
            color = (0, 255, 0) if toxicity == 'edible' else (0, 0, 255)  # Green=edible, Red=poisonous
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare labels
            label = f"{species}"
            toxicity_label = f"[{toxicity.upper()}]"
            conf_label = f"Det:{det_conf:.2f} Cls:{cls_conf:.2f}"
            
            # Calculate text sizes
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tox_size, _ = cv2.getTextSize(toxicity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            conf_size, _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            total_height = label_size[1] + tox_size[1] + conf_size[1] + 15
            max_width = max(label_size[0], tox_size[0], conf_size[0]) + 10
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - total_height), 
                         (x1 + max_width, y1), color, -1)
            
            # Draw text
            y_offset = y1 - total_height + label_size[1] + 3
            cv2.putText(image, label, (x1 + 2, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += tox_size[1] + 3
            cv2.putText(image, toxicity_label, (x1 + 2, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += conf_size[1] + 3
            cv2.putText(image, conf_label, (x1 + 2, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Optional: Show Top-3 predictions
            if show_top3:
                top3_y = y2 + 15
                for j, (sp, conf) in enumerate(zip(mushroom['top3_species'][:3], 
                                                    mushroom['top3_confidences'][:3])):
                    tox = self.species_toxicity.get(sp, 'unknown')
                    tox_char = 'E' if tox == 'edible' else 'P'
                    top3_text = f"{j+1}. {sp[:20]} ({tox_char}) {conf:.2f}"
                    cv2.putText(image, top3_text, (x1, top3_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    top3_y += 12
        
        # Add statistics
        stats_text = f"Detected: {result['num_mushrooms']} mushrooms | Time: {result['processing_time']:.2f}s | Model: ViT"
        cv2.putText(image, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save or return
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"  Saved: {output_path}")
        
        return image


def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Stage Pipeline: YOLO Detection + ViT Classification')
    parser.add_argument('--detector', type=str, 
                       default='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                       help='YOLO detector model path')
    parser.add_argument('--classifier', type=str,
                       default='../FT_ViT/vit_antioverfit/best_model.pth',
                       help='ViT species classifier model path')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--dir', type=str, help='Input directory with images')
    parser.add_argument('--output', type=str, default='output_vit',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--show-top3', action='store_true',
                       help='Show top-3 predictions on image')
    
    args = parser.parse_args()
    
    # Create Pipeline
    try:
        pipeline = MushroomViTPipeline(
            detector_path=args.detector,
            classifier_path=args.classifier
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained YOLO detection model (FT_YOLO_Detection)")
        print("2. Trained ViT classification model (FT_ViT)")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"\nProcessing: {image_path}")
        result = pipeline.process_image(image_path, conf_threshold=args.conf)
        
        # Display results
        print(f"\nDetected {result['num_mushrooms']} mushroom(s):")
        for i, mushroom in enumerate(result['mushrooms'], 1):
            print(f"\n  Mushroom #{i}:")
            print(f"    Species:   {mushroom['species']}")
            print(f"    Toxicity:  {mushroom['toxicity'].upper()}")
            print(f"    Detection conf:  {mushroom['detection_conf']:.2%}")
            print(f"    Classification conf: {mushroom['classification_conf']:.2%}")
            print(f"    Top-3 predictions:")
            for j, (sp, conf) in enumerate(zip(mushroom['top3_species'], mushroom['top3_confidences']), 1):
                tox = pipeline.species_toxicity.get(sp, 'unknown')
                print(f"      {j}. {sp:40s} ({tox:10s}) - {conf:.2%}")
        
        # Visualize
        output_path = output_dir / f"result_{image_path.name}"
        pipeline.visualize_result(image_path, result, output_path, show_top3=args.show_top3)
        
        # Save JSON
        json_path = output_dir / f"result_{image_path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved JSON: {json_path}")
    
    # Process directory
    elif args.dir:
        input_dir = Path(args.dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return
        
        # Find all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(input_dir.glob(ext))
        
        print(f"\nProcessing directory: {input_dir}")
        print(f"Found {len(image_paths)} images\n")
        
        all_results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
            
            try:
                result = pipeline.process_image(image_path, conf_threshold=args.conf)
                all_results.append(result)
                
                # Visualize
                output_path = output_dir / f"result_{image_path.name}"
                pipeline.visualize_result(image_path, result, output_path, show_top3=args.show_top3)
                
                print(f"  Detected {result['num_mushrooms']} mushroom(s)")
            except Exception as e:
                print(f"  Failed: {e}")
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary: {summary_path}")
        
        # Statistics
        total_mushrooms = sum(r['num_mushrooms'] for r in all_results)
        if total_mushrooms > 0:
            species_count = {}
            toxicity_count = {'edible': 0, 'poisonous': 0}
            
            for result in all_results:
                for mushroom in result['mushrooms']:
                    species = mushroom['species']
                    toxicity = mushroom['toxicity']
                    species_count[species] = species_count.get(species, 0) + 1
                    if toxicity in toxicity_count:
                        toxicity_count[toxicity] += 1
            
            print(f"\n{'='*70}")
            print("STATISTICS")
            print(f"{'='*70}")
            print(f"Images processed: {len(all_results)}")
            print(f"Total mushrooms detected: {total_mushrooms}")
            print(f"\nToxicity distribution:")
            print(f"  Edible:    {toxicity_count['edible']} ({toxicity_count['edible']/total_mushrooms*100:.1f}%)")
            print(f"  Poisonous: {toxicity_count['poisonous']} ({toxicity_count['poisonous']/total_mushrooms*100:.1f}%)")
            print(f"\nSpecies distribution:")
            for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True):
                toxicity = pipeline.species_toxicity.get(species, 'unknown')
                print(f"  {species:40s} ({toxicity:10s}): {count:3d} ({count/total_mushrooms*100:.1f}%)")
            print(f"{'='*70}")
    
    else:
        # Demo mode - test with a sample image from test set
        print("\nNo --image or --dir specified. Running demo mode...")
        
        test_dir = Path('../../Dataset/merged_mushroom_dataset/test/images')
        if test_dir.exists():
            test_images = list(test_dir.glob('*.jpg'))[:3]
            if test_images:
                print(f"\nProcessing {len(test_images)} sample images from test set...\n")
                
                for image_path in test_images:
                    print(f"\nProcessing: {image_path.name}")
                    result = pipeline.process_image(image_path)
                    
                    print(f"  Detected {result['num_mushrooms']} mushroom(s)")
                    for mushroom in result['mushrooms']:
                        print(f"    - {mushroom['species']} ({mushroom['toxicity']}) "
                              f"conf: {mushroom['classification_conf']:.2%}")
                    
                    output_path = output_dir / f"demo_{image_path.name}"
                    pipeline.visualize_result(image_path, result, output_path)
                
                print(f"\n Demo results saved to: {output_dir}/")
            else:
                print("No test images found.")
        else:
            print(f"Test directory not found: {test_dir}")
            parser.print_help()


if __name__ == '__main__':
    main()




