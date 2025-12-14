"""
Two-stage mushroom detection and species identification pipeline (16 classes)
Stage 1: YOLO detects mushroom locations
Stage 2: ResNet identifies species + toxicity (16 species)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import json
import time


class MushroomSpeciesPipeline:
    """Two-stage detection + species identification (16-class)"""
    
    def __init__(self, 
                 detector_path='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                 classifier_path='../FT_ResNet_MultiClass/species_classifier/best_model.pth',
                 mapping_file='../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json'):
        """
        Initialize pipeline.
        
        Args:
            detector_path: YOLO detector weights path
            classifier_path: ResNet species classifier path (16 classes)
            mapping_file: species-to-toxicity mapping file
        """
        print("="*70)
        print("Initializing two-stage mushroom detection & species pipeline (16 classes)")
        print("="*70)
        
        # Load species-to-toxicity mapping
        print(f"\n[0/3] Loading species-toxicity mapping: {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.species_toxicity = json.load(f)
        print(f"  ✓ Loaded {len(self.species_toxicity)} species mappings")
        
        # Load YOLO detector
        print(f"\n[1/3] Loading YOLO detector: {detector_path}")
        if not Path(detector_path).exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        self.detector = YOLO(detector_path)
        print("  ✓ YOLO detector loaded")
        
        # Load ResNet species classifier (16 classes)
        print(f"\n[2/3] Loading ResNet species classifier (16 classes): {classifier_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(classifier_path, map_location=self.device)
        
        # Rebuild model (16 classes)
        from torchvision import models
        self.classifier = models.resnet50(pretrained=False)
        num_ftrs = self.classifier.fc.in_features
        num_classes = len(self.species_toxicity)  # 16 species
        self.classifier.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        self.class_names = checkpoint.get('class_names', list(self.species_toxicity.keys()))
        print(f"  ✓ ResNet species classifier loaded")
        print(f"  Supported species: {len(self.class_names)}")
        
        # Preprocess for classifier
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("\n" + "="*70)
        print("Pipeline initialized!")
        print("="*70 + "\n")
    
    def detect_mushrooms(self, image_path, conf_threshold=0.25):
        """
        Stage 1: detect mushroom locations.
        
        Args:
            image_path: image path
            conf_threshold: confidence threshold
            
        Returns:
            detections: list of detections
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
        Stage 2: classify species and toxicity.
        
        Args:
            image: original image (numpy array)
            bbox: bounding box [x1, y1, x2, y2]
            
        Returns:
            dict with species, toxicity, confidence, top3 predictions
        """
        x1, y1, x2, y2 = bbox
        
        # Crop mushroom
        mushroom_crop = image[y1:y2, x1:x2]
        
        # To PIL
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
        Process a single image: detection + classification.
        
        Args:
            image_path: image path
            conf_threshold: detection confidence threshold
            
        Returns:
            dict with counts and per-mushroom results
        """
        start_time = time.time()
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Stage 1: detect
        detections = self.detect_mushrooms(image_path, conf_threshold)
        
        # Stage 2: classify each detection
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
        Visualize detection and classification results.
        
        Args:
            image_path: original image path
            result: result dict from process_image
            output_path: optional output image path
            show_top3: whether to show top-3 predictions
        """
        image = cv2.imread(str(image_path))
        
        for i, mushroom in enumerate(result['mushrooms'], 1):
            x1, y1, x2, y2 = mushroom['bbox']
            species = mushroom['species']
            toxicity = mushroom['toxicity']
            det_conf = mushroom['detection_conf']
            cls_conf = mushroom['classification_conf']
            
            # 根据毒性选择颜色
            color = (0, 255, 0) if toxicity == 'edible' else (0, 0, 255)  # 绿色=可食用，红色=有毒
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签
            label = f"{species} ({toxicity})"
            conf_label = f"Det:{det_conf:.2f} Cls:{cls_conf:.2f}"
            
            # 绘制标签背景
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            conf_size, _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            cv2.rectangle(image, (x1, y1 - label_size[1] - conf_size[1] - 10), 
                         (x1 + max(label_size[0], conf_size[0]) + 5, y1), color, -1)
            
            # 绘制文字
            cv2.putText(image, label, (x1 + 2, y1 - conf_size[1] - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, conf_label, (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Optional: show Top-3 predictions
            if show_top3:
                top3_text = "Top3: " + ", ".join([
                    f"{s}({c:.2f})" for s, c in 
                    zip(mushroom['top3_species'][:3], mushroom['top3_confidences'][:3])
                ])
                cv2.putText(image, top3_text, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加统计信息
        stats_text = f"Detected: {result['num_mushrooms']} mushrooms | Time: {result['processing_time']:.2f}s"
        cv2.putText(image, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 保存或显示
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"✓ Saved: {output_path}")
        
        return image


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Stage Mushroom Detection and Species Classification Pipeline')
    parser.add_argument('--detector', type=str, 
                       default='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                       help='YOLO detector model path')
    parser.add_argument('--classifier', type=str,
                       default='../FT_ResNet_MultiClass/species_classifier/best_model.pth',
                       help='ResNet species classifier model path (16 classes)')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--dir', type=str, help='Input directory with images')
    parser.add_argument('--output', type=str, default='output_species',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--show-top3', action='store_true',
                       help='Show top-3 predictions on image')
    
    args = parser.parse_args()
    
    # Create pipeline
    try:
        pipeline = MushroomSpeciesPipeline(
            detector_path=args.detector,
            classifier_path=args.classifier
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Train YOLO detector (FT_YOLO_Detection)")
        print("2. Train ResNet species classifier (FT_ResNet_MultiClass)")
        return
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"\nProcessing image: {image_path}")
        result = pipeline.process_image(image_path, conf_threshold=args.conf)
        
        # Display
        print(f"\nDetected {result['num_mushrooms']} mushrooms:")
        for i, mushroom in enumerate(result['mushrooms'], 1):
            print(f"\n  Mushroom #{i}:")
            print(f"    Species:   {mushroom['species']}")
            print(f"    Toxicity:  {mushroom['toxicity'].upper()}")
            print(f"    Det conf:  {mushroom['detection_conf']:.2%}")
            print(f"    Cls conf:  {mushroom['classification_conf']:.2%}")
            print(f"    Top-3:")
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
        print(f"✓ JSON saved: {json_path}")
    
    # Directory
    elif args.dir:
        input_dir = Path(args.dir)
        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return
        
        # Find images
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
                
                # 可视化
                output_path = output_dir / f"result_{image_path.name}"
                pipeline.visualize_result(image_path, result, output_path, show_top3=args.show_top3)
                
                print(f"  ✓ Detected {result['num_mushrooms']} mushrooms")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Summary saved: {summary_path}")
        
        # Stats
        total_mushrooms = sum(r['num_mushrooms'] for r in all_results)
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
        print("Statistics")
        print(f"{'='*70}")
        print(f"Images processed: {len(all_results)}")
        print(f"Total mushrooms: {total_mushrooms}")
        print(f"\nToxicity distribution:")
        print(f"  Edible: {toxicity_count['edible']} ({toxicity_count['edible']/total_mushrooms*100:.1f}%)")
        print(f"  Poisonous: {toxicity_count['poisonous']} ({toxicity_count['poisonous']/total_mushrooms*100:.1f}%)")
        print(f"\nSpecies distribution:")
        for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True):
            toxicity = pipeline.species_toxicity.get(species, 'unknown')
            print(f"  {species:40s} ({toxicity:10s}): {count:3d} ({count/total_mushrooms*100:.1f}%)")
        print(f"{'='*70}")
    
    else:
        print("\n❌ Please specify --image or --dir")
        parser.print_help()


if __name__ == '__main__':
    main()

