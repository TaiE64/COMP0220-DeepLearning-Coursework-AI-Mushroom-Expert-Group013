"""
Two-stage mushroom detection and toxicity classification pipeline
Stage 1: YOLO detects mushroom locations
Stage 2: ResNet classifies toxicity (edible/poisonous)
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


class MushroomPipeline:
    """Two-stage mushroom detection and toxicity classification pipeline"""
    
    def __init__(self, 
                 detector_path='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                 classifier_path='../FT_ResNet/resnet_classifier_no_dropout/best_model.pth'):
        """
        Initialize the pipeline.
        
        Args:
            detector_path: path to YOLO detector weights
            classifier_path: path to ResNet classifier weights
        """
        print("="*60)
        print("Initializing two-stage mushroom detection & classification pipeline")
        print("="*60)
        
        # Load YOLO detector
        print(f"\n[1/2] Loading YOLO detector: {detector_path}")
        if not Path(detector_path).exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        self.detector = YOLO(detector_path)
        print("  ✓ YOLO detector loaded")
        
        # Load ResNet classifier
        print(f"\n[2/2] Loading ResNet classifier: {classifier_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(classifier_path, map_location=self.device)
        
        # Rebuild model
        from torchvision import models
        self.classifier = models.resnet50(pretrained=False)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 2)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        self.class_names = checkpoint.get('class_names', ['edible', 'poisonous'])
        print(f"  ✓ ResNet classifier loaded")
        print(f"  Classes: {self.class_names}")
        
        # Classifier preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("\n" + "="*60)
        print("Pipeline initialized!")
        print("="*60 + "\n")
    
    def detect_mushrooms(self, image_path, conf_threshold=0.25):
        """
        Stage 1: detect mushroom locations.
        
        Args:
            image_path: image path
            conf_threshold: confidence threshold
            
        Returns:
            detections: list of detection results
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
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        
        return detections
    
    def classify_mushroom(self, image):
        """
        Stage 2: classify toxicity.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            class_name: class name
            confidence: confidence
            probabilities: per-class probabilities
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        class_name = self.class_names[class_idx]
        conf = confidence.item()
        probs = {self.class_names[i]: probabilities[0][i].item() 
                 for i in range(len(self.class_names))}
        
        return class_name, conf, probs
    
    def extract_species_from_filename(self, image_path):
        """Extract species name from filename if present."""
        filename = Path(image_path).stem
        # filename format: edible_Species-name_number_suffix
        # or: nonedible_Species-name_number_suffix
        parts = filename.split('_')
        if len(parts) >= 2:
            # remove edible/nonedible prefix
            species_parts = []
            for i, part in enumerate(parts):
                if i == 0 and part in ['edible', 'nonedible']:
                    continue
                if part.isdigit() or part == 'jpg.rf' or '.' in part:
                    break
                species_parts.append(part)
            if species_parts:
                return ' '.join(species_parts).replace('-', ' ').title()
        return "Unknown Species"
    
    def process_image(self, image_path, conf_threshold=0.25, save_result=False, output_dir='results'):
        """
        Full pipeline: detection + classification.
        
        Args:
            image_path: image path
            conf_threshold: detection confidence threshold
            save_result: whether to save annotated image
            output_dir: output directory
            
        Returns:
            results: processing results
        """
        start_time = time.time()
        
        # Extract species info from filename
        species = self.extract_species_from_filename(image_path)
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Stage 1: detection
        detections = self.detect_mushrooms(image_path, conf_threshold)
        
        if len(detections) == 0:
            return {
                'image_path': str(image_path),
                'num_mushrooms': 0,
                'mushrooms': [],
                'processing_time': time.time() - start_time
            }
        
        # Stage 2: classify each detection
        mushrooms = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Crop mushroom
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            # Classify
            class_name, class_conf, probs = self.classify_mushroom(cropped)
            
            mushrooms.append({
                'id': i + 1,
                'bbox': det['bbox'],
                'detection_confidence': det['confidence'],
                'species': species,  # species inferred from filename
                'class': class_name,
                'classification_confidence': class_conf,
                'probabilities': probs,
                'is_poisonous': (class_name == 'poisonous')
            })
        
        # Result summary
        result = {
            'image_path': str(image_path),
            'num_mushrooms': len(mushrooms),
            'mushrooms': mushrooms,
            'processing_time': time.time() - start_time
        }
        
        # Save visualization results
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            result_img = self.visualize_result(img.copy(), mushrooms)
            output_file = output_path / f"{Path(image_path).stem}_result.jpg"
            cv2.imwrite(str(output_file), result_img)
            result['output_image'] = str(output_file)
        
        return result
    
    def visualize_result(self, img, mushrooms):
        """Visualize detection and classification results"""
        for mushroom in mushrooms:
            x1, y1, x2, y2 = mushroom['bbox']
            species = mushroom.get('species', 'Unknown')
            class_name = mushroom['class']
            det_conf = mushroom['detection_confidence']
            cls_conf = mushroom['classification_confidence']
            is_poisonous = mushroom['is_poisonous']
            
            # Colors: poisonous=red, edible=green
            color = (0, 0, 255) if is_poisonous else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label - only show AI-recognized content
            label = f"{class_name} {cls_conf:.2f}"
            if is_poisonous:
                label = "WARNING! " + label
            
            # Background box (adjust font size to accommodate longer species names)
            font_scale = 0.5
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return img
    
    def process_batch(self, image_dir, conf_threshold=0.25, save_results=True, output_dir='results'):
        """
        Batch process images in a directory.
        
        Args:
            image_dir: image directory
            conf_threshold: detection confidence threshold
            save_results: whether to save results
            output_dir: output directory
            
        Returns:
            all_results: list of results
        """
        image_path = Path(image_dir)
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        
        print(f"\nBatch processing: found {len(image_files)} images")
        print("="*60)
        
        all_results = []
        total_mushrooms = 0
        poisonous_count = 0
        
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            result = self.process_image(img_file, conf_threshold, save_results, output_dir)
            all_results.append(result)
            
            total_mushrooms += result['num_mushrooms']
            poisonous_count += sum(1 for m in result['mushrooms'] if m['is_poisonous'])
            
            print(f"  Detected {result['num_mushrooms']} mushrooms")
            for m in result['mushrooms']:
                status = "⚠️  Poisonous" if m['is_poisonous'] else "✓ Edible"
                print(f"    Mushroom {m['id']}: {m['class']} ({m['classification_confidence']:.2%}) - {status}")
        
        # Summary
        print("\n" + "="*60)
        print("Batch processing complete!")
        print("="*60)
        print(f"Total images: {len(image_files)}")
        print(f"Total mushrooms: {total_mushrooms}")
        print(f"Edible: {total_mushrooms - poisonous_count}")
        print(f"Poisonous: {poisonous_count}")
        
        return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Stage Mushroom Detection & Classification Pipeline')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--detector', type=str, 
                       default='../FT_YOLO_Detection/yolo_detector/weights/best.pt',
                       help='YOLO detector model path')
    parser.add_argument('--classifier', type=str,
                       default='../FT_ResNet/resnet_classifier_no_dropout/best_model.pth',
                       help='ResNet classifier model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save visualization results')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MushroomPipeline(args.detector, args.classifier)
    
    # Process input
    source_path = Path(args.source)
    if source_path.is_file():
        # Single image
        result = pipeline.process_image(args.source, args.conf, args.save, args.output_dir)
        
        print("\n" + "="*60)
        print("Result")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"Mushrooms detected: {result['num_mushrooms']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        for m in result['mushrooms']:
            print(f"\nMushroom {m['id']}:")
            # Species info from filename only (not AI predicted)
            if 'species' in m and m['species'] != 'Unknown Species':
                print(f"  Filename species (reference): {m['species']}")
            print(f"  BBox: {m['bbox']}")
            print(f"  Detection confidence: {m['detection_confidence']:.2%}")
            print(f"  AI class: {m['class']}")
            print(f"  Classification confidence: {m['classification_confidence']:.2%}")
            print(f"  Status: {'⚠️  Poisonous!' if m['is_poisonous'] else '✓ Edible'}")
        
        if args.save:
            print(f"\nSaved result: {result['output_image']}")
    
    elif source_path.is_dir():
        # Batch mode
        results = pipeline.process_batch(args.source, args.conf, args.save, args.output_dir)
        
        # Save JSON
        output_json = Path(args.output_dir) / 'batch_results.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results: {output_json}")
    
    else:
        print(f"[ERROR] Invalid path: {args.source}")


if __name__ == "__main__":
    print("="*60)
    print("Two-Stage Mushroom Detection & Classification Pipeline")
    print("="*60)
    print("\nExamples:")
    print("1. Single image:")
    print("   python pipeline.py --source path/to/image.jpg --save")
    print("\n2. Batch process:")
    print("   python pipeline.py --source path/to/images/ --save")
    print("\n3. Adjust confidence:")
    print("   python pipeline.py --source image.jpg --conf 0.5 --save")
    print("\n" + "="*60 + "\n")
    
    main()

