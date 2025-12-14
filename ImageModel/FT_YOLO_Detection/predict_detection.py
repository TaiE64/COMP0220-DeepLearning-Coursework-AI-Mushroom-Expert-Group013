"""
YOLOv8 mushroom detection inference script.
Detects mushroom locations and returns bounding boxes.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path
import cv2
import argparse


class MushroomDetector:
    """Mushroom detector"""
    
    def __init__(self, model_path='yolo_detector/weights/best.pt'):
        """
        Initialize detector
        
        Args:
            model_path: path to model weights
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"[OK] Model loaded: {model_path}")
    
    def detect(self, image_path, conf_threshold=0.25, save=False, save_dir='detections'):
        """
        Detect mushrooms in an image.
        
        Args:
            image_path: image path
            conf_threshold: confidence threshold
            save: whether to save result image
            save_dir: directory to save
            
        Returns:
            detections: list of results with (x1, y1, x2, y2, confidence)
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save,
            project=save_dir,
            name='predict',
            exist_ok=True
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
    
    def detect_and_crop(self, image_path, conf_threshold=0.25, output_dir='cropped'):
        """
        Detect and crop mushroom regions.
        
        Args:
            image_path: image path
            conf_threshold: confidence threshold
            output_dir: output directory
            
        Returns:
            cropped_images: list of cropped images
        """
        # Detect
        detections = self.detect(image_path, conf_threshold=conf_threshold, save=False)
        
        if len(detections) == 0:
            print(f"[INFO] No mushrooms detected: {image_path}")
            return []
        
        # Read original image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            return []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Crop each detection box
        cropped_images = []
        image_name = Path(image_path).stem
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Crop
            cropped = img[y1:y2, x1:x2]
            
            # Save
            output_filename = f"{image_name}_mushroom{i}_conf{conf:.2f}.jpg"
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), cropped)
            
            cropped_images.append({
                'image': cropped,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'file_path': str(output_file)
            })
        
        print(f"[OK] Detected {len(detections)} mushrooms, saved to: {output_dir}")
        
        return cropped_images
    
    def detect_batch(self, image_dir, conf_threshold=0.25, save=True, save_dir='detections'):
        """
        Batch-detect all images in a directory.
        
        Args:
            image_dir: image directory
            conf_threshold: confidence threshold
            save: whether to save results
            save_dir: save directory
            
        Returns:
            all_detections: all detection results
        """
        image_path = Path(image_dir)
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        
        print(f"[INFO] Found {len(image_files)} images")
        
        all_detections = {}
        for img_file in image_files:
            detections = self.detect(img_file, conf_threshold=conf_threshold, save=save, save_dir=save_dir)
            all_detections[str(img_file)] = detections
            print(f"  {img_file.name}: detected {len(detections)} mushrooms")
        
        return all_detections


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='YOLOv8 Mushroom Detection')
    parser.add_argument('--model', type=str, default='yolo_detector/weights/best.pt', help='Model path')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'crop', 'batch'], 
                        help='Mode: detect, crop, or batch')
    parser.add_argument('--save_dir', type=str, default='detections', help='Save directory')
    args = parser.parse_args()
    
    # Initialize detector
    detector = MushroomDetector(args.model)
    
    if args.mode == 'detect':
        # Single image detection
        detections = detector.detect(args.source, conf_threshold=args.conf, save=True, save_dir=args.save_dir)
        print(f"\nDetections:")
        for i, det in enumerate(detections):
            print(f"  Mushroom {i+1}: bbox={det['bbox']}, confidence={det['confidence']:.3f}")
    
    elif args.mode == 'crop':
        # Detect and crop
        cropped = detector.detect_and_crop(args.source, conf_threshold=args.conf, output_dir='cropped_mushrooms')
        print(f"\nCropping done: {len(cropped)} mushrooms")
    
    elif args.mode == 'batch':
        # Batch detection
        all_detections = detector.detect_batch(args.source, conf_threshold=args.conf, save=True, save_dir=args.save_dir)
        total = sum(len(dets) for dets in all_detections.values())
        print(f"\nBatch detection done: {total} mushrooms detected")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("YOLOv8 Mushroom Detector")
    print("="*60)
    print("\nExamples:")
    print("1. Detect a single image:")
    print("   python predict_detection.py --source path/to/image.jpg --mode detect")
    print("\n2. Detect and crop:")
    print("   python predict_detection.py --source path/to/image.jpg --mode crop")
    print("\n3. Batch detect:")
    print("   python predict_detection.py --source path/to/images/ --mode batch")
    print("\n" + "="*60 + "\n")
    
    main()

