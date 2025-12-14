"""
Generate YOLO detection demo images with unified 'mushroom' label.
Only shows bounding boxes with 'mushroom' label (no edible/poisonous distinction).
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


def detect_and_draw(model, image_path, output_dir='detection_demo', conf_threshold=0.25):
    """
    Detect mushrooms and draw bounding boxes with unified 'mushroom' label.
    
    Args:
        model: YOLO model
        image_path: path to input image
        output_dir: directory to save result
        conf_threshold: confidence threshold
    """
    # Run detection
    results = model.predict(source=image_path, conf=conf_threshold, save=False)
    
    # Read original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None
    
    # Draw each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            
            # Draw bounding box (blue color)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 3)
            
            # Label text: just "mushroom" + confidence
            label = f"mushroom {conf:.2f}"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), (255, 100, 0), -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    output_file = output_path / Path(image_path).name
    cv2.imwrite(str(output_file), img)
    print(f"[OK] Saved: {output_file}")
    
    return str(output_file)


def main():
    # Load model
    model_path = 'yolo_detector/weights/best.pt'
    model = YOLO(model_path)
    print(f"[OK] Model loaded: {model_path}")
    
    # Test images
    test_images = [
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\merged_mushroom_dataset\test\images\edible_Boletus-rex-veris_184_jpg.rf.0c092c03900588ea82b5eaa66d0dd062.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\merged_mushroom_dataset\test\images\nonedible_Tylopilus-felleus_150_jpg.rf.b63dd3404d4e5f53b7a15f8087639e9e.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\merged_mushroom_dataset\test\images\edible_Cantharellus-californicus_163_jpg.rf.a8ef8765c9ca8bdab51c79be1bbc64d8.jpg",
        r"c:\Users\33582\Desktop\DLCourseWork\Dataset\merged_mushroom_dataset\test\images\nonedible_Trametes-versicolor_60_jpg.rf.9e576af3beb12696571c9bb063b2120e.jpg",
    ]
    
    output_dir = 'detection_demo'
    
    print(f"\n{'='*60}")
    print("Generating YOLO Detection Demo Images")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")
    
    for img_path in test_images:
        if Path(img_path).exists():
            detect_and_draw(model, img_path, output_dir=output_dir)
        else:
            print(f"[WARN] Image not found: {img_path}")
    
    print(f"\n[DONE] Detection demo images saved to: {output_dir}/")


if __name__ == "__main__":
    main()
