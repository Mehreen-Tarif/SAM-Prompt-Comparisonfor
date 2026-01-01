"""
Generate synthetic remote sensing dataset for SAM prompt comparison
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path
import random
from tqdm import tqdm

class SyntheticDatasetGenerator:
    def __init__(self, output_dir="data/synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Object classes
        self.classes = ['airplane', 'building', 'ship', 'storage_tank', 'vehicle']
        
        # Background types
        self.backgrounds = {
            'land': (139, 119, 101),      # Brown
            'vegetation': (34, 139, 34),   # Green
            'urban': (128, 128, 128),      # Gray
            'water': (30, 144, 255)        # Blue
        }
        
    def generate_background(self, bg_type, size=(512, 512)):
        """Generate a background image"""
        color = self.backgrounds[bg_type]
        background = np.ones((size[0], size[1], 3), dtype=np.uint8) * color
        return background
    
    def generate_airplane(self, size=(100, 100)):
        """Generate airplane shape"""
        obj = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        # Draw simplified airplane
        cv2.rectangle(obj, (30, 45), (70, 55), (255, 255, 255), -1)  # Body
        cv2.rectangle(obj, (45, 30), (55, 70), (255, 255, 255), -1)  # Wings
        cv2.fillPoly(obj, [np.array([[70,45], [90,45], [70,30]])], (255,255,255))  # Tail
        return obj
    
    def generate_building(self, size=(100, 100)):
        """Generate building shape"""
        obj = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.rectangle(obj, (30, 20), (70, 80), (255, 255, 255), -1)  # Main building
        cv2.rectangle(obj, (35, 10), (65, 20), (255, 255, 255), -1)  # Roof
        return obj
    
    def generate_ship(self, size=(100, 100)):
        """Generate ship shape"""
        obj = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.rectangle(obj, (20, 60), (80, 70), (255, 255, 255), -1)  # Hull
        cv2.fillPoly(obj, [np.array([[50,20], [70,20], [60,60]])], (255,255,255))  # Sail
        return obj
    
    def generate_storage_tank(self, size=(100, 100)):
        """Generate storage tank shape"""
        obj = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.ellipse(obj, (50, 50), (30, 20), 0, 0, 360, (255, 255, 255), -1)  # Tank body
        cv2.rectangle(obj, (45, 30), (55, 50), (255, 255, 255), -1)  # Top structure
        return obj
    
    def generate_vehicle(self, size=(100, 100)):
        """Generate vehicle shape"""
        obj = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.rectangle(obj, (30, 45), (70, 55), (255, 255, 255), -1)  # Body
        cv2.circle(obj, (35, 55), 5, (255, 255, 255), -1)  # Wheel
        cv2.circle(obj, (65, 55), 5, (255, 255, 255), -1)  # Wheel
        return obj
    
    def generate_sample(self, obj_class, bg_type, sample_id):
        """Generate a single sample"""
        # Generate background
        background = self.generate_background(bg_type)
        
        # Generate object
        if obj_class == 'airplane':
            obj = self.generate_airplane()
        elif obj_class == 'building':
            obj = self.generate_building()
        elif obj_class == 'ship':
            obj = self.generate_ship()
        elif obj_class == 'storage_tank':
            obj = self.generate_storage_tank()
        else:  # vehicle
            obj = self.generate_vehicle()
        
        # Random position
        h, w = obj.shape[:2]
        bg_h, bg_w = background.shape[:2]
        x = random.randint(0, bg_w - w)
        y = random.randint(0, bg_h - h)
        
        # Place object on background
        result = background.copy()
        mask = obj[:, :, 0] > 0
        result[y:y+h, x:x+w][mask] = obj[mask]
        
        # Generate annotations
        annotations = {
            'class': obj_class,
            'background': bg_type,
            'bbox': [x, y, x+w, y+h],
            'center_point': [x + w//2, y + h//2],
            'multiple_points': [
                [x, y],           # Top-left
                [x + w, y],       # Top-right
                [x, y + h]        # Bottom-left
            ]
        }
        
        return result, annotations
    
    def generate_dataset(self, samples_per_class=10):
        """Generate complete dataset"""
        metadata = []
        
        for obj_class in tqdm(self.classes, desc="Generating dataset"):
            class_dir = self.output_dir / obj_class
            class_dir.mkdir(exist_ok=True)
            
            for i in range(samples_per_class):
                # Random background
                bg_type = random.choice(list(self.backgrounds.keys()))
                
                # Generate sample
                image, annotations = self.generate_sample(obj_class, bg_type, i)
                
                # Save image
                img_path = class_dir / f"{obj_class}_{i:02d}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Save metadata
                metadata.append({
                    'image_path': str(img_path),
                    **annotations
                })
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset generated: {len(metadata)} samples")
        return metadata

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator()
    metadata = generator.generate_dataset(samples_per_class=10)
