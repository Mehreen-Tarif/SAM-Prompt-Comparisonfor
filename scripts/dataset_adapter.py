# scripts/dataset_adapter.py
import cv2
import numpy as np
from pathlib import Path
import json
import albumentations as A

class RemoteSensingDataset:
    """Adapter for your remote sensing dataset"""
    
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split = split
        
        # Load annotations
        self.annotations = self.load_annotations()
        
        # Define augmentations for training
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def load_annotations(self):
        """Load your dataset annotations"""
        # Assuming you have COCO format or similar
        annotations_path = self.data_root / 'annotations.json'
        
        if annotations_path.exists():
            with open(annotations_path) as f:
                return json.load(f)
        
        # Alternative: Load from folder structure
        annotations = []
        image_files = list((self.data_root / 'images').glob('*.tif')) + \
                     list((self.data_root / 'images').glob('*.png'))
        
        for img_path in image_files:
            # Create dummy annotation - replace with your actual annotation loading
            annotation = {
                'image_path': str(img_path),
                'bboxes': [[100, 100, 200, 200]],  # [x1, y1, x2, y2]
                'masks': None,  # Path to mask file
                'class': 'building'  # Your class
            }
            annotations.append(annotation)
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image = cv2.imread(annotation['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        if annotation['masks']:
            mask = cv2.imread(annotation['masks'], 0)
            mask = (mask > 0).astype(np.uint8)
        else:
            # Create dummy mask from bbox
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for bbox in annotation['bboxes']:
                x1, y1, x2, y2 = map(int, bbox)
                mask[y1:y2, x1:x2] = 1
        
        # Apply augmentations if training
        if self.split == 'train':
            transformed = self.transform(
                image=image,
                masks=[mask],
                bboxes=annotation['bboxes'],
                class_labels=[annotation['class']]
            )
            image = transformed['image']
            mask = transformed['masks'][0]
        
        # Use first bbox for prompt generation
        bbox = annotation['bboxes'][0] if annotation['bboxes'] else [0, 0, 100, 100]
        
        return image, mask, bbox, annotation['class']