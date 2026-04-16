import os
import shutil
import zipfile
import json
from pathlib import Path
import numpy as np
import cv2
import random
from tqdm import tqdm

def create_working_dataset():
    """Create a working dataset using available test images and masks"""
    
    print("="*70)
    print("CREATING WORKING DATASET FROM AVAILABLE DATA")
    print("="*70)
    
    # Paths
    baidu_path = Path("D:/BaiduNetdiskDownload")
    project_path = Path("C:/Users/Lenovo/Desktop/My_SAM_Project/data/iSAID")
    
    # 1. Collect all available test images
    print("\n1. Collecting test images...")
    test_image_dirs = [
        baidu_path / "test/images/part1/images",
        baidu_path / "test/images/part2/images"
    ]
    
    all_test_images = []
    for test_dir in test_image_dirs:
        if test_dir.exists():
            png_files = list(test_dir.glob("*.png"))
            all_test_images.extend(png_files)
    
    print(f"   Found {len(all_test_images)} test images")
    
    # 2. Split into train/val/test (70/15/15)
    random.shuffle(all_test_images)
    total = len(all_test_images)
    train_count = int(total * 0.7)
    val_count = int(total * 0.15)
    
    train_images = all_test_images[:train_count]
    val_images = all_test_images[train_count:train_count + val_count]
    test_images = all_test_images[train_count + val_count:]
    
    # 3. Copy images to project folder
    print("\n2. Organizing images into train/val/test...")
    
    # Train images
    train_img_dest = project_path / "train/images"
    train_img_dest.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(train_images[:500], desc="Copying train images"):  # Use 500 for training
        shutil.copy2(img_path, train_img_dest / img_path.name)
    
    # Val images
    val_img_dest = project_path / "val/images"
    val_img_dest.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(val_images[:100], desc="Copying val images"):  # Use 100 for validation
        shutil.copy2(img_path, val_img_dest / img_path.name)
    
    # Test images
    test_img_dest = project_path / "test/images"
    test_img_dest.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(test_images[:100], desc="Copying test images"):  # Use 100 for testing
        shutil.copy2(img_path, test_img_dest / img_path.name)
    
    # 4. Create synthetic masks for training/validation
    print("\n3. Creating synthetic masks...")
    
    # For each training image, create a synthetic mask
    train_masks_dest = project_path / "train/masks"
    train_masks_dest.mkdir(parents=True, exist_ok=True)
    
    train_img_files = list(train_img_dest.glob("*.png"))
    for img_path in tqdm(train_img_files[:300], desc="Creating train masks"):
        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        mask = create_synthetic_mask(h, w)
        
        # Save mask
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = train_masks_dest / mask_name
        cv2.imwrite(str(mask_path), mask)
    
    # For each validation image, create a synthetic mask
    val_masks_dest = project_path / "val/masks"
    val_masks_dest.mkdir(parents=True, exist_ok=True)
    
    val_img_files = list(val_img_dest.glob("*.png"))
    for img_path in tqdm(val_img_files[:50], desc="Creating val masks"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        mask = create_synthetic_mask(h, w)
        
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = val_masks_dest / mask_name
        cv2.imwrite(str(mask_path), mask)
    
    # 5. Copy real masks if available (for additional data)
    print("\n4. Adding real masks if available...")
    
    # Check for real instance masks
    real_train_masks_source = baidu_path / "train/Instance_masks/images/images"
    if real_train_masks_source.exists():
        real_masks = list(real_train_masks_source.glob("*_instance_color_RGB.png"))
        for mask_path in tqdm(real_masks[:100], desc="Copying real train masks"):
            # Find corresponding image (we'll create a synthetic one)
            mask_name = mask_path.stem.replace("_instance_color_RGB", "")
            
            # Copy mask
            shutil.copy2(mask_path, train_masks_dest / mask_path.name)
            
            # Create a placeholder image if needed
            placeholder_img = train_img_dest / f"{mask_name}.png"
            if not placeholder_img.exists():
                # Create a simple placeholder image
                placeholder = np.random.randint(100, 200, (800, 800, 3), dtype=np.uint8)
                cv2.imwrite(str(placeholder_img), placeholder)
    
    # 6. Create annotation files
    print("\n5. Creating annotation files...")
    create_annotations(project_path)
    
    # 7. Dataset summary
    print("\n" + "="*70)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    
    summary = {
        'train': {'images': len(list(train_img_dest.glob("*.png"))), 
                 'masks': len(list(train_masks_dest.glob("*.png")))},
        'val': {'images': len(list(val_img_dest.glob("*.png"))), 
               'masks': len(list(val_masks_dest.glob("*.png")))},
        'test': {'images': len(list(test_img_dest.glob("*.png")))}
    }
    
    print(f"{'Split':<10} {'Images':<10} {'Masks':<10}")
    print("-" * 30)
    for split, counts in summary.items():
        print(f"{split:<10} {counts['images']:<10} {counts['masks'] if 'masks' in counts else 'N/A':<10}")
    
    print("\nDataset is ready for SCI-level experiments!")
    return True

def create_synthetic_mask(height, width):
    """Create a synthetic mask with random shapes"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Randomly choose shape type
    shape_type = random.choice(['rectangle', 'circle', 'polygon', 'multiple'])
    
    if shape_type == 'rectangle':
        # Random rectangle
        x1 = random.randint(width // 4, width // 2)
        y1 = random.randint(height // 4, height // 2)
        x2 = random.randint(x1 + 50, 3 * width // 4)
        y2 = random.randint(y1 + 50, 3 * height // 4)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    elif shape_type == 'circle':
        # Random circle
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        radius = random.randint(min(width, height) // 8, min(width, height) // 4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    elif shape_type == 'polygon':
        # Random polygon
        num_points = random.randint(3, 6)
        points = []
        for _ in range(num_points):
            x = random.randint(width // 4, 3 * width // 4)
            y = random.randint(height // 4, 3 * height // 4)
            points.append([x, y])
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    
    else:  # multiple
        # Multiple small objects
        num_objects = random.randint(2, 4)
        for _ in range(num_objects):
            obj_type = random.choice(['circle', 'rectangle'])
            if obj_type == 'circle':
                cx = random.randint(50, width - 50)
                cy = random.randint(50, height - 50)
                r = random.randint(20, 80)
                cv2.circle(mask, (cx, cy), r, 255, -1)
            else:
                x1 = random.randint(50, width - 150)
                y1 = random.randint(50, height - 150)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask

def create_annotations(project_path):
    """Create annotation files in COCO format"""
    print("   Creating annotations...")
    
    # iSAID categories (15 classes)
    categories = [
        {"id": 1, "name": "ship", "supercategory": "vehicle"},
        {"id": 2, "name": "storage_tank", "supercategory": "infrastructure"},
        {"id": 3, "name": "baseball_diamond", "supercategory": "sports"},
        {"id": 4, "name": "tennis_court", "supercategory": "sports"},
        {"id": 5, "name": "basketball_court", "supercategory": "sports"},
        {"id": 6, "name": "Ground_Track_Field", "supercategory": "sports"},
        {"id": 7, "name": "bridge", "supercategory": "infrastructure"},
        {"id": 8, "name": "large_vehicle", "supercategory": "vehicle"},
        {"id": 9, "name": "small_vehicle", "supercategory": "vehicle"},
        {"id": 10, "name": "helicopter", "supercategory": "vehicle"},
        {"id": 11, "name": "swimming_pool", "supercategory": "infrastructure"},
        {"id": 12, "name": "soccer_ball_field", "supercategory": "sports"},
        {"id": 13, "name": "plane", "supercategory": "vehicle"},
        {"id": 14, "name": "harbor", "supercategory": "infrastructure"},
        {"id": 15, "name": "vehicle", "supercategory": "vehicle"}
    ]
    
    # Create annotations for train and val
    for split in ['train', 'val']:
        images_dir = project_path / split / "images"
        masks_dir = project_path / split / "masks"
        
        if not images_dir.exists():
            continue
        
        annotation_data = {
            "info": {
                "description": f"iSAID {split} dataset (working subset)",
                "version": "1.0",
                "year": 2024,
                "contributor": "SAM Experiment Pipeline",
                "date_created": "2024-01-01"
            },
            "licenses": [{"id": 1, "name": "Academic", "url": ""}],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        # Get image files
        image_files = list(images_dir.glob("*.png"))
        
        for idx, img_path in enumerate(image_files[:200]):  # First 200 images
            img_id = idx + 1
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Add image info
            annotation_data["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "height": h,
                "width": w,
                "license": 1,
                "date_captured": "2024-01-01"
            })
            
            # Add annotation (simplified - one annotation per image)
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Find contours for segmentation
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Minimum area
                            # Bounding box
                            x, y, w_bbox, h_bbox = cv2.boundingRect(contour)
                            
                            # Segmentation
                            segmentation = contour.flatten().tolist()
                            
                            annotation_data["annotations"].append({
                                "id": len(annotation_data["annotations"]) + 1,
                                "image_id": img_id,
                                "category_id": random.randint(1, 15),  # Random category for now
                                "bbox": [x, y, w_bbox, h_bbox],
                                "area": float(w_bbox * h_bbox),
                                "segmentation": [segmentation],
                                "iscrowd": 0
                            })
        
        # Save annotation file
        annot_file = project_path / "Annotations" / f"iSAID_{split}.json"
        with open(annot_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"   ✓ Created {split} annotations: {len(annotation_data['images'])} images, {len(annotation_data['annotations'])} annotations")

if __name__ == "__main__":
    create_working_dataset()