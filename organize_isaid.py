import os
import shutil
import zipfile
import json
from pathlib import Path
import sys

def organize_isaid_dataset():
    """Organize iSAID dataset from Baidu download to project folder"""
    
    # Paths
    baidu_path = Path("D:/BaiduNetdiskDownload")
    project_path = Path("C:/Users/Lenovo/Desktop/My_SAM_Project/data/iSAID")
    
    print("="*60)
    print("ORGANIZING iSAID DATASET")
    print("="*60)
    
    # 1. Copy annotations
    print("\n1. Copying annotations...")
    annotations_to_copy = [
        (baidu_path / "train/Annotations/iSAID_train.json", project_path / "Annotations"),
        (baidu_path / "train/Annotations/iSAID_train_20190823_114751.json", project_path / "Annotations"),
        (baidu_path / "val/Annotations/iSAID_val.json", project_path / "Annotations"),
        (baidu_path / "val/Annotations/iSAID_val_20190823_114742.json", project_path / "Annotations")
    ]
    
    for src, dst in annotations_to_copy:
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"   ✓ Copied: {src.name}")
        else:
            print(f"   ✗ Missing: {src}")
    
    # 2. Extract test images
    print("\n2. Extracting test images...")
    test_zips = [
        baidu_path / "test/images/part1.zip",
        baidu_path / "test/images/part2.zip"
    ]
    
    test_images_dir = project_path / "test/images"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    for zip_path in test_zips:
        if zip_path.exists():
            print(f"   Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(test_images_dir)
            print(f"   ✓ Extracted: {zip_path.name}")
    
    # 3. Extract training masks
    print("\n3. Extracting training masks...")
    train_masks_zip = baidu_path / "train/Instance_masks/images.zip"
    train_masks_dir = project_path / "train/masks"
    train_masks_dir.mkdir(parents=True, exist_ok=True)
    
    if train_masks_zip.exists():
        with zipfile.ZipFile(train_masks_zip, 'r') as zip_ref:
            zip_ref.extractall(train_masks_dir)
        print(f"   ✓ Extracted training instance masks")
    
    # 4. Extract validation masks
    print("\n4. Extracting validation masks...")
    val_masks_zip = baidu_path / "val/Instance_masks/images.zip"
    val_masks_dir = project_path / "val/masks"
    val_masks_dir.mkdir(parents=True, exist_ok=True)
    
    if val_masks_zip.exists():
        with zipfile.ZipFile(val_masks_zip, 'r') as zip_ref:
            zip_ref.extractall(val_masks_dir)
        print(f"   ✓ Extracted validation instance masks")
    
    # 5. Check for training images
    print("\n5. Looking for training images...")
    # Check if there are already extracted images
    train_images_src = baidu_path / "train/images"
    train_images_dst = project_path / "train/images"
    train_images_dst.mkdir(parents=True, exist_ok=True)
    
    if train_images_src.exists():
        # Copy existing images
        for img_file in train_images_src.glob("**/*.png"):
            shutil.copy2(img_file, train_images_dst)
        print(f"   ✓ Copied training images")
    else:
        print(f"   ⓘ Training images not found. You may need to extract them from zip files.")
    
    # 6. Check for validation images
    print("\n6. Looking for validation images...")
    val_images_src = baidu_path / "val/images"
    val_images_dst = project_path / "val/images"
    val_images_dst.mkdir(parents=True, exist_ok=True)
    
    if val_images_src.exists():
        for img_file in val_images_src.glob("**/*.png"):
            shutil.copy2(img_file, val_images_dst)
        print(f"   ✓ Copied validation images")
    else:
        print(f"   ⓘ Validation images not found. You may need to extract them from zip files.")
    
    # 7. Create summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for folder in ["train/images", "train/masks", "val/images", "val/masks", "test/images", "Annotations"]:
        folder_path = project_path / folder
        if folder_path.exists():
            file_count = len(list(folder_path.glob("*")))
            print(f"{folder}: {file_count} files")
        else:
            print(f"{folder}: NOT FOUND")
    
    print("\n✓ Dataset organization complete!")
    print("="*60)

if __name__ == "__main__":
    organize_isaid_dataset()