"""
Explore iSAID dataset structure
"""

import os
import cv2
import numpy as np

print("=" * 70)
print("EXPLORING iSAID DATASET STRUCTURE")
print("=" * 70)

base_path = r"D:\BaiduNetdiskDownload"
print(f"Base path: {base_path}")

# Check each split
splits = ["train", "val", "test"]

for split in splits:
    split_path = os.path.join(base_path, split)
    print(f"\n📁 {split.upper()} folder:")
    print("-" * 40)
    
    if not os.path.exists(split_path):
        print(f"   ❌ Not found: {split_path}")
        continue
    
    # List contents
    items = os.listdir(split_path)
    print(f"   Items in {split}: {len(items)}")
    
    # Check for common iSAID structure
    has_images = False
    has_masks = False
    
    for item in items[:10]:  # Show first 10
        item_path = os.path.join(split_path, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
            # Check if it's images or masks
            if 'image' in item.lower() or 'img' in item.lower():
                has_images = True
                # Count images
                images = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"      Contains {len(images)} images")
            elif 'mask' in item.lower() or 'label' in item.lower():
                has_masks = True
                masks = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"      Contains {len(masks)} masks")
        else:
            # If files are directly in split folder
            if item.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                has_images = True
                print(f"   📄 {item} (image)")
    
    if not has_images:
        # Maybe images are in subfolders
        print(f"   Looking for images recursively...")
        image_count = 0
        for root, dirs, files in os.walk(split_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_count += 1
                    if image_count == 1:
                        print(f"   Found image: {os.path.relpath(os.path.join(root, file), split_path)}")
        print(f"   Total images in {split}: {image_count}")

print("\n" + "=" * 70)
print("SAMPLING iSAID IMAGES")
print("=" * 70)

# Let's find and display a few images from train folder
train_path = os.path.join(base_path, "train")

# Find first few images
image_paths = []
for root, dirs, files in os.walk(train_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))
            if len(image_paths) >= 5:
                break
    if len(image_paths) >= 5:
        break

print(f"\nFound {len(image_paths)} images in train folder")
for i, img_path in enumerate(image_paths):
    rel_path = os.path.relpath(img_path, base_path)
    print(f"{i+1}. {rel_path}")
    
    # Try to load and show info
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 2:
                channels = 1
            else:
                channels = img.shape[2]
            print(f"   Size: {img.shape[1]}x{img.shape[0]}, Channels: {channels}")
            
            # Save first image as sample
            if i == 0:
                sample_path = "isaid_sample_image.jpg"
                if channels == 1:
                    cv2.imwrite(sample_path, img)
                else:
                    cv2.imwrite(sample_path, img)
                print(f"   ✅ Saved as: {sample_path}")
        else:
            print(f"   ❌ Could not read image")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("NEXT STEP: Run real experiments on iSAID images!")
print("=" * 70)

print("\nTo use iSAID in your paper:")
print("1. We'll load images from these folders")
print("2. Apply SAM with 3 prompt strategies")
print("3. Collect results for your paper tables")
print("4. Create figures with real remote sensing images")

input("\nPress Enter to continue...")