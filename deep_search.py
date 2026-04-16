"""
Deep search for any files in iSAID dataset
"""

import os

print("=" * 80)
print("DEEP SEARCH IN iSAID DATASET")
print("=" * 80)

base_path = r"D:\BaiduNetdiskDownload"
print(f"Base path: {base_path}")

# Count all files
all_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        all_files.append(os.path.join(root, file))

print(f"\nTotal files found: {len(all_files)}")

# Group by extension
from collections import defaultdict
ext_count = defaultdict(int)

for file in all_files:
    _, ext = os.path.splitext(file)
    ext_count[ext.lower()] += 1

print("\nFiles by extension:")
for ext, count in sorted(ext_count.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {ext or 'no extension'}: {count}")

# Look specifically for image-like files
print("\nLooking for image files...")
image_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']
image_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in image_exts:
            full_path = os.path.join(root, file)
            image_files.append(full_path)

print(f"Found {len(image_files)} potential image files")

if image_files:
    print("\nFirst 5 image files:")
    for i, img_file in enumerate(image_files[:5]):
        size = os.path.getsize(img_file) / (1024**2)  # MB
        print(f"{i+1}. {os.path.relpath(img_file, base_path)} ({size:.1f} MB)")
        
        # Try to open it
        try:
            import cv2
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"   ✅ Can be opened by OpenCV: {img.shape}")
            else:
                print(f"   ❌ OpenCV cannot read it")
        except:
            print(f"   ⚠️  Error trying to open")

# Check for .zip or .rar files (might be compressed)
print("\nLooking for compressed files...")
compressed_exts = ['.zip', '.rar', '.7z', '.tar', '.gz']
compressed_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in compressed_exts:
            full_path = os.path.join(root, file)
            compressed_files.append(full_path)
            size = os.path.getsize(full_path) / (1024**3)  # GB
            print(f"  Found: {os.path.relpath(full_path, base_path)} ({size:.1f} GB)")

if compressed_files:
    print(f"\n⚠️  Found {len(compressed_files)} compressed files.")
    print("   The dataset might need to be extracted!")
    print("   Use WinRAR or 7-Zip to extract them.")

# Check for .json annotation files
print("\nLooking for annotation files...")
json_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith('.json'):
            json_files.append(os.path.join(root, file))

print(f"Found {len(json_files)} JSON files")
if json_files:
    for i, json_file in enumerate(json_files[:3]):
        print(f"  {i+1}. {os.path.relpath(json_file, base_path)}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

if image_files:
    print("✅ Images found! We can proceed with experiments.")
else:
    print("❌ No images found. The dataset might be:")
    print("   1. Not fully downloaded")
    print("   2. In compressed format (.zip, .rar)")
    print("   3. Corrupted")
    print("\n   Ask your labmate: 'Is the iSAID dataset fully extracted?'")

print("\nNext steps:")
print("1. Check if there are .zip/.rar files that need extraction")
print("2. Ask labmate for complete dataset")
print("3. Use synthetic data for now (we already have working code)")

input("\nPress Enter to exit...")