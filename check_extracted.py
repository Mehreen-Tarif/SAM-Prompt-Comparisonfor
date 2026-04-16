"""
Check extracted iSAID dataset contents
"""

import os

base = r"D:\BaiduNetdiskDownload"
print("=" * 80)
print("CHECKING EXTRACTED iSAID DATASET")
print("=" * 80)

print(f"Base directory: {base}")

# Check all extracted folders
folders_to_check = [
    r"train\Instance_masks\images",
    r"train\Semantic_masks\images", 
    r"val\Instance_masks\images",
    r"val\Semantic_masks\images",
    r"test\images\part1",  # These might not be extracted yet
    r"test\images\part2"
]

total_images = 0
for folder in folders_to_check:
    full_path = os.path.join(base, folder)
    if os.path.exists(full_path):
        # Count image files
        images = []
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    images.append(os.path.join(root, file))
        
        print(f"\n📁 {folder}:")
        print(f"   Found {len(images)} images")
        if images:
            # Show first 3 images
            for i, img in enumerate(images[:3]):
                filename = os.path.basename(img)
                size_kb = os.path.getsize(img) / 1024
                print(f"   {i+1}. {filename} ({size_kb:.1f} KB)")
            total_images += len(images)
    else:
        print(f"\n📁 {folder}: ❌ Not found")

print("\n" + "=" * 80)
print(f"TOTAL IMAGES FOUND: {total_images}")
print("=" * 80)

# Also check for the original ZIP files that might still need extraction
import glob
zip_files = glob.glob(os.path.join(base, "**", "*.zip"), recursive=True)
if zip_files:
    print(f"\n⚠️  There are still {len(zip_files)} ZIP files that might need extraction:")
    for zip_file in zip_files[:3]:  # Show first 3
        size_gb = os.path.getsize(zip_file) / (1024**3)
        print(f"   • {os.path.relpath(zip_file, base)} ({size_gb:.1f} GB)")
    if len(zip_files) > 3:
        print(f"   ... and {len(zip_files) - 3} more")

print("\n" + "=" * 80)
print("RECOMMENDED NEXT STEPS:")
print("=" * 80)

if total_images > 0:
    print("✅ You have extracted images! You can now:")
    print("   1. Run experiments on real iSAID data")
    print("   2. Create paper figures with real remote sensing images")
    print("   3. Compare with synthetic results")
else:
    print("❌ No images found. You need to:")
    print("   1. Extract the test images (part1.zip and part2.zip)")
    print("   2. Check if images are in a different format")

print("\nLet's extract the remaining ZIP files if needed...")

# Check if test images need extraction
test_zip1 = os.path.join(base, "test", "images", "part1.zip")
test_zip2 = os.path.join(base, "test", "images", "part2.zip")

if os.path.exists(test_zip1):
    print(f"\nFound: part1.zip ({os.path.getsize(test_zip1) / (1024**3):.1f} GB)")
    print("This contains the main test images!")
    
if os.path.exists(test_zip2):
    print(f"Found: part2.zip ({os.path.getsize(test_zip2) / (1024**3):.1f} GB)")
    print("This contains the main test images!")

print("\n" + "=" * 80)
print("ACTION REQUIRED:")
print("=" * 80)
print("1. Extract part1.zip and part2.zip from test\images\ folder")
print("2. Then we can run real experiments")

input("\nPress Enter to continue...")