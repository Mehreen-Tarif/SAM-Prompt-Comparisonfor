"""
Check iSAID dataset structure at D:\BaiduNetdiskDownload
"""

import os

print("=" * 70)
print("CHECKING iSAID DATASET STRUCTURE")
print("=" * 70)

isaid_path = r"D:\BaiduNetdiskDownload"

if not os.path.exists(isaid_path):
    print(f"❌ Path not found: {isaid_path}")
    exit()

print(f"Path exists: {isaid_path}")
print("\nListing contents...")
print("-" * 70)

# List all items in the folder
items = os.listdir(isaid_path)
print(f"Total items in folder: {len(items)}")

# Show first 20 items
for i, item in enumerate(items[:20]):
    item_path = os.path.join(isaid_path, item)
    if os.path.isdir(item_path):
        print(f"{i+1:3}. 📁 {item}/")
        # Look for images inside
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            images = [f for f in os.listdir(item_path) if f.lower().endswith(ext)]
            if images:
                print(f"     Contains {len(images)} {ext} files")
                if len(images) > 0:
                    print(f"     Example: {images[0]}")
                break
    else:
        size_mb = os.path.getsize(item_path) / (1024**2)
        print(f"{i+1:3}. 📄 {item} ({size_mb:.1f} MB)")

# Search for iSAID specifically
print("\n" + "=" * 70)
print("SEARCHING FOR iSAID FOLDERS...")
print("-" * 70)

import glob
# Look for folders containing 'iSAID' in name
isaid_folders = []
for root, dirs, files in os.walk(isaid_path, topdown=True):
    for dir_name in dirs:
        if 'isaid' in dir_name.lower():
            full_path = os.path.join(root, dir_name)
            isaid_folders.append(full_path)
            print(f"✅ Found iSAID folder: {full_path}")
            
            # Check structure
            sub_items = os.listdir(full_path)[:5]
            for sub in sub_items:
                sub_path = os.path.join(full_path, sub)
                if os.path.isdir(sub_path):
                    print(f"    📁 {sub}/")
                else:
                    print(f"    📄 {sub}")

if not isaid_folders:
    print("❌ No folders with 'iSAID' in name found.")
    print("\nLet's search for common iSAID folder structure...")
    
    # Check for common iSAID structure
    possible_paths = [
        os.path.join(isaid_path, "iSAID"),
        os.path.join(isaid_path, "iSAID_Dataset"),
        os.path.join(isaid_path, "iSAID-dataset"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n✅ Found: {path}")
            # List contents
            for item in os.listdir(path)[:10]:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                else:
                    size_mb = os.path.getsize(item_path) / (1024**2)
                    print(f"   📄 {item} ({size_mb:.1f} MB)")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Check the folder structure above")
print("2. Look for 'images' and 'masks' folders")
print("3. Look for 'train', 'val', 'test' splits")
print("4. Count number of images")

input("\nPress Enter to continue...")