print("SAFE TEST 1 - Only checking packages")
print("=" * 50)

# Test 1: Basic Python
print("1. Python test... OK")

# Test 2: Check if files exist
import os
print("\n2. Checking for important files:")
files = ["sam_vit_b_01ec64.pth", "sam_vit_h_4b8939.pth"]
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024**3)
        print(f"   ✅ {file} - {size:.2f} GB")
    else:
        print(f"   ❌ {file} - Not found")

print("\n✅ Test complete - no GPU operations")
input("Press Enter to exit...")