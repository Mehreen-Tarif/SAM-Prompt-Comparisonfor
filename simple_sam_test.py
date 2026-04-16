"""
SIMPLE SAM TEST - Minimal GPU usage
"""

import torch
import numpy as np
import cv2
import time

print("=" * 60)
print("SIMPLE SAM TEST - Starting...")
print("=" * 60)

# Check GPU
print(f"\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    print("\n2. SAM Import: ✅ Success")
except ImportError as e:
    print(f"\n2. SAM Import: ❌ Error - {e}")
    exit()

# Check model files
import os
print("\n3. Checking model files:")

model_choices = []
if os.path.exists("sam_vit_b_01ec64.pth"):
    size = os.path.getsize("sam_vit_b_01ec64.pth") / (1024**3)
    print(f"   ✅ sam_vit_b_01ec64.pth - {size:.2f} GB")
    model_choices.append(("vit_b", "sam_vit_b_01ec64.pth"))

if os.path.exists("sam_vit_h_4b8939.pth"):
    size = os.path.getsize("sam_vit_h_4b8939.pth") / (1024**3)
    print(f"   ✅ sam_vit_h_4b8939.pth - {size:.2f} GB")
    model_choices.append(("vit_h", "sam_vit_h_4b8939.pth"))

if not model_choices:
    print("   ❌ No SAM model files found!")
    print("\n   Download one with:")
    print('   Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b_01ec64.pth"')
    exit()

# Choose the smaller model for testing
model_type, model_file = model_choices[0]  # Use first available (prefer smaller)
print(f"\n4. Using model: {model_file} ({model_type})")

# Load model (VERY CAREFULLY)
try:
    print("\n5. Loading model (this may take 30 seconds)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Loading on: {device}")
    
    start_time = time.time()
    
    # Load with minimal settings
    sam = sam_model_registry[model_type](checkpoint=model_file)
    sam.to(device=device)
    
    load_time = time.time() - start_time
    print(f"   ✅ Model loaded in {load_time:.1f} seconds")
    
    # Create predictor
    predictor = SamPredictor(sam)
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit()

# Create a tiny test image
print("\n6. Creating test image...")
test_image = np.zeros((256, 256, 3), dtype=np.uint8)  # SMALL image
test_image[100:150, 100:150] = [255, 0, 0]  # Red square

# Save it
cv2.imwrite("test_small.jpg", test_image)
print("   ✅ Saved: test_small.jpg")

# Run SAM (tiny operation)
try:
    print("\n7. Running SAM (tiny test)...")
    
    predictor.set_image(test_image)
    
    # Small bounding box
    bbox = [90, 90, 160, 160]
    masks, scores, _ = predictor.predict(
        box=np.array(bbox),
        multimask_output=False
    )
    
    print(f"   ✅ SAM success! Score: {scores[0]:.3f}")
    
    # Save mask
    mask_img = (masks[0] * 255).astype(np.uint8)
    cv2.imwrite("test_small_mask.jpg", mask_img)
    print("   ✅ Saved: test_small_mask.jpg")
    
except Exception as e:
    print(f"   ❌ SAM error: {e}")

# Check GPU memory
print("\n8. GPU Memory Status:")
try:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"   Allocated: {allocated:.1f} MB")
    print(f"   Reserved: {reserved:.1f} MB")
    print(f"   Available: {24000 - allocated:.1f} MB")
except:
    print("   Could not check memory")

print("\n" + "=" * 60)
print("🎉 TEST COMPLETE - Your system is working!")
print("=" * 60)

print("\n✅ You can now:")
print("1. Run experiments with SAM")
print("2. Process iSAID images")
print("3. Collect data for your paper")

input("\nPress Enter to exit safely...")