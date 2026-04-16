import torch
import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

print("="*80)
print("STARTING SCI-LEVEL EXPERIMENTS")
print("="*80)

# 1. Check system
print("\n1. System Check:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 2. Load SAM
print("\n2. Loading SAM model...")
try:
    checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    print("✓ SAM loaded successfully")
except Exception as e:
    print(f"✗ Error loading SAM: {e}")
    exit()

# 3. Check dataset
print("\n3. Checking dataset...")
data_path = Path("data/iSAID")
if data_path.exists():
    train_images = len(list((data_path / "train/images").glob("*.png")))
    train_masks = len(list((data_path / "train/masks").glob("*.png")))
    val_images = len(list((data_path / "val/images").glob("*.png")))
    val_masks = len(list((data_path / "val/masks").glob("*.png")))
    
    print(f"✓ Dataset found:")
    print(f"  Train: {train_images} images, {train_masks} masks")
    print(f"  Val: {val_images} images, {val_masks} masks")
else:
    print("✗ Dataset not found. Make sure you ran create_working_dataset.py")
    exit()

print("\n" + "="*80)
print("SYSTEM READY! Starting experiments...")
print("="*80)
