import torch
import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleSCIExperiment:
    """Simple but comprehensive SCI-level experiments"""
    
    def __init__(self):
        print("="*80)
        print("SIMPLE SCI EXPERIMENTS FOR PAPER")
        print("="*80)
        
        # Setup
        self.device = "cuda"
        self.model_type = "vit_h"
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load model
        print(f"\nLoading SAM {self.model_type}...")
        self.sam = sam_model_registry[self.model_type](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Create directories
        Path("results").mkdir(exist_ok=True)
        Path("figures").mkdir(exist_ok=True)
        
        print(f"✓ Experiment ID: {self.exp_id}")
        print(f"✓ Device: {self.device}")
        print("="*80)
    
    def run_experiments(self):
        """Run all experiments"""
        print("\nStarting experiments...")
        
        # Experiment 1: Prompt strategy comparison
        print("\n" + "-"*40)
        print("EXPERIMENT 1: Prompt Strategies")
        print("-"*40)
        exp1_df = self.experiment_prompt_strategies(num_samples=50)
        
        # Experiment 2: Quick robustness test
        print("\n" + "-"*40)
        print("EXPERIMENT 2: Robustness Test")
        print("-"*40)
        exp2_df = self.experiment_robustness(num_samples=20)
        
        # Generate report
        self.generate_report(exp1_df, exp2_df)
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETED!")
        print("="*80)
        print("\nResults saved in:")
        print("  - results/ folder (CSV files)")
        print("  - figures/ folder (PNG files)")
        print("\nNow you can update your paper with these results!")
    
    def experiment_prompt_strategies(self, num_samples=50):
        """Compare different prompt strategies"""
        print(f"Testing {num_samples} samples...")
        
        # Get image files
        img_dir = Path("data/iSAID/train/images")
        mask_dir = Path("data/iSAID/train/masks")
        
        img_files = list(img_dir.glob("*.png"))[:num_samples]
        
        strategies = ['center', 'bbox', 'multi_point']
        results = []
        
        for i, img_path in enumerate(tqdm(img_files, desc="Processing")):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load or create mask
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8) * 255
            else:
                # Create synthetic mask
                h, w = image.shape[:2]
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(gt_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
            # Set image in predictor
            self.predictor.set_image(image)
            
            for strategy in strategies:
                prompt = self.generate_prompt(gt_mask, strategy)
                if prompt is None:
                    continue
                
                try:
                    masks, scores, _ = self.predictor.predict(**prompt)
                    
                    if len(masks) > 0:
                        pred_mask = masks[0]
                        confidence = float(scores[0])
                        
                        # Calculate IoU
                        iou = self.calculate_iou(pred_mask, gt_mask)
                        
                        results.append({
                            'image': img_path.stem,
                            'strategy': strategy,
                            'iou': iou,
                            'confidence': confidence
                        })
                except:
                    continue
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"results/exp1_prompt_strategies_{self.exp_id}.csv", index=False)
        
        # Create visualization
        self.plot_prompt_comparison(df)
        
        return df
    
    def experiment_robustness(self, num_samples=20):
        """Test robustness to image variations"""
        print(f"Testing {num_samples} samples with variations...")
        
        img_dir = Path("data/iSAID/val/images")
        mask_dir = Path("data/iSAID/val/masks")
        
        img_files = list(img_dir.glob("*.png"))[:num_samples]
        
        variations = [
            ('original', lambda x: x),
            ('noise', self.add_noise),
            ('blur', self.add_blur),
            ('bright', self.increase_brightness)
        ]
        
        results = []
        
        for i, img_path in enumerate(tqdm(img_files, desc="Robustness test")):
            # Load image and mask
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8) * 255
            else:
                h, w = image.shape[:2]
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(gt_mask, (w//2, h//2), min(w, h)//4, 255, -1)
            
            for var_name, transform in variations:
                transformed = transform(image.copy())
                self.predictor.set_image(transformed)
                
                prompt = self.generate_prompt(gt_mask, 'bbox')
                if prompt is None:
                    continue
                
                try:
                    masks, scores, _ = self.predictor.predict(**prompt)
                    
                    if len(masks) > 0:
                        pred_mask = masks[0]
                        iou = self.calculate_iou(pred_mask, gt_mask)
                        
                        results.append({
                            'image': img_path.stem,
                            'variation': var_name,
                            'iou': iou,
                            'confidence': float(scores[0])
                        })
                except:
                    continue
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"results/exp2_robustness_{self.exp_id}.csv", index=False)
        
        # Create visualization
        self.plot_robustness(df)
        
        return df
    
    def generate_prompt(self, mask, strategy):
        """Generate prompt from mask"""
        if mask.sum() == 0:
            return None
        
        coords = np.column_stack(np.where(mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        if strategy == 'center':
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            return {'points': [[x_center, y_center]], 'labels': [1]}
        elif strategy == 'bbox':
            return {'bbox': [x_min, y_min, x_max, y_max]}
        elif strategy == 'multi_point':
            points = [
                [x_min, y_min], [x_max, y_min],
                [x_min, y_max], [x_max, y_max]
            ]
            return {'points': points, 'labels': [1, 1, 1, 1]}
        
        return None
    
    def calculate_iou(self, pred_mask, gt_mask):
        """Calculate Intersection over Union"""
        pred_bin = (pred_mask > 0.5).astype(np.uint8)
        gt_bin = (gt_mask > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        
        return intersection / (union + 1e-8)
    
    def add_noise(self, image):
        """Add Gaussian noise"""
        noise = np.random.normal(0, 25, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def add_blur(self, image):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def increase_brightness(self, image):
        """Increase brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.3, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def plot_prompt_comparison(self, df):
        """Create visualization for prompt comparison"""
        plt.figure(figsize=(12, 5))
        
        # Plot 1: IoU comparison
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='strategy', y='iou')
        plt.title('IoU by Prompt Strategy', fontweight='bold')
        plt.xlabel('Strategy')
        plt.ylabel('Intersection over Union')
        
        # Plot 2: Confidence comparison
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='strategy', y='confidence')
        plt.title('Confidence by Prompt Strategy', fontweight='bold')
        plt.xlabel('Strategy')
        plt.ylabel('SAM Confidence')
        
        plt.tight_layout()
        plt.savefig(f'figures/prompt_comparison_{self.exp_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Prompt comparison figure saved")
    
    def plot_robustness(self, df):
        """Create visualization for robustness test"""
        plt.figure(figsize=(10, 5))
        
        # Calculate relative performance
        original_ious = {}
        for _, row in df[df['variation'] == 'original'].iterrows():
            original_ious[row['image']] = row['iou']
        
        # Add relative iou column
        df['relative_iou'] = df.apply(
            lambda x: x['iou'] / original_ious.get(x['image'], x['iou']),
            axis=1
        )
        
        # Plot
        variation_order = ['original', 'noise', 'blur', 'bright']
        df_filtered = df[df['variation'].isin(variation_order)]
        
        sns.boxplot(data=df_filtered, x='variation', y='relative_iou')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        plt.title('Robustness to Image Variations\n(Relative to Original)', fontweight='bold')
        plt.xlabel('Variation Type')
        plt.ylabel('Relative IoU')
        
        plt.tight_layout()
        plt.savefig(f'figures/robustness_{self.exp_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Robustness figure saved")
    
    def generate_report(self, exp1_df, exp2_df):
        """Generate simple report"""
        report = f"""EXPERIMENT REPORT
=================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment ID: {self.exp_id}
Model: SAM {self.model_type}
Device: {self.device}

EXPERIMENT 1: Prompt Strategies
--------------------------------
Total evaluations: {len(exp1_df)}
"""
        
        # Add strategy performance
        if not exp1_df.empty:
            best_by_iou = exp1_df.groupby('strategy')['iou'].mean().idxmax()
            best_iou = exp1_df.groupby('strategy')['iou'].mean().max()
            
            report += f"Best strategy by IoU: {best_by_iou} ({best_iou:.3f})\n\n"
            
            report += "Average performance by strategy:\n"
            for strategy in ['center', 'bbox', 'multi_point']:
                subset = exp1_df[exp1_df['strategy'] == strategy]
                if not subset.empty:
                    avg_iou = subset['iou'].mean()
                    avg_conf = subset['confidence'].mean()
                    report += f"  {strategy}: IoU={avg_iou:.3f}, Confidence={avg_conf:.3f}\n"
        
        report += f"""
EXPERIMENT 2: Robustness
------------------------
Total evaluations: {len(exp2_df)}
"""
        
        # Add robustness summary
        if not exp2_df.empty:
            for variation in ['original', 'noise', 'blur', 'bright']:
                subset = exp2_df[exp2_df['variation'] == variation]
                if not subset.empty:
                    avg_iou = subset['iou'].mean()
                    report += f"  {variation}: Avg IoU={avg_iou:.3f}\n"
        
        # Save report
        with open(f'results/report_{self.exp_id}.txt', 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved: results/report_{self.exp_id}.txt")

# Run the experiments
if __name__ == "__main__":
    try:
        experiment = SimpleSCIExperiment()
        experiment.run_experiments()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
