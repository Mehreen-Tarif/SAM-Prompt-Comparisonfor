"""
Evaluate SAM with three prompt strategies
"""

import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import json
from tqdm import tqdm

class SAMEvaluator:
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """
        Initialize SAM model
        
        Args:
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download SAM checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = "sam_vit_h_4b8939.pth"
            if not Path(checkpoint_path).exists():
                print("Downloading SAM checkpoint...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                urllib.request.urlretrieve(url, checkpoint_path)
        
        # Load SAM model
        print(f"Loading SAM model ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        
    def evaluate_single_image(self, image_path, annotations):
        """
        Evaluate SAM on a single image with three prompt strategies
        
        Returns:
            Dictionary with confidence scores for each strategy
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        results = {}
        
        # 1. Center Point Prompt
        center_point = np.array([annotations['center_point']])
        center_label = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True,
        )
        results['center_point_score'] = float(scores[0])
        
        # 2. Bounding Box Prompt
        bbox = np.array(annotations['bbox'])
        masks, scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=True,
        )
        results['bounding_box_score'] = float(scores[0])
        
        # 3. Multiple Points Prompt
        points = np.array(annotations['multiple_points'])
        labels = np.array([1, 1, 1])
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        results['multiple_points_score'] = float(scores[0])
        
        return results
    
    def evaluate_dataset(self, metadata_path, output_path="results/confidence_scores.csv"):
        """
        Evaluate complete dataset
        
        Args:
            metadata_path: Path to dataset metadata JSON
            output_path: Path to save results
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        results = []
        
        print("Evaluating SAM on dataset...")
        for item in tqdm(metadata):
            try:
                # Get scores
                scores = self.evaluate_single_image(item['image_path'], item)
                
                # Combine with metadata
                result = {
                    'image_path': item['image_path'],
                    'object_class': item['class'],
                    'background': item['background'],
                    'center_point_score': scores['center_point_score'],
                    'bounding_box_score': scores['bounding_box_score'],
                    'multiple_points_score': scores['multiple_points_score'],
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Calculate summary statistics
        self.calculate_statistics(df)
        
        return df
    
    def calculate_statistics(self, df):
        """Calculate and print summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Group by object class
        grouped = df.groupby('object_class')
        
        stats = []
        for name, group in grouped:
            row = {
                'object_class': name,
                'center_mean': group['center_point_score'].mean(),
                'center_std': group['center_point_score'].std(),
                'box_mean': group['bounding_box_score'].mean(),
                'box_std': group['bounding_box_score'].std(),
                'multi_mean': group['multiple_points_score'].mean(),
                'multi_std': group['multiple_points_score'].std(),
            }
            
            # Determine best strategy
            means = {
                'Center Point': row['center_mean'],
                'Bounding Box': row['box_mean'],
                'Multiple Points': row['multi_mean']
            }
            best_strategy = max(means, key=means.get)
            row['best_strategy'] = best_strategy
            
            stats.append(row)
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Save statistics
        stats_path = "results/statistical_analysis.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics saved to {stats_path}")
        
        # Print table
        print("\nAverage Confidence Scores by Class:")
        print("-"*70)
        print(f"{'Class':<15} {'Center Point':<15} {'Bounding Box':<15} {'Multiple Points':<12} {'Best'}")
        print("-"*70)
        
        for _, row in stats_df.iterrows():
            print(f"{row['object_class']:<15} "
                  f"{row['center_mean']:.3f}±{row['center_std']:.3f}  "
                  f"{row['box_mean']:.3f}±{row['box_std']:.3f}  "
                  f"{row['multi_mean']:.3f}±{row['multi_std']:.3f}  "
                  f"{row['best_strategy']}")

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = SAMEvaluator()
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        metadata_path="data/synthetic_dataset/metadata.json",
        output_path="results/confidence_scores.csv"
    )
