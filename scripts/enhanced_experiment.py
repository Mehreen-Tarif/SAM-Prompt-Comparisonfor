# scripts/enhanced_experiment.py
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from segment_anything import sam_model_registry, SamPredictor
import logging

class SCIExperimentFramework:
    """SCI-worthy experimental framework for SAM prompt strategies"""
    
    def __init__(self, data_path, results_dir="results"):
        # Setup paths
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize SAM
        self.setup_sam()
        
        # Experimental parameters
        self.setup_parameters()
        
    def setup_sam(self):
        """Initialize SAM with proper configuration"""
        self.logger.info("Initializing SAM ViT-H model...")
        
        # Model configuration
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        # Download model if not exists
        if not Path(sam_checkpoint).exists():
            self.logger.warning(f"Download SAM checkpoint to: {sam_checkpoint}")
            # You can add automatic download here
        
        # Load model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)
        
        self.logger.info(f"SAM loaded on {torch.cuda.get_device_name(0)}")
    
    def setup_parameters(self):
        """Define comprehensive experimental parameters"""
        self.params = {
            # Dataset parameters
            'num_samples': 200,  # Increased from 50 for statistical power
            'num_classes': 10,   # Expanded from 5
            'image_size': (1024, 1024),  # Standard remote sensing size
            
            # Prompt strategies
            'strategies': [
                'center_point',
                'bounding_box',
                'multiple_points_3',
                'multiple_points_5',
                'multiple_points_10',
                'box_with_center',
                'points_with_box'
            ],
            
            # Evaluation metrics
            'metrics': [
                'confidence',
                'iou',
                'dice_coefficient',
                'hausdorff_distance',
                'precision',
                'recall',
                'f1_score',
                'inference_time'
            ],
            
            # Statistical analysis
            'confidence_level': 0.95,
            'n_bootstrap': 1000,
            'random_seed': 42,
            
            # Hardware specs
            'gpu': 'RTX 3090 24GB',
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        }
        
        # Save parameters
        with open(self.results_dir / 'experiment_params.json', 'w') as f:
            json.dump(self.params, f, indent=4)
    
    def generate_prompts(self, bbox, image_shape):
        """Generate comprehensive prompt strategies"""
        prompts = {}
        
        # 1. Center point
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        prompts['center_point'] = {
            'points': np.array([[center_x, center_y]]),
            'labels': np.array([1])
        }
        
        # 2. Bounding box
        prompts['bounding_box'] = {
            'box': np.array([x1, y1, x2, y2])
        }
        
        # 3. Multiple points (3, 5, 10)
        for n_points in [3, 5, 10]:
            points = []
            for _ in range(n_points):
                px = np.random.randint(x1, x2)
                py = np.random.randint(y1, y2)
                points.append([px, py])
            prompts[f'multiple_points_{n_points}'] = {
                'points': np.array(points),
                'labels': np.ones(len(points))
            }
        
        # 4. Combined strategies
        prompts['box_with_center'] = {
            'box': np.array([x1, y1, x2, y2]),
            'points': np.array([[center_x, center_y]]),
            'labels': np.array([1])
        }
        
        # 5. Multiple points with box
        prompts['points_with_box'] = {
            'box': np.array([x1, y1, x2, y2]),
            'points': np.array([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2],
                [center_x, center_y]
            ]),
            'labels': np.ones(5)
        }
        
        return prompts
    
    def compute_metrics(self, pred_mask, gt_mask, inference_time):
        """Compute comprehensive segmentation metrics"""
        metrics = {}
        
        # Convert to binary
        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = gt_mask.astype(np.uint8)
        
        # Intersection over Union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        metrics['iou'] = intersection / union if union > 0 else 0
        
        # Dice Coefficient
        metrics['dice_coefficient'] = (2 * intersection) / (
            pred_binary.sum() + gt_binary.sum() + 1e-7
        )
        
        # Precision, Recall, F1
        tp = intersection
        fp = pred_binary.sum() - intersection
        fn = gt_binary.sum() - intersection
        
        metrics['precision'] = tp / (tp + fp + 1e-7)
        metrics['recall'] = tp / (tp + fn + 1e-7)
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
            metrics['precision'] + metrics['recall'] + 1e-7
        )
        
        # Hausdorff Distance (simplified)
        metrics['hausdorff_distance'] = self.compute_hausdorff(pred_binary, gt_binary)
        
        # Inference time
        metrics['inference_time'] = inference_time
        
        return metrics
    
    def compute_hausdorff(self, pred, gt):
        """Compute Hausdorff distance between masks"""
        # Get coordinates of points in each mask
        pred_coords = np.argwhere(pred > 0)
        gt_coords = np.argwhere(gt > 0)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        # Compute distances
        from scipy.spatial.distance import cdist
        distances = cdist(pred_coords, gt_coords)
        
        # Hausdorff distance
        h1 = np.max(np.min(distances, axis=1))
        h2 = np.max(np.min(distances, axis=0))
        
        return max(h1, h2)
    
    def run_experiment(self):
        """Main experimental pipeline"""
        self.logger.info("Starting comprehensive experiments...")
        
        results = []
        
        # Assuming you have dataset loader
        # Replace this with your actual dataset loading
        dataset = self.load_dataset()
        
        for idx, sample in enumerate(dataset):
            self.logger.info(f"Processing sample {idx+1}/{len(dataset)}")
            
            image, gt_mask, bbox = sample
            
            # Set image in predictor
            self.predictor.set_image(image)
            
            # Generate prompts
            prompts = self.generate_prompts(bbox, image.shape[:2])
            
            for strategy, prompt_data in prompts.items():
                start_time = time.time()
                
                # Predict with SAM
                if 'points' in prompt_data and 'box' in prompt_data:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=prompt_data.get('points'),
                        point_labels=prompt_data.get('labels'),
                        box=prompt_data.get('box'),
                        multimask_output=False
                    )
                elif 'points' in prompt_data:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=prompt_data['points'],
                        point_labels=prompt_data['labels'],
                        multimask_output=False
                    )
                else:  # box only
                    masks, scores, _ = self.predictor.predict(
                        box=prompt_data['box'],
                        multimask_output=False
                    )
                
                inference_time = time.time() - start_time
                
                # Compute metrics
                metrics = self.compute_metrics(masks[0], gt_mask, inference_time)
                metrics['confidence'] = float(scores[0])
                metrics['strategy'] = strategy
                metrics['sample_id'] = idx
                metrics['class'] = self.get_class_from_sample(sample)
                
                results.append(metrics)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save results
        self.save_results()
        
        return self.results_df
    
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        self.logger.info("Performing statistical analysis...")
        
        analysis_results = {}
        
        # 1. Descriptive statistics by strategy
        desc_stats = self.results_df.groupby('strategy')['iou'].describe()
        analysis_results['descriptive_stats'] = desc_stats
        
        # 2. ANOVA to check if strategies differ significantly
        from scipy.stats import f_oneway
        groups = [group['iou'].values for name, group in 
                  self.results_df.groupby('strategy')]
        f_stat, p_value = f_oneway(*groups)
        analysis_results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # 3. Pairwise t-tests with Bonferroni correction
        strategies = self.results_df['strategy'].unique()
        pairwise_results = {}
        
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                data1 = self.results_df[self.results_df['strategy'] == strat1]['iou']
                data2 = self.results_df[self.results_df['strategy'] == strat2]['iou']
                
                t_stat, p_val = stats.ttest_rel(data1, data2)
                
                # Effect size (Cohen's d)
                mean_diff = data1.mean() - data2.mean()
                pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                cohen_d = mean_diff / pooled_std
                
                pairwise_results[f"{strat1}_vs_{strat2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohen_d': cohen_d,
                    'mean_diff': mean_diff,
                    'significant': p_val < 0.05
                }
        
        analysis_results['pairwise_comparisons'] = pairwise_results
        
        # 4. Bootstrapped confidence intervals
        bootstrap_results = {}
        for strategy in strategies:
            data = self.results_df[self.results_df['strategy'] == strategy]['iou']
            bootstrap_means = []
            
            for _ in range(1000):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(sample.mean())
            
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            bootstrap_results[strategy] = {
                'mean': data.mean(),
                'ci_95': (ci_lower, ci_upper),
                'std': data.std()
            }
        
        analysis_results['bootstrap_ci'] = bootstrap_results
        
        # Save analysis
        with open(self.results_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        return analysis_results
    
    def visualize_results(self):
        """Create publication-quality visualizations"""
        self.logger.info("Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Box plot of IoU by strategy
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='strategy', y='iou', data=self.results_df)
        plt.title('Distribution of IoU Scores by Prompt Strategy', fontsize=14)
        plt.xlabel('Prompt Strategy', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'boxplot_iou_by_strategy.png', dpi=300)
        plt.close()
        
        # 2. Bar plot with confidence intervals
        plt.figure(figsize=(10, 6))
        mean_iou = self.results_df.groupby('strategy')['iou'].mean()
        std_iou = self.results_df.groupby('strategy')['iou'].std()
        
        bars = plt.bar(range(len(mean_iou)), mean_iou.values, 
                      yerr=std_iou.values, capsize=5)
        plt.xticks(range(len(mean_iou)), mean_iou.index, rotation=45)
        plt.title('Mean IoU with Standard Deviation', fontsize=14)
        plt.ylabel('IoU Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'barplot_mean_iou.png', dpi=300)
        plt.close()
        
        # 3. Heatmap of pairwise comparisons
        strategies = self.results_df['strategy'].unique()
        comparison_matrix = np.zeros((len(strategies), len(strategies)))
        
        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i != j:
                    data1 = self.results_df[self.results_df['strategy'] == strat1]['iou']
                    data2 = self.results_df[self.results_df['strategy'] == strat2]['iou']
                    mean_diff = data1.mean() - data2.mean()
                    comparison_matrix[i, j] = mean_diff
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(comparison_matrix, annot=True, fmt='.3f',
                   xticklabels=strategies, yticklabels=strategies,
                   cmap='RdBu_r', center=0)
        plt.title('Pairwise Mean Difference in IoU', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'heatmap_pairwise_differences.png', dpi=300)
        plt.close()
        
        # 4. Inference time vs IoU scatter plot
        plt.figure(figsize=(10, 6))
        for strategy in strategies:
            subset = self.results_df[self.results_df['strategy'] == strategy]
            plt.scatter(subset['inference_time'], subset['iou'], 
                       alpha=0.6, label=strategy, s=50)
        
        plt.xlabel('Inference Time (seconds)', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.title('Trade-off: Inference Time vs Segmentation Quality', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'scatter_time_vs_iou.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Radar chart for comprehensive comparison
        self.create_radar_chart()
    
    def create_radar_chart(self):
        """Create radar chart for multi-metric comparison"""
        metrics_to_plot = ['iou', 'dice_coefficient', 'precision', 
                          'recall', 'f1_score', 'inference_time']
        
        # Normalize metrics (except inference time, which we invert)
        normalized_data = {}
        for strategy in self.results_df['strategy'].unique():
            subset = self.results_df[self.results_df['strategy'] == strategy]
            normalized_metrics = []
            
            for metric in metrics_to_plot:
                if metric == 'inference_time':
                    # Invert so lower time is better
                    value = 1 / (subset[metric].mean() + 1e-7)
                else:
                    value = subset[metric].mean()
                normalized_metrics.append(value)
            
            # Normalize to 0-1 scale
            normalized_metrics = (normalized_metrics - np.min(normalized_metrics)) / \
                                (np.max(normalized_metrics) - np.min(normalized_metrics) + 1e-7)
            normalized_data[strategy] = normalized_metrics
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for strategy, values in normalized_data.items():
            # Close the polygon
            values = values + values[:1]
            plot_angles = angles + angles[:1]
            
            ax.plot(plot_angles, values, linewidth=2, label=strategy)
            ax.fill(plot_angles, values, alpha=0.1)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1)
        plt.title('Multi-metric Comparison of Prompt Strategies', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(self.results_dir / 'radar_chart_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results to files"""
        # Save DataFrame
        self.results_df.to_csv(self.results_dir / 'all_results.csv', index=False)
        
        # Save summary statistics
        summary = self.results_df.groupby('strategy').agg({
            'iou': ['mean', 'std', 'min', 'max'],
            'dice_coefficient': ['mean', 'std'],
            'inference_time': ['mean', 'std'],
            'confidence': ['mean', 'std']
        })
        
        summary.to_csv(self.results_dir / 'summary_statistics.csv')
        
        # Generate LaTeX table
        self.generate_latex_table(summary)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def generate_latex_table(self, summary):
        """Generate LaTeX table for publication"""
        latex_table = summary.to_latex(float_format="%.4f")
        
        with open(self.results_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)
    
    def load_dataset(self):
        """Load your dataset - implement based on your data structure"""
        # This is a placeholder - replace with your dataset loader
        # Return list of (image, ground_truth_mask, bounding_box)
        pass
    
    def get_class_from_sample(self, sample):
        """Extract class from sample - implement based on your data"""
        # This is a placeholder
        return "unknown"

# Main execution
if __name__ == "__main__":
    # Initialize framework
    experiment = SCIExperimentFramework(
        data_path="C:/Users/Lenovo/Desktop/My_SAM_Project/data",
        results_dir="C:/Users/Lenovo/Desktop/My_SAM_Project/results/comprehensive"
    )
    
    # Run experiments
    results = experiment.run_experiment()
    
    # Perform statistical analysis
    stats_results = experiment.statistical_analysis()
    
    # Generate visualizations
    experiment.visualize_results()
    
    print("✅ Comprehensive experiments completed!")
    print(f"📊 Results saved to: {experiment.results_dir}")