"""
Statistical analysis and visualization of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class ResultsAnalyzer:
    def __init__(self, results_path="results/confidence_scores.csv"):
        """Initialize analyzer with results data"""
        self.results = pd.read_csv(results_path)
        self.stats = None
        
    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        stats_list = []
        
        for obj_class in self.results['object_class'].unique():
            class_data = self.results[self.results['object_class'] == obj_class]
            
            # Calculate means and stds
            center_mean = class_data['center_point_score'].mean()
            center_std = class_data['center_point_score'].std()
            
            box_mean = class_data['bounding_box_score'].mean()
            box_std = class_data['bounding_box_score'].std()
            
            multi_mean = class_data['multiple_points_score'].mean()
            multi_std = class_data['multiple_points_score'].std()
            
            # Count wins for each strategy
            wins = {'center': 0, 'box': 0, 'multi': 0}
            for _, row in class_data.iterrows():
                scores = {
                    'center': row['center_point_score'],
                    'box': row['bounding_box_score'],
                    'multi': row['multiple_points_score']
                }
                best = max(scores, key=scores.get)
                wins[best] += 1
            
            # Perform t-tests
            t_test_box_vs_center = stats.ttest_rel(
                class_data['bounding_box_score'],
                class_data['center_point_score']
            )
            
            t_test_box_vs_multi = stats.ttest_rel(
                class_data['bounding_box_score'],
                class_data['multiple_points_score']
            )
            
            stats_list.append({
                'object_class': obj_class,
                'center_mean': center_mean,
                'center_std': center_std,
                'box_mean': box_mean,
                'box_std': box_std,
                'multi_mean': multi_mean,
                'multi_std': multi_std,
                'center_wins': wins['center'],
                'box_wins': wins['box'],
                'multi_wins': wins['multi'],
                'p_value_box_vs_center': t_test_box_vs_center.pvalue,
                'p_value_box_vs_multi': t_test_box_vs_multi.pvalue,
                'best_strategy': 'Bounding Box' if box_mean > center_mean and box_mean > multi_mean else
                               'Center Point' if center_mean > multi_mean else 'Multiple Points'
            })
        
        self.stats = pd.DataFrame(stats_list)
        return self.stats
    
    def generate_latex_table(self):
        """Generate LaTeX table for paper"""
        if self.stats is None:
            self.calculate_statistics()
        
        latex = """
\\begin{table}[ht]
\\centering
\\caption{Segmentation performance comparison across prompt strategies (n=10 per class). 
Values show mean ± standard deviation of SAM confidence scores. 
Best Count indicates per-instance wins: C=Center, B=Box, M=Multiple points.}
\\label{tab:sam_results}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Object Class} & \\textbf{Center Point} & \\textbf{Bounding Box} & \\textbf{Multiple Points} & \\textbf{Best Method} & \\textbf{Best Count} & \\textbf{N} \\\\
\\midrule
"""
        
        for _, row in self.stats.iterrows():
            latex += f"{row['object_class'].title().replace('_', ' '):<15} & "
            latex += f"{row['center_mean']:.3f} ± {row['center_std']:.3f} & "
            latex += f"{row['box_mean']:.3f} ± {row['box_std']:.3f} & "
            latex += f"{row['multi_mean']:.3f} ± {row['multi_std']:.3f} & "
            latex += f"{row['best_strategy']} & "
            latex += f"{row['center_wins']}C, {row['box_wins']}B, {row['multi_wins']}M & 10 \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        # Save LaTeX table
        with open("results/latex_table.txt", "w") as f:
            f.write(latex)
        
        return latex
    
    def create_figures(self):
        """Create all figures for the paper"""
        # Create figures directory
        os.makedirs("figures", exist_ok=True)
        
        # 1. Bar chart comparison
        self._create_bar_chart()
        
        # 2. Win rate chart
        self._create_win_rate_chart()
        
        # 3. Prompt strategies figure (conceptual)
        self._create_prompt_strategies_figure()
        
        print("Figures saved to 'figures/' directory")
    
    def _create_bar_chart(self):
        """Create bar chart comparing strategies"""
        if self.stats is None:
            self.calculate_statistics()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = self.stats['object_class']
        x = np.arange(len(classes))
        width = 0.25
        
        # Plot bars
        center_bars = ax.bar(x - width, self.stats['center_mean'], width, 
                           label='Center Point', color='#1f77b4', 
                           yerr=self.stats['center_std'], capsize=5)
        box_bars = ax.bar(x, self.stats['box_mean'], width, 
                        label='Bounding Box', color='#2ca02c',
                        yerr=self.stats['box_std'], capsize=5)
        multi_bars = ax.bar(x + width, self.stats['multi_mean'], width, 
                          label='Multiple Points', color='#ff7f0e',
                          yerr=self.stats['multi_std'], capsize=5)
        
        # Labels and formatting
        ax.set_xlabel('Object Class', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Average Segmentation Confidence by Object Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in classes])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.96, 1.03)
        
        plt.tight_layout()
        plt.savefig('figures/results_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_win_rate_chart(self):
        """Create win rate stacked bar chart"""
        if self.stats is None:
            self.calculate_statistics()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = self.stats['object_class']
        center_wins = self.stats['center_wins']
        box_wins = self.stats['box_wins']
        multi_wins = self.stats['multi_wins']
        
        # Convert to percentages
        total = 10  # 10 samples per class
        center_percent = center_wins / total * 100
        box_percent = box_wins / total * 100
        multi_percent = multi_wins / total * 100
        
        # Create stacked bars
        ax.bar(classes, center_percent, label='Center Point', color='#1f77b4')
        ax.bar(classes, box_percent, bottom=center_percent, label='Bounding Box', color='#2ca02c')
        ax.bar(classes, multi_percent, 
               bottom=[c+b for c,b in zip(center_percent, box_percent)], 
               label='Multiple Points', color='#ff7f0e')
        
        # Labels and formatting
        ax.set_ylabel('Percentage of Wins (%)', fontsize=12)
        ax.set_xlabel('Object Class', fontsize=12)
        ax.set_title('Per-Instance Strategy Win Rates', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (c, b, m) in enumerate(zip(center_percent, box_percent, multi_percent)):
            if c > 0:
                ax.text(i, c/2, f'{c:.0f}%', ha='center', va='center', 
                       color='white', fontweight='bold')
            if b > 0:
                ax.text(i, c + b/2, f'{b:.0f}%', ha='center', va='center', 
                       color='white', fontweight='bold')
            if m > 0:
                ax.text(i, c + b + m/2, f'{m:.0f}%', ha='center', va='center', 
                       color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('figures/win_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prompt_strategies_figure(self):
        """Create conceptual figure showing prompt strategies"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Panel 1: Center Point
        ax1 = axes[0]
        ax1.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.6, fill=True, color='gray', alpha=0.5))
        ax1.plot([0.5], [0.5], 'ro', markersize=15)
        ax1.text(0.5, 0.9, '(a) Center Point', ha='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Panel 2: Bounding Box
        ax2 = axes[1]
        ax2.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.6, fill=True, color='gray', alpha=0.5))
        ax2.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.6, fill=False, 
                                   edgecolor='green', linewidth=3))
        ax2.text(0.5, 0.9, '(b) Bounding Box', ha='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Panel 3: Multiple Points
        ax3 = axes[2]
        ax3.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.6, fill=True, color='gray', alpha=0.5))
        ax3.plot([0.2, 0.8, 0.2], [0.2, 0.2, 0.8], 'bo', markersize=15)
        ax3.text(0.5, 0.9, '(c) Multiple Points', ha='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        plt.suptitle('Prompt Strategies for SAM', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/prompt_strategies.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResultsAnalyzer("results/confidence_scores.csv")
    
    # Calculate statistics
    stats = analyzer.calculate_statistics()
    print(stats)
    
    # Generate LaTeX table
    latex_table = analyzer.generate_latex_table()
    print("\nLaTeX table generated and saved to 'results/latex_table.txt'")
    
    # Create figures
    analyzer.create_figures()
