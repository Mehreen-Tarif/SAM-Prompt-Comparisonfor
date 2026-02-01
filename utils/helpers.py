#!/usr/bin/env python3
"""
Utility functions for SAM Remote Sensing Project
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def plot_results(strategies, iou_scores, save_path=None):
    """Plot bar chart of strategy performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(strategies, iou_scores)
    ax.set_ylabel('IoU Score', fontsize=12)
    ax.set_title('SAM Prompt Strategy Performance', fontsize=14)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, iou_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def print_statistics(results):
    """Print formatted statistics."""
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")