# create_paper_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# Load your results
df = pd.read_csv('experiment_results/sam_experiment_results.csv')

# 1. Main comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
order = df.groupby('strategy')['iou'].mean().sort_values(ascending=False).index
sns.boxplot(x='strategy', y='iou', data=df, ax=axes[0], order=order)
axes[0].set_title('(a) IoU Distribution by Prompt Strategy', fontweight='bold')
axes[0].set_xlabel('Prompt Strategy')
axes[0].set_ylabel('IoU Score')
axes[0].tick_params(axis='x', rotation=45)

# Bar plot with error bars
summary = df.groupby('strategy').agg({
    'iou': ['mean', 'std', 'count'],
    'inference_time': ['mean']
}).round(4)

means = summary[('iou', 'mean')].reindex(order)
stds = summary[('iou', 'std')].reindex(order)

x_pos = np.arange(len(means))
axes[1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
axes[1].set_title('(b) Mean IoU with Standard Deviation', fontweight='bold')
axes[1].set_xlabel('Prompt Strategy')
axes[1].set_ylabel('IoU Score')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(order, rotation=45)

plt.tight_layout()
plt.savefig('experiment_results/figure1_prompt_comparison.png', bbox_inches='tight', dpi=600)

# 2. Confidence vs Accuracy scatter plot
plt.figure(figsize=(8, 6))
strategies = df['strategy'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

for strategy, color in zip(strategies, colors):
    subset = df[df['strategy'] == strategy]
    plt.scatter(subset['confidence'], subset['iou'], 
                alpha=0.6, s=50, color=color, label=strategy)

plt.xlabel('SAM Confidence Score', fontsize=12)
plt.ylabel('IoU (Ground Truth)', fontsize=12)
plt.title('SAM Confidence vs. Actual Segmentation Accuracy', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('experiment_results/figure2_confidence_vs_iou.png', bbox_inches='tight', dpi=600)

# 3. Class-wise analysis
plt.figure(figsize=(10, 6))
class_iou = df.groupby(['class', 'strategy'])['iou'].mean().unstack()
class_iou.plot(kind='bar', figsize=(12, 6))
plt.title('Performance Variation Across Object Classes', fontweight='bold')
plt.xlabel('Object Class')
plt.ylabel('Mean IoU')
plt.legend(title='Prompt Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('experiment_results/figure3_class_analysis.png', bbox_inches='tight', dpi=600)

print("✅ Publication figures created!")
print("Check 'experiment_results/' folder for:")
print("1. figure1_prompt_comparison.png - Main results")
print("2. figure2_confidence_vs_iou.png - Confidence analysis")
print("3. figure3_class_analysis.png - Class-wise analysis")