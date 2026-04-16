# generate_paper_materials.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('experiment_results/sam_experiment_results.csv')

print("📊 GENERATING PAPER MATERIALS")
print("=" * 50)

# 1. Calculate 95% confidence intervals
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    return ci

# Calculate for each strategy
strategies = sorted(df['strategy'].unique())
ci_data = {}

for strategy in strategies:
    subset = df[df['strategy'] == strategy]['iou']
    ci = calculate_ci(subset)
    ci_data[strategy] = ci

# 2. Generate complete results table
print("\n1. COMPLETE RESULTS TABLE:")
print("-" * 40)

summary = df.groupby('strategy').agg({
    'iou': ['mean', 'std', 'count'],
    'confidence': ['mean', 'std'],
    'inference_time': ['mean']
})

for strategy in strategies:
    mean_iou = summary.loc[strategy, ('iou', 'mean')]
    std_iou = summary.loc[strategy, ('iou', 'std')]
    ci_lower, ci_upper = ci_data[strategy]
    time_ms = summary.loc[strategy, ('inference_time', 'mean')] * 1000
    
    print(f"{strategy:20s} | IoU: {mean_iou:.4f} ± {std_iou:.4f} | "
          f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] | Time: {time_ms:.1f}ms")

# 3. Generate statistical analysis table
print("\n2. STATISTICAL ANALYSIS TABLE:")
print("-" * 40)

# Create significance matrix
print(f"{'Strategy':20s}", end="")
for s in strategies:
    print(f" | {s:10s}", end="")
print()

for i, s1 in enumerate(strategies):
    print(f"{s1:20s}", end="")
    for j, s2 in enumerate(strategies):
        if i == j:
            print(f" | {'-':10s}", end="")
        elif i < j:
            data1 = df[df['strategy'] == s1]['iou']
            data2 = df[df['strategy'] == s2]['iou']
            t_stat, p_val = stats.ttest_rel(data1, data2)
            
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            print(f" | {p_val:.6f}{sig}", end="")
        else:
            print(f" | {'':10s}", end="")
    print()

# 4. Create publication-ready figure
print("\n3. CREATING PUBLICATION FIGURES...")

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# A: Main comparison
order = summary[('iou', 'mean')].sort_values(ascending=False).index
sns.boxplot(x='strategy', y='iou', data=df, ax=axes[0,0], order=order)
axes[0,0].set_title('(A) IoU Distribution by Prompt Strategy', fontweight='bold')
axes[0,0].set_xlabel('Prompt Strategy')
axes[0,0].set_ylabel('IoU Score')
axes[0,0].tick_params(axis='x', rotation=45)

# B: Bar plot with confidence intervals
means = [summary.loc[s, ('iou', 'mean')] for s in order]
ci_lowers = [ci_data[s][0] for s in order]
ci_uppers = [ci_data[s][1] for s in order]
errors = [(means[i] - ci_lowers[i], ci_uppers[i] - means[i]) for i in range(len(means))]
errors = np.array(errors).T

x_pos = np.arange(len(means))
axes[0,1].bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
axes[0,1].set_title('(B) Mean IoU with 95% Confidence Intervals', fontweight='bold')
axes[0,1].set_xlabel('Prompt Strategy')
axes[0,1].set_ylabel('IoU Score')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels(order, rotation=45)

# C: Time vs Accuracy trade-off
times = [summary.loc[s, ('inference_time', 'mean')] * 1000 for s in order]
scatter = axes[1,0].scatter(times, means, s=100, alpha=0.6)
for i, txt in enumerate(order):
    axes[1,0].annotate(txt.split('_')[-1], (times[i], means[i]), 
                       xytext=(5,5), textcoords='offset points')
axes[1,0].set_title('(C) Accuracy vs. Inference Time Trade-off', fontweight='bold')
axes[1,0].set_xlabel('Inference Time (ms)')
axes[1,0].set_ylabel('Mean IoU')
axes[1,0].grid(True, alpha=0.3)

# D: Confidence vs Actual Accuracy
scatter = axes[1,1].scatter(df['confidence'], df['iou'], 
                            c=pd.Categorical(df['strategy']).codes, 
                            cmap='tab10', alpha=0.6, s=30)
axes[1,1].set_title('(D) SAM Confidence vs. Ground Truth Accuracy', fontweight='bold')
axes[1,1].set_xlabel('SAM Confidence Score')
axes[1,1].set_ylabel('IoU (Ground Truth)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_results/figure1_comprehensive_analysis.png', 
            dpi=600, bbox_inches='tight')

# 5. Generate LaTeX code automatically
print("\n4. GENERATING LaTeX CODE...")

latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of SAM prompt strategies for remote sensing segmentation}
\\label{tab:results}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Strategy} & \\textbf{Mean IoU} & \\textbf{Std Dev} & \\textbf{95\\% CI} & \\textbf{Time (ms)} \\\\
\\midrule
"""

for strategy in order:
    mean_iou = summary.loc[strategy, ('iou', 'mean')]
    std_iou = summary.loc[strategy, ('iou', 'std')]
    ci_lower, ci_upper = ci_data[strategy]
    time_ms = summary.loc[strategy, ('inference_time', 'mean')] * 1000
    
    if strategy == 'bounding_box':
        latex_table += f"\\textbf{{{strategy}}} & \\textbf{{{mean_iou:.4f}}} & {std_iou:.4f} & [{ci_lower:.4f}, {ci_upper:.4f}] & {time_ms:.1f} \\\\\n"
    else:
        latex_table += f"{strategy} & {mean_iou:.4f} & {std_iou:.4f} & [{ci_lower:.4f}, {ci_upper:.4f}] & {time_ms:.1f} \\\\\n"

latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""

# Save LaTeX table
with open('experiment_results/table_results.tex', 'w') as f:
    f.write(latex_table)

print("\n✅ COMPLETE PAPER MATERIALS GENERATED!")
print("=" * 50)
print("\n📁 Files created in 'experiment_results/' folder:")
print("1. figure1_comprehensive_analysis.png - 4-panel figure")
print("2. table_results.tex - LaTeX table ready for paper")
print("\n📝 For your SCI paper:")
print("- Use Figure 1 for visual results")
print("- Use the LaTeX table in your methodology/results")
print("- Cite the statistical significance (*** p < 0.001)")
print("- Mention RTX 3090 24GB hardware for reproducibility")