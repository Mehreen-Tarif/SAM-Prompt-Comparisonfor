# scripts/generate_paper_tables.py
import pandas as pd
import numpy as np

def create_publication_tables(results_path):
    """Generate publication-ready tables"""
    
    df = pd.read_csv(results_path)
    
    # Table 1: Overall performance comparison
    table1 = df.groupby('strategy').agg({
        'iou': ['mean', 'std', 'median'],
        'dice_coefficient': ['mean', 'std'],
        'inference_time': ['mean'],
        'confidence': ['mean']
    }).round(4)
    
    # Format for LaTeX
    table1_latex = table1.to_latex(
        caption='Overall performance comparison of prompt strategies',
        label='tab:overall_results',
        column_format='l' + 'c' * (len(table1.columns) // len(table1.index))
    )
    
    # Table 2: Statistical significance
    strategies = df['strategy'].unique()
    significance_matrix = pd.DataFrame(
        index=strategies,
        columns=strategies,
        dtype=str
    )
    
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            data1 = df[df['strategy'] == s1]['iou']
            data2 = df[df['strategy'] == s2]['iou']
            
            from scipy import stats
            _, p_value = stats.ttest_rel(data1, data2)
            
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            significance_matrix.loc[s1, s2] = sig
            significance_matrix.loc[s2, s1] = f"p={p_value:.4f}"
    
    # Save tables
    output_dir = Path("results/tables")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "table1_overall.tex", "w") as f:
        f.write(table1_latex)
    
    significance_matrix.to_latex(
        output_dir / "table2_significance.tex",
        caption='Statistical significance of pairwise comparisons',
        label='tab:pairwise_significance'
    )
    
    print("✅ Publication tables generated!")
    print(f"📊 Saved to: {output_dir}")

if __name__ == "__main__":
    create_publication_tables("results/comprehensive/all_results.csv")