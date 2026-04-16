# scripts/run_all_experiments.py
import subprocess
import sys
from pathlib import Path

def setup_experiments():
    """Setup and run all experiments"""
    
    experiments = [
        {
            'name': 'main_comparison',
            'script': 'enhanced_experiment.py',
            'params': '--samples 200 --strategies all'
        },
        {
            'name': 'ablation_study',
            'script': 'ablation_study.py',
            'params': '--focus points_variation'
        },
        {
            'name': 'robustness_analysis',
            'script': 'robustness_analysis.py',
            'params': '--noise_levels 5'
        }
    ]
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable,
            f"scripts/{exp['script']}",
            *exp['params'].split()
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {exp['name']} completed successfully!")
        else:
            print(f"❌ {exp['name']} failed!")
            print(f"Error: {result.stderr}")
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    setup_experiments()