# File: src/trainer/ablation_analysis.py

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is on PYTHONPATH  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def load_ablation_results(results_dir: str):
    """Load ablation study results from directory"""
    results_dir = Path(results_dir)
    
    # Load main results
    csv_file = results_dir / "ablation_results.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Results file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Load statistics if available
    stats_file = results_dir / "ablation_statistics.json"
    stats = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    
    # Load comparisons if available
    comp_file = results_dir / "statistical_comparisons.json"
    comparisons = []
    if comp_file.exists():
        with open(comp_file, 'r') as f:
            comparisons = json.load(f)
    
    return df, stats, comparisons

def analyze_component_contributions(df):
    """Analyze individual component contributions"""
    # Get mean accuracy for each mode
    mode_means = df.groupby('mode')['best_val_acc'].agg(['mean', 'std', 'count']).round(4)
    
    # Calculate improvements relative to baseline
    baseline_mean = mode_means.loc['baseline', 'mean'] if 'baseline' in mode_means.index else 0
    
    contributions = {}
    for mode in mode_means.index:
        if mode != 'baseline':
            improvement = mode_means.loc[mode, 'mean'] - baseline_mean
            contributions[mode] = {
                'absolute_acc': mode_means.loc[mode, 'mean'],
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_mean * 100) if baseline_mean > 0 else 0
            }
    
    return mode_means, contributions

def create_ablation_report(results_dir: str, output_file: str = None):
    """Generate comprehensive ablation analysis report"""
    
    # Load data
    df, stats, comparisons = load_ablation_results(results_dir)
    
    # Analyze contributions
    mode_means, contributions = analyze_component_contributions(df)
    
    # Generate report
    report = []
    report.append("# Ablation Study Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append("| Mode | Mean Acc | Std Dev | Count | Improvement |")
    report.append("|------|----------|---------|--------|-------------|")
    
    baseline_acc = mode_means.loc['baseline', 'mean'] if 'baseline' in mode_means.index else 0
    
    for mode in ['baseline', 'monitor_only', 'controller_only', 'full_adaptive']:
        if mode in mode_means.index:
            row = mode_means.loc[mode]
            improvement = row['mean'] - baseline_acc if mode != 'baseline' else 0
            report.append(f"| {mode} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} | {improvement:+.4f} |")
    
    report.append("")
    
    # Component analysis
    report.append("## Component Contribution Analysis")
    report.append("")
    
    if 'monitor_only' in contributions and 'controller_only' in contributions and 'full_adaptive' in contributions:
        monitor_contrib = contributions['monitor_only']['improvement']
        controller_contrib = contributions['controller_only']['improvement']
        full_contrib = contributions['full_adaptive']['improvement']
        
        # Check for synergy/interference
        expected_combined = monitor_contrib + controller_contrib
        actual_combined = full_contrib
        synergy = actual_combined - expected_combined
        
        report.append(f"- **RL Monitor contribution**: {monitor_contrib:+.4f} ({contributions['monitor_only']['improvement_pct']:+.2f}%)")
        report.append(f"- **Meta-Controller contribution**: {controller_contrib:+.4f} ({contributions['controller_only']['improvement_pct']:+.2f}%)")
        report.append(f"- **Combined system**: {full_contrib:+.4f} ({contributions['full_adaptive']['improvement_pct']:+.2f}%)")
        report.append(f"- **Synergy effect**: {synergy:+.4f} ({'positive' if synergy > 0 else 'negative' if synergy < 0 else 'neutral'})")
        report.append("")
        
        # Interpret synergy
        if abs(synergy) < 0.001:
            interpretation = "The components work independently (additive effect)."
        elif synergy > 0.005:
            interpretation = "Strong positive synergy - components enhance each other significantly."
        elif synergy > 0:
            interpretation = "Mild positive synergy - components work well together."
        elif synergy > -0.005:
            interpretation = "Mild interference - components slightly conflict."
        else:
            interpretation = "Strong interference - components significantly conflict."
        
        report.append(f"**Interpretation**: {interpretation}")
        report.append("")
    
    # Statistical significance
    if comparisons:
        report.append("## Statistical Significance Tests")
        report.append("")
        report.append("| Comparison | Mean Diff | p-value | Cohen's d | Significant |")
        report.append("|------------|-----------|---------|-----------|-------------|")
        
        for comp in comparisons:
            sig_mark = "✓" if comp['significant_05'] else "✗"
            report.append(f"| {comp['mode2']} vs {comp['mode1']} | {comp['mean_diff']:+.4f} | {comp['p_value']:.4f} | {comp['cohen_d']:.3f} | {sig_mark} |")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if contributions:
        best_single = max(contributions.items(), key=lambda x: x[1]['improvement'])
        
        if 'full_adaptive' in contributions:
            full_improvement = contributions['full_adaptive']['improvement']
            best_single_improvement = best_single[1]['improvement']
            
            if full_improvement > best_single_improvement + 0.005:
                report.append("✓ **Use full adaptive system** - Both components contribute positively with synergy.")
            elif full_improvement > best_single_improvement:
                report.append("✓ **Use full adaptive system** - Marginal benefit from combining components.")
            else:
                report.append(f"⚠ **Consider {best_single[0]} only** - Full system shows no clear benefit over best single component.")
        else:
            report.append(f"✓ **Use {best_single[0]}** - Best performing single component.")
    
    # Join report
    report_text = "\n".join(report)
    
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text, mode_means, contributions, comparisons

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--results_dir", type=str, default="results/ablation_study",
                        help="Directory containing ablation results")
    parser.add_argument("--output", type=str, default="results/ablation_study/analysis_report.md",
                        help="Output file for analysis report")
    args = parser.parse_args()
    
    try:
        report_text, _, _, _ = create_ablation_report(args.results_dir, args.output)
        print("Ablation Analysis Complete!")
        print("\n" + report_text)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the ablation study first with run_ablation_study.py")

if __name__ == "__main__":
    main()
