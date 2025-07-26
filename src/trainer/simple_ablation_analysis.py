import csv
from pathlib import Path
from collections import defaultdict
import statistics

def load_ablation_results_simple(results_dir):
    """Load ablation results using only built-in libraries"""
    results_dir = Path(results_dir)
    csv_file = results_dir / "ablation_results.csv"
    
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'mode': row['mode'],
                'seed': int(row['seed']),
                'best_val_acc': float(row['best_val_acc']),
                'best_epoch': int(row['best_epoch']),
                'time_sec': float(row['time_sec'])
            })
    return results

def analyze_results(results_dir):
    """Analyze ablation results"""
    results = load_ablation_results_simple(results_dir)
    
    # Group by mode
    mode_data = defaultdict(list)
    for result in results:
        mode_data[result['mode']].append(result['best_val_acc'])
    
    # Compute statistics
    print("Ablation Study Results:")
    print("=" * 50)
    
    baseline_mean = 0
    if 'baseline' in mode_data:
        baseline_mean = statistics.mean(mode_data['baseline'])
    
    for mode in ['baseline', 'monitor_only', 'controller_only', 'full_adaptive']:
        if mode in mode_data:
            accs = mode_data[mode]
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            improvement = mean_acc - baseline_mean if mode != 'baseline' else 0
            
            print(f"{mode:15}: {mean_acc:.4f} ± {std_acc:.4f} (Δ={improvement:+.4f})")
    
    # Component analysis
    if all(mode in mode_data for mode in ['baseline', 'monitor_only', 'controller_only', 'full_adaptive']):
        baseline_acc = statistics.mean(mode_data['baseline'])
        monitor_contrib = statistics.mean(mode_data['monitor_only']) - baseline_acc
        controller_contrib = statistics.mean(mode_data['controller_only']) - baseline_acc
        full_contrib = statistics.mean(mode_data['full_adaptive']) - baseline_acc
        
        print("\nComponent Analysis:")
        print(f"RL Monitor contribution:     {monitor_contrib:+.4f}")
        print(f"Meta-Controller contribution: {controller_contrib:+.4f}")
        print(f"Full system improvement:     {full_contrib:+.4f}")
        print(f"Synergy effect:              {full_contrib - (monitor_contrib + controller_contrib):+.4f}")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/ablation_study"
    analyze_results(results_dir)
