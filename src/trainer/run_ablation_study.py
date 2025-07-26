# File: src/trainer/run_ablation_study.py

import os
import sys
import csv
import time
import argparse
import json
from pathlib import Path
from itertools import product
import numpy as np

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.trainer.train_ablation import main as train_ablation_main

def json_serialize_fix(obj):
    """Custom JSON serializer to handle numpy types"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return str(obj)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run Complete Ablation Study")
    parser.add_argument("--seeds", nargs='+', type=int, default=[1, 42, 123, 456, 789],
                        help="List of random seeds to run")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs per run")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results/ablation_study",
                        help="Directory to save all ablation results")
    parser.add_argument("--modes", nargs='+', type=str, 
                        default=["baseline", "monitor_only", "controller_only", "full_adaptive"],
                        help="Ablation modes to test")
    return parser.parse_args()

def run_single_ablation(mode: str, seed: int, args):
    """Run a single ablation experiment"""
    logdir = f"{args.output_dir}/{mode}_seed{seed}"
    os.makedirs(logdir, exist_ok=True)

    # Build command-line args for train_ablation
    cmd_args = [
        f"--ablation_mode={mode}",
        f"--seed={seed}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--batch_size={args.batch_size}",
        f"--logdir={logdir}"
    ]

    start = time.time()
    old_argv = sys.argv
    sys.argv = ["train_ablation.py"] + cmd_args
    try:
        train_ablation_main()
    finally:
        sys.argv = old_argv
    elapsed = time.time() - start

    # Read results
    metrics_file = Path(logdir) / "best_metrics.txt"
    detailed_file = Path(logdir) / "detailed_metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"{metrics_file} not found.")
    
    best_epoch, best_acc, runtime = metrics_file.read_text().strip().split(',')
    
    # Load detailed metrics if available
    detailed_metrics = {}
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            detailed_metrics = json.load(f)
    
    return {
        'mode': mode,
        'seed': int(seed),
        'best_val_acc': float(best_acc),
        'best_epoch': int(best_epoch),
        'time_sec': float(runtime),
        'detailed': detailed_metrics
    }

def compute_statistics(results):
    """Compute summary statistics for each ablation mode"""
    from collections import defaultdict
    import numpy as np
    
    mode_results = defaultdict(list)
    
    # Group results by mode
    for result in results:
        mode_results[result['mode']].append(result['best_val_acc'])
    
    # Compute statistics for each mode
    stats = {}
    for mode, accs in mode_results.items():
        accs = np.array(accs)
        stats[mode] = {
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs, ddof=1)),
            'min_acc': float(np.min(accs)),
            'max_acc': float(np.max(accs)),
            'n_seeds': len(accs)
        }
    
    return stats

def perform_statistical_tests(results):
    """Perform statistical tests between ablation modes"""
    from collections import defaultdict
    import numpy as np
    from scipy import stats as scipy_stats
    
    mode_results = defaultdict(list)
    for result in results:
        mode_results[result['mode']].append(result['best_val_acc'])
    
    # Convert to arrays
    mode_arrays = {mode: np.array(accs) for mode, accs in mode_results.items()}
    
    # Pairwise comparisons
    comparisons = []
    modes = list(mode_arrays.keys())
    
    for i in range(len(modes)):
        for j in range(i+1, len(modes)):
            mode1, mode2 = modes[i], modes[j]
            arr1, arr2 = mode_arrays[mode1], mode_arrays[mode2]
            
            # Paired t-test (assuming same seeds were used)
            if len(arr1) == len(arr2):
                t_stat, p_value = scipy_stats.ttest_rel(arr2, arr1)  # arr2 - arr1
                
                # Cohen's d for paired samples
                diff = arr2 - arr1
                cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else float('inf')
                
                comparisons.append({
                    'mode1': mode1,
                    'mode2': mode2,
                    'mean_diff': float(np.mean(diff)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohen_d': float(cohen_d),
                    'significant_05': p_value < 0.05
                })
    
    return comparisons

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting ablation study with {len(args.modes)} modes and {len(args.seeds)} seeds")
    print(f"Total experiments: {len(args.modes) * len(args.seeds)}")
    
    results = []
    
    # Run all combinations of modes and seeds
    total_runs = len(args.modes) * len(args.seeds)
    current_run = 0
    
    for mode, seed in product(args.modes, args.seeds):
        current_run += 1
        print(f"\n[{current_run}/{total_runs}] Running {mode} with seed {seed}")
        
        try:
            result = run_single_ablation(mode, seed, args)
            results.append(result)
            print(f"  ✓ Completed: Best acc = {result['best_val_acc']:.4f} at epoch {result['best_epoch']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\nCompleted {len(results)} out of {total_runs} experiments")
    
    # Save raw results
    results_file = Path(args.output_dir) / "ablation_results.csv"
    with open(results_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=['mode', 'seed', 'best_val_acc', 'best_epoch', 'time_sec'])
            writer.writeheader()
            for result in results:
                writer.writerow({k: v for k, v in result.items() if k != 'detailed'})
    
    # Compute and save statistics
    if results:
        stats = compute_statistics(results)
        stats_file = Path(args.output_dir) / "ablation_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Statistical tests
        try:
            comparisons = perform_statistical_tests(results)
            comparisons_file = Path(args.output_dir) / "statistical_comparisons.json"
            with open(comparisons_file, 'w') as f:
                json.dump(comparisons, f, indent=2, default=json_serialize_fix)
        except ImportError:
            print("Warning: scipy not available for statistical tests")
            comparisons = []
        
        # Print summary
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        
        print("\nMean Validation Accuracy by Mode:")
        for mode in ["baseline", "monitor_only", "controller_only", "full_adaptive"]:
            if mode in stats:
                s = stats[mode]
                print(f"  {mode:15}: {s['mean_acc']:.4f} ± {s['std_acc']:.4f} (n={s['n_seeds']})")
        
        if comparisons:
            print("\nSignificant Improvements (p < 0.05):")
            significant_found = False
            for comp in comparisons:
                if comp['significant_05'] and comp['mean_diff'] > 0:
                    print(f"  {comp['mode2']} > {comp['mode1']}: "
                          f"Δ = {comp['mean_diff']:.4f}, p = {comp['p_value']:.4f}, "
                          f"d = {comp['cohen_d']:.3f}")
                    significant_found = True
            
            if not significant_found:
                print("  No significant improvements found")
        
        print(f"\nResults saved to: {args.output_dir}")
    
    else:
        print("No successful experiments completed!")

if __name__ == "__main__":
    main()
