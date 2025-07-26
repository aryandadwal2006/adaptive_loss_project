import os
import sys
# Ensure project root (folder containing `src/`) is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import csv
import time
import argparse
from pathlib import Path

from src.trainer.train_phase1 import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 experiments")
    parser.add_argument("--seeds", nargs='+', type=int, default=[1, 42, 123],
                        help="List of random seeds to run")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs per run")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (must match cifar10_loader)")
    parser.add_argument("--output", type=str, default="results/phase1_results.csv",
                        help="CSV file to save aggregated metrics")
    return parser.parse_args()

def run_one(adaptive: bool, seed: int, args):
    logdir = f"runs/phase1_seed{seed}_{'adaptive' if adaptive else 'baseline'}"
    os.makedirs(logdir, exist_ok=True)

    cmd_args = [
        f"--seed={seed}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--batch_size={args.batch_size}",
        f"--logdir={logdir}"
    ]
    if adaptive:
        cmd_args.append("--adaptive")

    start = time.time()
    old_argv = sys.argv
    sys.argv = ["train_phase1.py"] + cmd_args
    try:
        train_main()
    finally:
        sys.argv = old_argv
    elapsed = time.time() - start

    metrics_file = Path(logdir) / "best_metrics.txt"
    if not metrics_file.exists():
        raise FileNotFoundError(f"{metrics_file} not found.")
    best_epoch, best_acc = metrics_file.read_text().strip().split(',')

    return int(seed), adaptive, float(best_acc), int(best_epoch), elapsed

def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seed", "adaptive", "best_val_acc", "best_epoch", "time_sec"])
        for adaptive in [False, True]:
            for seed in args.seeds:
                print(f"Running seed={seed} adaptive={adaptive}")
                try:
                    row = run_one(adaptive, seed, args)
                except Exception as e:
                    print(f"Error in run seed={seed} adaptive={adaptive}: {e}")
                    continue
                writer.writerow(row)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
