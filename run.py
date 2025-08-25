import argparse
import sys
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.evaluation.experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Adaptive Loss Project Runner")
    parser.add_argument("--mode", choices=["baseline","adaptive","ablation","sweep"], required=True)
    parser.add_argument("--config", help="Path to YAML/JSON overrides", required=False)
    parser.add_argument("--template", help="Schema name under experiments/configs", default="default")
    parser.add_argument("--output", help="Results directory", default="experiments/results")
    args = parser.parse_args()

    cfg_mgr = ConfigManager(schema_dir="experiments/configs")
    # Load base schema
    base = cfg_mgr.load_schema(f"{args.mode}_configs")
    # Load overrides
    overrides = cfg_mgr.load_config(args.config) if args.config else {}
    # Merge
    cfg = cfg_mgr.merge({"default": base}[ "default" ], overrides)

    runner = ExperimentRunner(output_dir=args.output, max_concurrent_experiments=1)

    # Create ExperimentConfig
    exp_cfg = runner.create_experiment_config(
        name=cfg.get("name", f"{args.mode}_experiment"),
        template=args.mode + "_default",
        overrides=cfg
    )

    # Add and run
    runner.add_experiment(exp_cfg)
    results = runner.run_experiments()

    # Summarize
    print("=== Experiment Status ===")
    for name, result in results.items():
        print(f"{name}: {result.status.value}, metrics={result.metrics}")

if __name__ == "__main__":
    main()
