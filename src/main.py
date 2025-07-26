#!/usr/bin/env python3
"""
Unified CLI for ADAPTIVE_LOSS_PROJECT workflows:
- run: single adaptive or baseline experiment
- sweep: parameter sweep
- ablate: systematic ablation
- report: generate dashboard & markdown
"""

import argparse
import sys
from pathlib import Path
from utils.config_manager import ConfigManager
from evaluation.experiment_runner import ExperimentRunner, ExperimentConfig
from visualization.dashboard import DashboardBuilder
from visualization.report_generator import ReportGenerator

def parse_args():
    p = argparse.ArgumentParser(prog="adaptive_loss")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    run = sub.add_parser("run", help="Run single experiment")
    run.add_argument("--config", required=True, help="Config name from experiments/configs/*.yaml")
    run.add_argument("--adaptive", action="store_true", help="Use adaptive trainer")
    run.add_argument("--baseline", action="store_true", help="Use baseline trainer")

    # sweep
    sweep = sub.add_parser("sweep", help="Parameter sweep")
    sweep.add_argument("--config", required=True, help="Base config name")
    sweep.add_argument("--params", nargs="+", help="param=value pairs", required=True)

    # ablate
    ablate = sub.add_parser("ablate", help="Run ablation study")
    ablate.add_argument("--config", required=True, help="Base config name")

    # report
    rpt = sub.add_parser("report", help="Generate dashboard and markdown report")
    rpt.add_argument("--exp_dir", required=True, help="Experiment results directory")

    return p.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    cfg_mgr = ConfigManager(project_root, config_dir="experiments/configs")
    runner = ExperimentRunner(output_dir=str(project_root/"experiments/results"))

    if args.cmd == "run":
        base_cfg = cfg_mgr.get(args.config)
        exp_cfg = ExperimentConfig(
            name=args.config,
            model_config=base_cfg.get("model", {}),
            training_config=base_cfg.get("training", {}),
            dataset_config=base_cfg.get("dataset", {}),
            adaptive_config=base_cfg.get("adaptive", {}),
            hardware_config=base_cfg.get("hardware", {})
        )
        runner.add_experiment(exp_cfg)
        results = runner.run_experiments()
        print(f"Run complete: {results}")

    elif args.cmd == "sweep":
        base_cfg = cfg_mgr.get(args.config)
        exp_cfg = ExperimentConfig(
            name=args.config,
            model_config=base_cfg.get("model", {}),
            training_config=base_cfg.get("training", {}),
            dataset_config=base_cfg.get("dataset", {}),
            adaptive_config=base_cfg.get("adaptive", {}),
            hardware_config=base_cfg.get("hardware", {})
        )
        # parse params e.g. training.learning_rate=0.01
        param_space = {}
        for pair in args.params:
            k,v = pair.split("=")
            section, key = k.split(".",1)
            param_space[f"{section}.{key}"] = [type(base_cfg[section][key])(v)]
        sweeps = runner.create_parameter_sweep(exp_cfg, param_space)
        for sw in sweeps:
            runner.add_experiment(sw)
        runner.run_experiments()

    elif args.cmd == "ablate":
        base_cfg = cfg_mgr.get(args.config)
        exp_cfg = ExperimentConfig(
            name=args.config,
            model_config=base_cfg.get("model", {}),
            training_config=base_cfg.get("training", {}),
            dataset_config=base_cfg.get("dataset", {}),
            adaptive_config=base_cfg.get("adaptive", {}),
            hardware_config=base_cfg.get("hardware", {})
        )
        runner.run_ablation_study(exp_cfg)

    elif args.cmd == "report":
        exp_dir = Path(args.exp_dir)
        # Generate dashboard
        pngs = [f.name for f in exp_dir.glob("*.png")]
        DashboardBuilder(plot_dir=str(exp_dir), output_path=str(exp_dir/"dashboard.png")).build(pngs)
        # Collect summary JSON
        summary = {}
        if (exp_dir/"summary.json").exists():
            import json
            summary = json.loads((exp_dir/"summary.json").read_text())
        ReportGenerator(results_dir=str(exp_dir), report_path=str(exp_dir/"REPORT.md")).generate(summary, pngs)

    else:
        sys.exit("Unknown command")

if __name__ == "__main__":
    main()
