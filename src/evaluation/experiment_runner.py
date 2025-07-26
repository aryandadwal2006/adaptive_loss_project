"""
Automated Experiment Runner for Adaptive Loss Systems
Orchestrates multiple experiments with different configurations and manages execution
"""

import os
import json
import yaml
import time
import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import subprocess
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
import argparse
import torch

from ..utils.logging_utils import ExperimentLogger
from ..utils.checkpoint_utils import CheckpointManager, CheckpointType
from ..evaluation.statistical_tests import ExperimentalValidation
from ..evaluation.ablation_studies import AblationStudyRunner

class ExperimentStatus(Enum):
    """Status of an experiment"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    adaptive_config: Dict[str, Any]
    hardware_config: Dict[str, Any]
    seed: int = 42
    priority: int = 1
    max_retries: int = 3
    timeout_hours: int = 24
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ExperimentResult:
    """Result of an experiment execution"""
    config: ExperimentConfig
    status: ExperimentStatus
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    duration: Optional[float]
    metrics: Dict[str, float]
    logs: List[str]
    artifacts: List[str]
    error_message: Optional[str] = None
    retry_count: int = 0

class ExperimentQueue:
    """Queue for managing experiment execution"""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.pending_experiments = []
        self.running_experiments = []
        self.completed_experiments = []
        self.failed_experiments = []
        self._lock = threading.Lock()
    
    def add_experiment(self, config: ExperimentConfig) -> str:
        """Add an experiment to the queue"""
        with self._lock:
            self.pending_experiments.append(config)
            # Sort by priority (higher priority first)
            self.pending_experiments.sort(key=lambda x: x.priority, reverse=True)
        return config.name
    
    def get_next_experiment(self) -> Optional[ExperimentConfig]:
        """Get the next experiment to run"""
        with self._lock:
            if (self.pending_experiments and 
                len(self.running_experiments) < self.max_concurrent):
                return self.pending_experiments.pop(0)
        return None
    
    def mark_running(self, config: ExperimentConfig):
        """Mark an experiment as running"""
        with self._lock:
            self.running_experiments.append(config)
    
    def mark_completed(self, config: ExperimentConfig, result: ExperimentResult):
        """Mark an experiment as completed"""
        with self._lock:
            self.running_experiments.remove(config)
            self.completed_experiments.append(result)
    
    def mark_failed(self, config: ExperimentConfig, result: ExperimentResult):
        """Mark an experiment as failed"""
        with self._lock:
            self.running_experiments.remove(config)
            self.failed_experiments.append(result)
    
    def get_status(self) -> Dict[str, int]:
        """Get queue status"""
        with self._lock:
            return {
                'pending': len(self.pending_experiments),
                'running': len(self.running_experiments),
                'completed': len(self.completed_experiments),
                'failed': len(self.failed_experiments)
            }

class ExperimentRunner:
    """
    Automated experiment runner that orchestrates multiple experiments
    with different configurations and manages their execution
    """
    
    def __init__(self, output_dir: str = "./experiment_results", 
                 max_concurrent_experiments: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = ExperimentLogger("experiment_runner", str(self.output_dir))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / "checkpoints")
        )
        self.experiment_queue = ExperimentQueue(max_concurrent_experiments)
        
        # Experiment tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.running = False
        
        # Configuration templates
        self.config_templates = self._load_config_templates()
        
        self.logger.log_info(f"Experiment Runner initialized")
        self.logger.log_info(f"Session ID: {self.session_id}")
        self.logger.log_info(f"Output directory: {self.output_dir}")
    
    def _load_config_templates(self) -> Dict[str, Dict]:
        """Load configuration templates"""
        templates = {}
        
        # Default template
        templates['default'] = {
            'model_config': {
                'type': 'SimpleCNN',
                'input_channels': 1,
                'num_classes': 5,
                'hidden_dim': 64
            },
            'training_config': {
                'num_epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4
            },
            'dataset_config': {
                'task_type': 'classification',
                'num_samples': 1000,
                'image_size': 32,
                'train_split': 0.8
            },
            'adaptive_config': {
                'use_rl_agent': True,
                'use_meta_controller': True,
                'rl_hidden_dim': 64,
                'meta_hidden_dim': 128
            },
            'hardware_config': {
                'device': 'cpu',
                'num_threads': 4
            }
        }
        
        # Load from files if they exist
        config_dir = Path("experiments/configs")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        templates[config_file.stem] = yaml.safe_load(f)
                except Exception as e:
                    self.logger.log_warning(f"Failed to load config {config_file}: {e}")
        
        return templates
    
    def create_experiment_config(self, name: str, 
                                template: str = 'default',
                                overrides: Dict[str, Any] = None) -> ExperimentConfig:
        """Create an experiment configuration"""
        if template not in self.config_templates:
            raise ValueError(f"Template {template} not found")
        
        # Start with template
        config = self.config_templates[template].copy()
        
        # Apply overrides
        if overrides:
            config = self._deep_merge_dicts(config, overrides)
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            name=name,
            model_config=config['model_config'],
            training_config=config['training_config'],
            dataset_config=config['dataset_config'],
            adaptive_config=config['adaptive_config'],
            hardware_config=config['hardware_config']
        )
        
        return experiment_config
    
    def _deep_merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def add_experiment(self, config: ExperimentConfig) -> str:
        """Add an experiment to the execution queue"""
        experiment_id = self.experiment_queue.add_experiment(config)
        self.logger.log_info(f"Added experiment: {config.name}")
        return experiment_id
    
    def create_parameter_sweep(self, base_config: ExperimentConfig,
                              parameter_space: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Create a parameter sweep experiment"""
        experiments = []
        
        # Generate all combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        for i, combination in enumerate(itertools.product(*param_values)):
            # Create config for this combination
            config = ExperimentConfig(
                name=f"{base_config.name}_sweep_{i:03d}",
                model_config=base_config.model_config.copy(),
                training_config=base_config.training_config.copy(),
                dataset_config=base_config.dataset_config.copy(),
                adaptive_config=base_config.adaptive_config.copy(),
                hardware_config=base_config.hardware_config.copy(),
                seed=base_config.seed + i,
                priority=base_config.priority
            )
            
            # Apply parameter values
            for param_name, param_value in zip(param_names, combination):
                self._set_nested_parameter(config, param_name, param_value)
            
            experiments.append(config)
        
        return experiments
    
    def _set_nested_parameter(self, config: ExperimentConfig, param_path: str, value: Any):
        """Set a nested parameter in the configuration"""
        parts = param_path.split('.')
        
        # Navigate to the correct config section
        if parts[0] == 'model':
            target = config.model_config
        elif parts[0] == 'training':
            target = config.training_config
        elif parts[0] == 'dataset':
            target = config.dataset_config
        elif parts[0] == 'adaptive':
            target = config.adaptive_config
        elif parts[0] == 'hardware':
            target = config.hardware_config
        else:
            raise ValueError(f"Unknown config section: {parts[0]}")
        
        # Navigate to the final parameter
        for part in parts[1:-1]:
            target = target[part]
        
        # Set the value
        target[parts[-1]] = value
    
    def run_experiments(self, blocking: bool = True) -> Dict[str, ExperimentResult]:
        """Run all queued experiments"""
        if self.running:
            raise RuntimeError("Experiments are already running")
        
        self.running = True
        self.logger.log_info("Starting experiment execution")
        
        if blocking:
            self._run_experiments_blocking()
        else:
            # Run in background thread
            thread = threading.Thread(target=self._run_experiments_blocking)
            thread.daemon = True
            thread.start()
        
        return self.results
    
    def _run_experiments_blocking(self):
        """Run experiments in blocking mode"""
        while True:
            # Get next experiment
            config = self.experiment_queue.get_next_experiment()
            if config is None:
                # Check if any experiments are still running
                if len(self.experiment_queue.running_experiments) == 0:
                    break
                time.sleep(1)
                continue
            
            # Run experiment
            self._run_single_experiment(config)
        
        self.running = False
        self.logger.log_info("All experiments completed")
        
        # Generate final report
        self._generate_final_report()
    
    def _run_single_experiment(self, config: ExperimentConfig):
        """Run a single experiment"""
        self.logger.log_info(f"Starting experiment: {config.name}")
        
        # Mark as running
        self.experiment_queue.mark_running(config)
        
        # Create experiment result
        result = ExperimentResult(
            config=config,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.datetime.now(),
            end_time=None,
            duration=None,
            metrics={},
            logs=[],
            artifacts=[]
        )
        
        try:
            # Run the actual experiment
            experiment_metrics = self._execute_experiment(config)
            
            # Mark as completed
            result.status = ExperimentStatus.COMPLETED
            result.end_time = datetime.datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.metrics = experiment_metrics
            
            self.experiment_queue.mark_completed(config, result)
            
            self.logger.log_info(f"Experiment completed: {config.name}")
            
        except Exception as e:
            # Mark as failed
            result.status = ExperimentStatus.FAILED
            result.end_time = datetime.datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.error_message = str(e)
            
            self.experiment_queue.mark_failed(config, result)
            
            self.logger.log_error(f"Experiment failed: {config.name}", {'error': str(e)})
        
        # Store result
        self.results[config.name] = result
    
    def _execute_experiment(self, config: ExperimentConfig) -> Dict[str, float]:
        """Execute the actual experiment"""
        # Create experiment directory
        experiment_dir = self.output_dir / config.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Import and run experiment
        from ..dataset import create_dataloaders
        from ..models import SimpleCNN
        from ..main_trainer import AdaptiveLossTrainer
        from ..baseline.baseline_trainer import BaselineTrainer
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            task_type=config.dataset_config['task_type'],
            batch_size=config.training_config['batch_size'],
            train_split=config.dataset_config['train_split']
        )
        
        # Create model
        model = SimpleCNN(
            input_channels=config.model_config['input_channels'],
            num_classes=config.model_config['num_classes']
        )
        
        # Create trainer
        if config.adaptive_config['use_rl_agent']:
            trainer = AdaptiveLossTrainer(
                model=model,
                base_loss_fn=torch.nn.CrossEntropyLoss(),
                aux_loss_fn=torch.nn.MSELoss()
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training_config['learning_rate'],
                weight_decay=config.training_config['weight_decay']
            )
            trainer = BaselineTrainer(
                model=model,
                loss_fn=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                experiment_name=config.name
            )
        
        # Train model
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=config.training_config['num_epochs']
        )
        
        # Get final metrics
        final_metrics = trainer.get_final_metrics()
        
        # Save results
        results_path = experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save model checkpoint
        checkpoint_path = experiment_dir / "final_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        return final_metrics
    
    def run_ablation_study(self, base_config: ExperimentConfig) -> Dict[str, Any]:
        """Run comprehensive ablation study"""
        self.logger.log_info("Starting ablation study")
        
        # Create ablation runner
        ablation_runner = AblationStudyRunner(
            base_config=asdict(base_config),
            output_dir=str(self.output_dir / "ablation_study")
        )
        
        # Create data loaders
        from ..dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            task_type=base_config.dataset_config['task_type'],
            batch_size=base_config.training_config['batch_size']
        )
        
        # Run ablation study
        results = ablation_runner.run_systematic_ablation(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=base_config.training_config['num_epochs'],
            num_runs=3
        )
        
        self.logger.log_info("Ablation study completed")
        return results
    
    def run_statistical_validation(self, experiment_results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """Run statistical validation on experiment results"""
        self.logger.log_info("Starting statistical validation")
        
        validator = ExperimentalValidation()
        
        # Group results by configuration type
        baseline_results = []
        adaptive_results = []
        
        for result in experiment_results.values():
            if result.status == ExperimentStatus.COMPLETED:
                if result.config.adaptive_config['use_rl_agent']:
                    adaptive_results.append(result.metrics['final_val_acc'])
                else:
                    baseline_results.append(result.metrics['final_val_acc'])
        
        # Run validation
        validation_results = {}
        if baseline_results and adaptive_results:
            validation_results = validator.validate_experiment(
                baseline_results=baseline_results,
                adaptive_results=adaptive_results,
                experiment_name="adaptive_vs_baseline"
            )
        
        self.logger.log_info("Statistical validation completed")
        return validation_results
    
    def _generate_final_report(self):
        """Generate final experiment report"""
        report_path = self.output_dir / "experiment_report.md"
        
        # Collect statistics
        queue_status = self.experiment_queue.get_status()
        
        with open(report_path, 'w') as f:
            f.write("# Experiment Runner Report\n\n")
            
            # Summary
            f.write("## Summary\n")
            f.write(f"- Session ID: {self.session_id}\n")
            f.write(f"- Total experiments: {sum(queue_status.values())}\n")
            f.write(f"- Completed: {queue_status['completed']}\n")
            f.write(f"- Failed: {queue_status['failed']}\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n")
            for name, result in self.results.items():
                f.write(f"### {name}\n")
                f.write(f"- Status: {result.status.value}\n")
                f.write(f"- Duration: {result.duration:.2f}s\n")
                if result.metrics:
                    f.write(f"- Final validation accuracy: {result.metrics.get('final_val_acc', 'N/A'):.4f}\n")
                f.write("\n")
        
        self.logger.log_info(f"Final report saved to {report_path}")
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current status of all experiments"""
        return {
            'queue_status': self.experiment_queue.get_status(),
            'session_id': self.session_id,
            'running': self.running,
            'results': {name: result.status.value for name, result in self.results.items()}
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Run automated experiments")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, default="./experiment_results", help="Output directory")
    parser.add_argument("--template", type=str, default="default", help="Configuration template")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(output_dir=args.output)
    
    if args.config:
        # Load configuration from file
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create experiment config
        experiment_config = runner.create_experiment_config(
            name=config_data['name'],
            template=args.template,
            overrides=config_data
        )
    else:
        # Use default configuration
        experiment_config = runner.create_experiment_config(
            name="default_experiment",
            template=args.template
        )
    
    if args.sweep:
        # Run parameter sweep
        parameter_space = {
            'training.learning_rate': [0.001, 0.01, 0.1],
            'training.batch_size': [16, 32, 64],
            'adaptive.rl_hidden_dim': [32, 64, 128]
        }
        
        sweep_experiments = runner.create_parameter_sweep(
            experiment_config, parameter_space
        )
        
        for exp in sweep_experiments:
            runner.add_experiment(exp)
    
    elif args.ablation:
        # Run ablation study
        results = runner.run_ablation_study(experiment_config)
        print(f"Ablation study completed: {results}")
    
    else:
        # Run single experiment
        runner.add_experiment(experiment_config)
    
    # Run experiments
    if not args.ablation:
        results = runner.run_experiments()
        print(f"Experiments completed: {len(results)}")

if __name__ == "__main__":
    main()
