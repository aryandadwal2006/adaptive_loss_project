"""
Comprehensive Ablation Studies Framework
Systematic component removal and analysis for adaptive loss systems
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import itertools
import json
from pathlib import Path
import time
from collections import defaultdict

from ..main_trainer import AdaptiveLossTrainer
from ..baseline.baseline_trainer import BaselineTrainer
from ..evaluation.metrics import TrainingMetrics, ComparisonMetrics
from ..utils.logging_utils import ExperimentLogger

class AblationComponent(Enum):
    """Components that can be ablated"""
    RL_AGENT = "rl_agent"
    META_CONTROLLER = "meta_controller"
    AUXILIARY_LOSS = "auxiliary_loss"
    GRADIENT_CLIPPING = "gradient_clipping"
    LOSS_SCHEDULING = "loss_scheduling"
    EARLY_STOPPING = "early_stopping"
    BATCH_NORMALIZATION = "batch_normalization"
    DROPOUT = "dropout"

@dataclass
class AblationConfiguration:
    """Configuration for an ablation study"""
    name: str
    removed_components: List[AblationComponent]
    modified_params: Dict[str, Any]
    description: str
    expected_impact: str = ""

class AblationStudyRunner:
    """
    Comprehensive ablation study runner that systematically removes/modifies
    components to understand their contribution to the adaptive loss system
    """
    
    def __init__(self, base_config: Dict[str, Any], 
                 output_dir: str = "./ablation_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.logger = ExperimentLogger("ablation_study", str(self.output_dir))
        self.results = {}
        self.comparison_metrics = ComparisonMetrics()
        
        # Predefined ablation configurations
        self.ablation_configs = self._create_ablation_configs()
        
        print(f"Ablation Study Runner initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Base configuration: {base_config}")
    
    def _create_ablation_configs(self) -> List[AblationConfiguration]:
        """Create predefined ablation configurations"""
        configs = []
        
        # 1. Remove RL Agent
        configs.append(AblationConfiguration(
            name="no_rl_agent",
            removed_components=[AblationComponent.RL_AGENT],
            modified_params={'use_static_loss': True},
            description="Remove RL-based loss adaptation, use static loss",
            expected_impact="Reduced adaptability, potentially worse convergence"
        ))
        
        # 2. Remove Meta-Controller
        configs.append(AblationConfiguration(
            name="no_meta_controller",
            removed_components=[AblationComponent.META_CONTROLLER],
            modified_params={'use_meta_learning': False},
            description="Remove meta-learning component",
            expected_impact="Reduced generalization across tasks"
        ))
        
        # 3. Remove Auxiliary Loss
        configs.append(AblationConfiguration(
            name="no_auxiliary_loss",
            removed_components=[AblationComponent.AUXILIARY_LOSS],
            modified_params={'auxiliary_loss_weight': 0.0},
            description="Remove auxiliary loss component",
            expected_impact="Potentially reduced regularization"
        ))
        
        # 4. Remove Both RL and Meta-Learning
        configs.append(AblationConfiguration(
            name="no_adaptive_components",
            removed_components=[AblationComponent.RL_AGENT, AblationComponent.META_CONTROLLER],
            modified_params={'use_static_loss': True, 'use_meta_learning': False},
            description="Remove all adaptive components",
            expected_impact="Equivalent to baseline training"
        ))
        
        # 5. Remove Gradient Clipping
        configs.append(AblationConfiguration(
            name="no_gradient_clipping",
            removed_components=[AblationComponent.GRADIENT_CLIPPING],
            modified_params={'max_grad_norm': None},
            description="Remove gradient clipping",
            expected_impact="Potential training instability"
        ))
        
        # 6. Simplified RL Agent
        configs.append(AblationConfiguration(
            name="simplified_rl",
            removed_components=[],
            modified_params={'rl_hidden_dim': 32, 'rl_memory_size': 1000},
            description="Use simplified RL agent with reduced capacity",
            expected_impact="Reduced learning capacity but faster training"
        ))
        
        # 7. No Exploration (Greedy RL)
        configs.append(AblationConfiguration(
            name="greedy_rl",
            removed_components=[],
            modified_params={'epsilon': 0.0, 'epsilon_decay': 1.0},
            description="Remove exploration from RL agent",
            expected_impact="Reduced exploration, potential suboptimal policies"
        ))
        
        # 8. Different Loss Scheduling
        configs.append(AblationConfiguration(
            name="linear_loss_decay",
            removed_components=[AblationComponent.LOSS_SCHEDULING],
            modified_params={'loss_decay_type': 'linear', 'loss_decay_rate': 0.95},
            description="Use linear loss decay instead of adaptive scheduling",
            expected_impact="Less flexible loss adaptation"
        ))
        
        return configs
    
    def run_systematic_ablation(self, train_loader, val_loader, 
                               num_epochs: int = 50, 
                               num_runs: int = 3) -> Dict[str, Any]:
        """
        Run systematic ablation study across all configurations
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            num_runs: Number of runs per configuration
            
        Returns:
            Comprehensive ablation results
        """
        self.logger.log_info("Starting systematic ablation study")
        self.logger.log_info(f"Configurations to test: {len(self.ablation_configs)}")
        self.logger.log_info(f"Runs per configuration: {num_runs}")
        
        all_results = {}
        
        # Run baseline first
        baseline_results = self._run_baseline_experiments(
            train_loader, val_loader, num_epochs, num_runs
        )
        all_results['baseline'] = baseline_results
        
        # Run each ablation configuration
        for config in self.ablation_configs:
            self.logger.log_info(f"Running ablation: {config.name}")
            self.logger.log_info(f"Description: {config.description}")
            
            config_results = self._run_ablation_configuration(
                config, train_loader, val_loader, num_epochs, num_runs
            )
            all_results[config.name] = config_results
        
        # Generate comprehensive analysis
        analysis_results = self._analyze_ablation_results(all_results)
        
        # Save results
        self._save_ablation_results(all_results, analysis_results)
        
        self.logger.log_info("Systematic ablation study completed")
        
        return {
            'experiment_results': all_results,
            'analysis': analysis_results
        }
    
    def _run_baseline_experiments(self, train_loader, val_loader, 
                                 num_epochs: int, num_runs: int) -> Dict[str, Any]:
        """Run baseline experiments for comparison"""
        self.logger.log_info("Running baseline experiments")
        
        baseline_results = {
            'configurations': [],
            'metrics': [],
            'raw_results': []
        }
        
        for run in range(num_runs):
            self.logger.log_info(f"Baseline run {run + 1}/{num_runs}")
            
            # Create baseline trainer
            from ..models import SimpleCNN
            model = SimpleCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = torch.nn.CrossEntropyLoss()
            
            trainer = BaselineTrainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                experiment_name=f"baseline_run_{run}"
            )
            
            # Train
            start_time = time.time()
            history = trainer.train(train_loader, val_loader, num_epochs)
            training_time = time.time() - start_time
            
            # Collect metrics
            final_metrics = trainer.get_final_metrics()
            final_metrics['training_time'] = training_time
            
            baseline_results['raw_results'].append(history)
            baseline_results['metrics'].append(final_metrics)
            
            # Add to comparison metrics
            self.comparison_metrics.add_result(f"baseline_run_{run}", final_metrics)
        
        # Aggregate baseline results
        baseline_results['aggregated_metrics'] = self._aggregate_run_results(
            baseline_results['metrics']
        )
        
        return baseline_results
    
    def _run_ablation_configuration(self, config: AblationConfiguration,
                                   train_loader, val_loader, 
                                   num_epochs: int, num_runs: int) -> Dict[str, Any]:
        """Run experiments for a specific ablation configuration"""
        config_results = {
            'configuration': config,
            'metrics': [],
            'raw_results': []
        }
        
        for run in range(num_runs):
            self.logger.log_info(f"Ablation {config.name} run {run + 1}/{num_runs}")
            
            # Create modified trainer
            trainer = self._create_ablated_trainer(config, f"{config.name}_run_{run}")
            
            # Train
            start_time = time.time()
            history = trainer.train(train_loader, val_loader, num_epochs)
            training_time = time.time() - start_time
            
            # Collect metrics
            final_metrics = trainer.get_final_metrics()
            final_metrics['training_time'] = training_time
            
            config_results['raw_results'].append(history)
            config_results['metrics'].append(final_metrics)
            
            # Add to comparison metrics
            self.comparison_metrics.add_result(f"{config.name}_run_{run}", final_metrics)
        
        # Aggregate results
        config_results['aggregated_metrics'] = self._aggregate_run_results(
            config_results['metrics']
        )
        
        return config_results
    
    def _create_ablated_trainer(self, config: AblationConfiguration, 
                               experiment_name: str):
        """Create a trainer with specified ablations"""
        from ..models import SimpleCNN
        from ..main_trainer import AdaptiveLossTrainer
        
        # Create base components
        model = SimpleCNN()
        base_loss = torch.nn.CrossEntropyLoss()
        aux_loss = torch.nn.MSELoss()
        
        # Apply ablations
        if AblationComponent.AUXILIARY_LOSS in config.removed_components:
            aux_loss = None
        
        # Create trainer
        if AblationComponent.RL_AGENT in config.removed_components:
            # Use baseline trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = BaselineTrainer(
                model=model,
                loss_fn=base_loss,
                optimizer=optimizer,
                experiment_name=experiment_name
            )
        else:
            # Use adaptive trainer with modifications
            trainer = AdaptiveLossTrainer(
                model=model,
                base_loss_fn=base_loss,
                aux_loss_fn=aux_loss
            )
            
            # Apply parameter modifications
            self._apply_parameter_modifications(trainer, config.modified_params)
        
        return trainer
    
    def _apply_parameter_modifications(self, trainer, modified_params: Dict[str, Any]):
        """Apply parameter modifications to trainer"""
        for param_name, param_value in modified_params.items():
            if param_name == 'rl_hidden_dim':
                trainer.loss_monitor.hidden_dim = param_value
            elif param_name == 'rl_memory_size':
                trainer.loss_monitor.memory = trainer.loss_monitor.memory.__class__(
                    maxlen=param_value
                )
            elif param_name == 'epsilon':
                trainer.loss_monitor.epsilon = param_value
            elif param_name == 'epsilon_decay':
                trainer.loss_monitor.epsilon_decay = param_value
            elif param_name == 'auxiliary_loss_weight':
                # This would be handled in the loss computation
                pass
            elif param_name == 'max_grad_norm':
                # This would be handled in the training loop
                pass
    
    def _aggregate_run_results(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate results from multiple runs"""
        if not metrics_list:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        aggregated = {}
        for metric_name in all_metrics:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            
            if values:
                aggregated[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }
        
        return aggregated
    
    def _analyze_ablation_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ablation results and generate insights"""
        analysis = {
            'component_importance': {},
            'performance_ranking': {},
            'statistical_significance': {},
            'insights': []
        }
        
        # Get baseline performance
        baseline_metrics = all_results['baseline']['aggregated_metrics']
        
        # Analyze each ablation
        for config_name, config_results in all_results.items():
            if config_name == 'baseline':
                continue
            
            config_metrics = config_results['aggregated_metrics']
            
            # Calculate impact
            impact = self._calculate_ablation_impact(baseline_metrics, config_metrics)
            analysis['component_importance'][config_name] = impact
        
        # Performance ranking
        ranking_metrics = ['final_val_loss', 'final_val_acc', 'convergence_speed']
        for metric in ranking_metrics:
            if metric in baseline_metrics:
                ranking = self._rank_configurations(all_results, metric)
                analysis['performance_ranking'][metric] = ranking
        
        # Generate insights
        insights = self._generate_insights(all_results, analysis)
        analysis['insights'] = insights
        
        return analysis
    
    def _calculate_ablation_impact(self, baseline_metrics: Dict, 
                                  ablation_metrics: Dict) -> Dict[str, float]:
        """Calculate the impact of an ablation"""
        impact = {}
        
        for metric_name in baseline_metrics:
            if metric_name in ablation_metrics:
                baseline_val = baseline_metrics[metric_name]['mean']
                ablation_val = ablation_metrics[metric_name]['mean']
                
                if baseline_val != 0:
                    # Calculate relative change
                    if 'loss' in metric_name:
                        # Lower is better for loss
                        impact[metric_name] = (baseline_val - ablation_val) / baseline_val
                    else:
                        # Higher is better for accuracy, etc.
                        impact[metric_name] = (ablation_val - baseline_val) / baseline_val
                else:
                    impact[metric_name] = 0.0
        
        return impact
    
    def _rank_configurations(self, all_results: Dict, metric_name: str) -> List[Tuple[str, float]]:
        """Rank configurations by a specific metric"""
        rankings = []
        
        for config_name, config_results in all_results.items():
            if 'aggregated_metrics' in config_results:
                metrics = config_results['aggregated_metrics']
                if metric_name in metrics:
                    value = metrics[metric_name]['mean']
                    rankings.append((config_name, value))
        
        # Sort based on metric type
        reverse = 'loss' not in metric_name  # Lower is better for loss
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        return rankings
    
    def _generate_insights(self, all_results: Dict, analysis: Dict) -> List[str]:
        """Generate insights from ablation results"""
        insights = []
        
        # Component importance insights
        importance = analysis['component_importance']
        
        # Find most impactful ablations
        most_impactful = []
        for config_name, impacts in importance.items():
            if 'final_val_acc' in impacts:
                most_impactful.append((config_name, impacts['final_val_acc']))
        
        most_impactful.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if most_impactful:
            top_impact = most_impactful[0]
            insights.append(
                f"Most impactful ablation: {top_impact[0]} "
                f"({top_impact[1]:.1%} change in validation accuracy)"
            )
        
        # Performance ranking insights
        if 'final_val_acc' in analysis['performance_ranking']:
            ranking = analysis['performance_ranking']['final_val_acc']
            best_config = ranking[0][0]
            insights.append(f"Best performing configuration: {best_config}")
        
        # Component necessity insights
        critical_components = []
        for config_name, impacts in importance.items():
            if 'final_val_acc' in impacts and impacts['final_val_acc'] < -0.05:
                critical_components.append(config_name)
        
        if critical_components:
            insights.append(f"Critical components (>5% performance drop): {critical_components}")
        
        # Training efficiency insights
        efficiency_ranking = []
        for config_name, config_results in all_results.items():
            if 'aggregated_metrics' in config_results:
                metrics = config_results['aggregated_metrics']
                if 'training_time' in metrics and 'final_val_acc' in metrics:
                    time_val = metrics['training_time']['mean']
                    acc_val = metrics['final_val_acc']['mean']
                    efficiency = acc_val / time_val  # Accuracy per second
                    efficiency_ranking.append((config_name, efficiency))
        
        if efficiency_ranking:
            efficiency_ranking.sort(key=lambda x: x[1], reverse=True)
            most_efficient = efficiency_ranking[0][0]
            insights.append(f"Most efficient configuration: {most_efficient}")
        
        return insights
    
    def _save_ablation_results(self, all_results: Dict, analysis: Dict):
        """Save ablation results to files"""
        # Save raw results
        results_file = self.output_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(all_results)
            json.dump(json_results, f, indent=2)
        
        # Save analysis
        analysis_file = self.output_dir / "ablation_analysis.json"
        with open(analysis_file, 'w') as f:
            json_analysis = self._convert_for_json(analysis)
            json.dump(json_analysis, f, indent=2)
        
        # Save summary report
        self._save_summary_report(all_results, analysis)
        
        self.logger.log_info(f"Ablation results saved to {self.output_dir}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._convert_for_json(obj.__dict__)
        else:
            return obj
    
    def _save_summary_report(self, all_results: Dict, analysis: Dict):
        """Save a human-readable summary report"""
        report_file = self.output_dir / "ablation_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# Ablation Study Summary Report\n\n")
            
            # Overview
            f.write("## Overview\n")
            f.write(f"Total configurations tested: {len(all_results)}\n")
            f.write(f"Baseline: {all_results['baseline']['aggregated_metrics']['final_val_acc']['mean']:.3f} validation accuracy\n\n")
            
            # Component importance
            f.write("## Component Importance\n")
            for config_name, impacts in analysis['component_importance'].items():
                f.write(f"### {config_name}\n")
                if 'final_val_acc' in impacts:
                    f.write(f"- Validation accuracy impact: {impacts['final_val_acc']:.1%}\n")
                if 'final_val_loss' in impacts:
                    f.write(f"- Validation loss impact: {impacts['final_val_loss']:.1%}\n")
                f.write("\n")
            
            # Performance ranking
            f.write("## Performance Ranking\n")
            if 'final_val_acc' in analysis['performance_ranking']:
                f.write("### By Validation Accuracy\n")
                for rank, (config, score) in enumerate(analysis['performance_ranking']['final_val_acc'], 1):
                    f.write(f"{rank}. {config}: {score:.3f}\n")
            f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n")
            for insight in analysis['insights']:
                f.write(f"- {insight}\n")
            f.write("\n")
        
        self.logger.log_info(f"Summary report saved to {report_file}")

# Component-specific ablation functions
def run_component_ablation(component: AblationComponent, 
                          train_loader, val_loader, 
                          base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run ablation for a specific component"""
    runner = AblationStudyRunner(base_config)
    
    # Create specific configuration
    config = AblationConfiguration(
        name=f"ablate_{component.value}",
        removed_components=[component],
        modified_params={},
        description=f"Ablation study for {component.value}"
    )
    
    # Run experiment
    results = runner._run_ablation_configuration(
        config, train_loader, val_loader, num_epochs=30, num_runs=3
    )
    
    return results

if __name__ == "__main__":
    # Test ablation studies
    print("Testing Ablation Studies Framework...")
    
    # Create mock data loaders
    from ..dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(task_type='classification')
    
    # Base configuration
    base_config = {
        'model_type': 'SimpleCNN',
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    
    # Create ablation runner
    runner = AblationStudyRunner(base_config)
    
    # Run a single ablation
    single_result = run_component_ablation(
        AblationComponent.AUXILIARY_LOSS, 
        train_loader, val_loader, 
        base_config
    )
    
    print(f"Single ablation result: {single_result['aggregated_metrics']}")
    
    # Run systematic ablation (commented out for testing)
    # full_results = runner.run_systematic_ablation(
    #     train_loader, val_loader, num_epochs=10, num_runs=2
    # )
    
    print("Ablation Studies Framework test completed successfully!")
