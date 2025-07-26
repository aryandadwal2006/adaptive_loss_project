"""
Comprehensive Evaluation Metrics for Adaptive Loss Systems
Advanced metrics for measuring training dynamics and performance
"""

import torch
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from collections import defaultdict
import warnings

class TrainingMetrics:
    """Comprehensive training metrics calculator"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
               adaptive_metrics: Optional[Dict] = None):
        """Update metrics for current epoch"""
        self.metrics_history['epoch'].append(epoch)
        
        # Basic metrics
        for key, value in train_metrics.items():
            self.metrics_history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.metrics_history[f'val_{key}'].append(value)
        
        # Adaptive-specific metrics
        if adaptive_metrics:
            for key, value in adaptive_metrics.items():
                self.metrics_history[f'adaptive_{key}'].append(value)
    
    def calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence-related metrics"""
        val_losses = self.metrics_history['val_loss']
        
        if len(val_losses) < 10:
            return {
                'convergence_epoch': len(val_losses),
                'convergence_speed': 1.0,
                'stability_score': 0.0,
                'plateau_detection': False
            }
        
        # Find convergence point (95% of best performance)
        best_loss = min(val_losses)
        convergence_threshold = best_loss * 1.05
        
        convergence_epoch = len(val_losses)
        for i, loss in enumerate(val_losses):
            if loss <= convergence_threshold:
                convergence_epoch = i
                break
        
        # Convergence speed (normalized)
        convergence_speed = convergence_epoch / len(val_losses)
        
        # Stability score (inverse of variance in last 20% of training)
        last_20_percent = int(len(val_losses) * 0.8)
        recent_losses = val_losses[last_20_percent:]
        stability_score = 1.0 / (1.0 + np.var(recent_losses))
        
        # Plateau detection
        plateau_detection = self._detect_plateau(val_losses)
        
        return {
            'convergence_epoch': convergence_epoch,
            'convergence_speed': convergence_speed,
            'stability_score': stability_score,
            'plateau_detection': plateau_detection,
            'best_loss': best_loss,
            'final_loss': val_losses[-1]
        }
    
    def _detect_plateau(self, losses: List[float], window_size: int = 10, 
                       threshold: float = 0.01) -> bool:
        """Detect if training has plateaued"""
        if len(losses) < window_size * 2:
            return False
        
        recent_window = losses[-window_size:]
        earlier_window = losses[-window_size*2:-window_size]
        
        recent_mean = np.mean(recent_window)
        earlier_mean = np.mean(earlier_window)
        
        return abs(recent_mean - earlier_mean) < threshold
    
    def calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate training efficiency metrics"""
        epoch_times = self.metrics_history.get('train_epoch_time', [])
        train_losses = self.metrics_history.get('train_loss', [])
        val_accuracies = self.metrics_history.get('val_accuracy', [])
        
        if not epoch_times or not train_losses:
            return {}
        
        # Time efficiency
        avg_epoch_time = np.mean(epoch_times)
        total_training_time = sum(epoch_times)
        
        # Sample efficiency (how quickly we reach good performance)
        if val_accuracies:
            target_accuracy = max(val_accuracies) * 0.9
            sample_efficiency = len(val_accuracies)
            
            for i, acc in enumerate(val_accuracies):
                if acc >= target_accuracy:
                    sample_efficiency = i + 1
                    break
        else:
            sample_efficiency = len(train_losses)
        
        # Parameter efficiency (performance per parameter)
        # This would need model parameter count
        
        return {
            'avg_epoch_time': avg_epoch_time,
            'total_training_time': total_training_time,
            'sample_efficiency': sample_efficiency,
            'time_to_target': avg_epoch_time * sample_efficiency
        }
    
    def calculate_robustness_metrics(self) -> Dict[str, float]:
        """Calculate robustness metrics"""
        gradient_norms = self.metrics_history.get('train_gradient_norm', [])
        val_losses = self.metrics_history.get('val_loss', [])
        
        if not gradient_norms or not val_losses:
            return {}
        
        # Gradient stability
        gradient_stability = 1.0 / (1.0 + np.std(gradient_norms))
        
        # Loss volatility
        loss_volatility = np.std(val_losses) / np.mean(val_losses)
        
        # Overfitting detection
        train_losses = self.metrics_history.get('train_loss', [])
        if train_losses:
            final_gap = val_losses[-1] - train_losses[-1]
            overfitting_score = max(0, final_gap)
        else:
            overfitting_score = 0.0
        
        return {
            'gradient_stability': gradient_stability,
            'loss_volatility': loss_volatility,
            'overfitting_score': overfitting_score
        }

class AdaptiveLossMetrics:
    """Metrics specific to adaptive loss systems"""
    
    def __init__(self):
        self.adaptation_history = []
        self.reward_history = []
        self.meta_learning_history = []
    
    def update_adaptation(self, adaptation_params: Dict, reward: float, 
                         meta_prediction: Optional[np.ndarray] = None):
        """Update adaptation metrics"""
        self.adaptation_history.append(adaptation_params)
        self.reward_history.append(reward)
        
        if meta_prediction is not None:
            self.meta_learning_history.append(meta_prediction)
    
    def calculate_adaptation_metrics(self) -> Dict[str, float]:
        """Calculate adaptation-specific metrics"""
        if not self.adaptation_history:
            return {}
        
        # Adaptation frequency
        main_weights = [params.get('main_weight', 1.0) for params in self.adaptation_history]
        aux_weights = [params.get('aux_weight', 0.0) for params in self.adaptation_history]
        
        # How often does the system adapt?
        adaptation_frequency = len(set(main_weights)) / len(main_weights)
        
        # Adaptation magnitude
        weight_changes = [abs(main_weights[i] - main_weights[i-1]) 
                         for i in range(1, len(main_weights))]
        avg_adaptation_magnitude = np.mean(weight_changes) if weight_changes else 0.0
        
        # Reward progression
        reward_trend = 0.0
        if len(self.reward_history) > 1:
            x = np.arange(len(self.reward_history))
            y = np.array(self.reward_history)
            try:
                slope, _, _, _, _ = stats.linregress(x, y)
                reward_trend = slope
            except:
                reward_trend = 0.0
        
        # Meta-learning effectiveness
        meta_consistency = 0.0
        if len(self.meta_learning_history) > 10:
            # Calculate consistency of meta-predictions
            predictions = np.array(self.meta_learning_history)
            meta_consistency = 1.0 / (1.0 + np.mean(np.std(predictions, axis=0)))
        
        return {
            'adaptation_frequency': adaptation_frequency,
            'avg_adaptation_magnitude': avg_adaptation_magnitude,
            'reward_trend': reward_trend,
            'meta_consistency': meta_consistency,
            'final_reward': self.reward_history[-1] if self.reward_history else 0.0
        }

class ComparisonMetrics:
    """Metrics for comparing different training approaches"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, metrics: Dict[str, float]):
        """Add results from a training run"""
        self.results[name] = metrics
    
    def calculate_improvements(self, baseline_name: str) -> Dict[str, Dict[str, float]]:
        """Calculate improvements over baseline"""
        if baseline_name not in self.results:
            return {}
        
        baseline_metrics = self.results[baseline_name]
        improvements = {}
        
        for name, metrics in self.results.items():
            if name == baseline_name:
                continue
            
            improvement = {}
            for metric_name, value in metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    
                    # Calculate relative improvement
                    if baseline_value != 0:
                        if metric_name in ['loss', 'convergence_epoch', 'training_time']:
                            # Lower is better
                            improvement[metric_name] = (baseline_value - value) / baseline_value
                        else:
                            # Higher is better
                            improvement[metric_name] = (value - baseline_value) / baseline_value
                    else:
                        improvement[metric_name] = 0.0
            
            improvements[name] = improvement
        
        return improvements
    
    def rank_methods(self, metric_name: str, higher_is_better: bool = True) -> List[Tuple[str, float]]:
        """Rank methods by specific metric"""
        if metric_name not in self.results[list(self.results.keys())[0]]:
            return []
        
        items = [(name, metrics[metric_name]) for name, metrics in self.results.items()]
        items.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        return items
    
    def generate_comparison_report(self, baseline_name: str) -> str:
        """Generate a comprehensive comparison report"""
        improvements = self.calculate_improvements(baseline_name)
        
        report = f"# Comparison Report (Baseline: {baseline_name})\n\n"
        
        for method_name, method_improvements in improvements.items():
            report += f"## {method_name}\n\n"
            
            for metric_name, improvement in method_improvements.items():
                sign = "+" if improvement > 0 else ""
                report += f"- {metric_name}: {sign}{improvement:.1%}\n"
            
            report += "\n"
        
        return report

def calculate_statistical_significance(results_a: List[float], results_b: List[float], 
                                     alpha: float = 0.05) -> Dict[str, Any]:
    """Calculate statistical significance between two sets of results"""
    try:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results_a) - 1) * np.var(results_a, ddof=1) + 
                             (len(results_b) - 1) * np.var(results_b, ddof=1)) / 
                            (len(results_a) + len(results_b) - 2))
        
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'u_statistic': u_stat,
            'u_p_value': u_p_value,
            'mean_a': np.mean(results_a),
            'mean_b': np.mean(results_b),
            'std_a': np.std(results_a),
            'std_b': np.std(results_b)
        }
    except Exception as e:
        warnings.warn(f"Statistical test failed: {e}")
        return {
            'significant': False,
            'error': str(e)
        }
