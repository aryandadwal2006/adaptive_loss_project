"""
Advanced Visualization Tools for Adaptive Loss Experiments
Comprehensive plotting and visualization utilities for experiment analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExperimentVisualizer:
    """
    Comprehensive visualization system for adaptive loss experiments
    Supports both static (matplotlib) and interactive (plotly) visualizations
    """
    
    def __init__(self, output_dir: str = "./plots", 
                 style: str = "seaborn", 
                 figsize: Tuple[int, int] = (12, 8)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.figsize = figsize
        
        # Color schemes
        self.colors = {
            'baseline': '#1f77b4',
            'adaptive': '#ff7f0e',
            'ablation': '#2ca02c',
            'best': '#d62728',
            'worst': '#9467bd'
        }
        
        # Plot configurations
        self.plot_configs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        print(f"Experiment Visualizer initialized")
        print(f"Output directory: {self.output_dir}")
    
    def plot_training_curves(self, history: Dict[str, List[float]], 
                           title: str = "Training Curves",
                           save_name: str = "training_curves") -> plt.Figure:
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color=self.colors['adaptive'])
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color=self.colors['baseline'])
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color=self.colors['adaptive'])
            axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color=self.colors['baseline'])
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if 'gradient_norms' in history:
            axes[1, 0].plot(history['gradient_norms'], color=self.colors['best'])
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rates
        if 'learning_rates' in history:
            axes[1, 1].plot(history['learning_rates'], color=self.colors['worst'])
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig
    
    def plot_adaptive_metrics(self, adaptive_history: Dict[str, List[float]], 
                             title: str = "Adaptive Loss Metrics",
                             save_name: str = "adaptive_metrics") -> plt.Figure:
        """Plot adaptive loss specific metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Reward progression
        if 'rewards' in adaptive_history:
            rewards = adaptive_history['rewards']
            axes[0, 0].plot(rewards, color=self.colors['adaptive'])
            axes[0, 0].set_title('RL Agent Rewards')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average
            if len(rewards) > 10:
                window_size = min(50, len(rewards) // 10)
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                axes[0, 0].plot(moving_avg, color=self.colors['best'], 
                              label=f'Moving Average ({window_size})', alpha=0.7)
                axes[0, 0].legend()
        
        # Action distribution
        if 'actions' in adaptive_history:
            actions = adaptive_history['actions']
            action_counts = pd.Series(actions).value_counts().sort_index()
            axes[0, 1].bar(action_counts.index, action_counts.values, 
                          color=self.colors['adaptive'], alpha=0.7)
            axes[0, 1].set_title('Action Distribution')
            axes[0, 1].set_xlabel('Action')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss weight evolution
        if 'main_weights' in adaptive_history:
            axes[1, 0].plot(adaptive_history['main_weights'], 
                          label='Main Weight', color=self.colors['adaptive'])
            if 'aux_weights' in adaptive_history:
                axes[1, 0].plot(adaptive_history['aux_weights'], 
                              label='Auxiliary Weight', color=self.colors['baseline'])
            axes[1, 0].set_title('Loss Weight Evolution')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon decay (exploration)
        if 'epsilon' in adaptive_history:
            axes[1, 1].plot(adaptive_history['epsilon'], color=self.colors['worst'])
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig
    
    def plot_comparison_results(self, results: Dict[str, Dict[str, float]], 
                               title: str = "Experiment Comparison",
                               save_name: str = "comparison") -> plt.Figure:
        """Plot comparison between different experiments"""
        # Convert results to DataFrame
        df_data = []
        for exp_name, metrics in results.items():
            for metric_name, value in metrics.items():
                df_data.append({
                    'Experiment': exp_name,
                    'Metric': metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(df_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Bar plot for final metrics
        final_metrics = ['final_val_acc', 'final_val_loss', 'final_train_acc', 'final_train_loss']
        
        for i, metric in enumerate(final_metrics):
            if i >= 4:
                break
            
            ax = axes[i // 2, i % 2]
            metric_data = df[df['Metric'] == metric]
            
            if not metric_data.empty:
                bars = ax.bar(metric_data['Experiment'], metric_data['Value'])
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Color bars
                for j, bar in enumerate(bars):
                    if 'baseline' in metric_data.iloc[j]['Experiment']:
                        bar.set_color(self.colors['baseline'])
                    else:
                        bar.set_color(self.colors['adaptive'])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig
    
    def plot_ablation_analysis(self, ablation_results: Dict[str, Any], 
                              title: str = "Ablation Study Analysis",
                              save_name: str = "ablation_analysis") -> plt.Figure:
        """Plot ablation study results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Component importance
        if 'component_importance' in ablation_results:
            importance = ablation_results['component_importance']
            
            # Extract validation accuracy impacts
            names = []
            impacts = []
            
            for config_name, config_impacts in importance.items():
                if 'final_val_acc' in config_impacts:
                    names.append(config_name.replace('_', ' ').title())
                    impacts.append(config_impacts['final_val_acc'] * 100)  # Convert to percentage
            
            if names:
                bars = axes[0, 0].barh(names, impacts)
                axes[0, 0].set_title('Component Importance\n(Validation Accuracy Impact)')
                axes[0, 0].set_xlabel('Impact (%)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Color bars based on impact
                for bar, impact in zip(bars, impacts):
                    if impact < 0:
                        bar.set_color(self.colors['worst'])
                    else:
                        bar.set_color(self.colors['best'])
        
        # Performance ranking
        if 'performance_ranking' in ablation_results:
            ranking = ablation_results['performance_ranking']
            
            if 'final_val_acc' in ranking:
                rank_data = ranking['final_val_acc']
                names = [item[0].replace('_', ' ').title() for item in rank_data]
                scores = [item[1] for item in rank_data]
                
                bars = axes[0, 1].bar(range(len(names)), scores)
                axes[0, 1].set_title('Performance Ranking\n(Validation Accuracy)')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].set_xticks(range(len(names)))
                axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Color bars
                for i, bar in enumerate(bars):
                    if i == 0:
                        bar.set_color(self.colors['best'])
                    elif i == len(bars) - 1:
                        bar.set_color(self.colors['worst'])
                    else:
                        bar.set_color(self.colors['adaptive'])
        
        # Create heatmap of component interactions (if available)
        if len(importance) > 1:
            interaction_matrix = np.zeros((len(importance), len(importance)))
            component_names = list(importance.keys())
            
            for i, comp1 in enumerate(component_names):
                for j, comp2 in enumerate(component_names):
                    if i != j:
                        # Simulate interaction effect (in real implementation, this would be calculated)
                        interaction_matrix[i, j] = np.random.normal(0, 0.1)
            
            im = axes[1, 0].imshow(interaction_matrix, cmap='RdBu', aspect='auto')
            axes[1, 0].set_title('Component Interactions\n(Simulated)')
            axes[1, 0].set_xticks(range(len(component_names)))
            axes[1, 0].set_yticks(range(len(component_names)))
            axes[1, 0].set_xticklabels([name.replace('_', ' ') for name in component_names], 
                                      rotation=45, ha='right')
            axes[1, 0].set_yticklabels([name.replace('_', ' ') for name in component_names])
            plt.colorbar(im, ax=axes[1, 0])
        
        # Summary statistics
        if 'insights' in ablation_results:
            insights = ablation_results['insights']
            
            # Create text summary
            axes[1, 1].text(0.05, 0.95, 'Key Insights:', fontsize=14, fontweight='bold',
                           transform=axes[1, 1].transAxes, verticalalignment='top')
            
            for i, insight in enumerate(insights[:5]):  # Show top 5 insights
                axes[1, 1].text(0.05, 0.85 - i*0.15, f"• {insight}", fontsize=10,
                               transform=axes[1, 1].transAxes, verticalalignment='top',
                               wrap=True)
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig
    
    def create_interactive_dashboard(self, experiment_data: Dict[str, Any], 
                                   title: str = "Adaptive Loss Dashboard") -> go.Figure:
        """Create interactive dashboard using plotly"""
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Validation Accuracy', 
                          'Reward Progression', 'Action Distribution',
                          'Loss Weight Evolution', 'Convergence Comparison'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Training loss
        if 'train_loss' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['train_loss']))),
                          y=experiment_data['train_loss'],
                          name='Train Loss',
                          line=dict(color=self.colors['adaptive'])),
                row=1, col=1
            )
        
        if 'val_loss' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['val_loss']))),
                          y=experiment_data['val_loss'],
                          name='Validation Loss',
                          line=dict(color=self.colors['baseline'])),
                row=1, col=1
            )
        
        # Validation accuracy
        if 'val_acc' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['val_acc']))),
                          y=experiment_data['val_acc'],
                          name='Validation Accuracy',
                          line=dict(color=self.colors['best'])),
                row=1, col=2
            )
        
        # Reward progression
        if 'rewards' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['rewards']))),
                          y=experiment_data['rewards'],
                          name='Rewards',
                          line=dict(color=self.colors['adaptive'])),
                row=2, col=1
            )
        
        # Action distribution
        if 'actions' in experiment_data:
            action_counts = pd.Series(experiment_data['actions']).value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=action_counts.index,
                       y=action_counts.values,
                       name='Actions',
                       marker_color=self.colors['adaptive']),
                row=2, col=2
            )
        
        # Loss weight evolution
        if 'main_weights' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['main_weights']))),
                          y=experiment_data['main_weights'],
                          name='Main Weight',
                          line=dict(color=self.colors['adaptive'])),
                row=3, col=1
            )
        
        if 'aux_weights' in experiment_data:
            fig.add_trace(
                go.Scatter(x=list(range(len(experiment_data['aux_weights']))),
                          y=experiment_data['aux_weights'],
                          name='Auxiliary Weight',
                          line=dict(color=self.colors['baseline'])),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            height=900,
            hovermode='x unified'
        )
        
        # Save interactive plot
        save_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(save_path))
        
        return fig
    
    def plot_statistical_analysis(self, statistical_results: Dict[str, Any], 
                                 title: str = "Statistical Analysis",
                                 save_name: str = "statistical_analysis") -> plt.Figure:
        """Plot statistical analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # P-value visualization
        if 'statistical_test' in statistical_results:
            test_result = statistical_results['statistical_test']
            
            # Create p-value comparison
            p_value = test_result.p_value
            significance_level = 0.05
            
            axes[0, 0].bar(['P-value', 'Significance Level'], 
                          [p_value, significance_level],
                          color=[self.colors['adaptive'] if p_value < significance_level else self.colors['worst'],
                                self.colors['baseline']])
            axes[0, 0].set_title('Statistical Significance')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add significance indicator
            if p_value < significance_level:
                axes[0, 0].text(0.5, 0.8, 'SIGNIFICANT', 
                              transform=axes[0, 0].transAxes,
                              fontsize=14, fontweight='bold',
                              ha='center', color=self.colors['best'])
        
        # Effect size visualization
        if 'statistical_test' in statistical_results:
            effect_size = statistical_results['statistical_test'].effect_size
            
            # Cohen's d interpretation
            if abs(effect_size) < 0.2:
                effect_label = 'Small'
                color = self.colors['worst']
            elif abs(effect_size) < 0.5:
                effect_label = 'Medium'
                color = self.colors['adaptive']
            else:
                effect_label = 'Large'
                color = self.colors['best']
            
            axes[0, 1].bar(['Effect Size'], [effect_size], color=color)
            axes[0, 1].set_title(f'Effect Size ({effect_label})')
            axes[0, 1].set_ylabel("Cohen's d")
            axes[0, 1].grid(True, alpha=0.3)
        
        # Confidence intervals
        if 'baseline_stats' in statistical_results and 'adaptive_stats' in statistical_results:
            baseline_stats = statistical_results['baseline_stats']
            adaptive_stats = statistical_results['adaptive_stats']
            
            # Create confidence interval plot
            methods = ['Baseline', 'Adaptive']
            means = [baseline_stats['mean'], adaptive_stats['mean']]
            stds = [baseline_stats['std'], adaptive_stats['std']]
            
            axes[1, 0].bar(methods, means, yerr=stds, capsize=10,
                          color=[self.colors['baseline'], self.colors['adaptive']])
            axes[1, 0].set_title('Performance Comparison\n(Mean ± Std)')
            axes[1, 0].set_ylabel('Performance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Power analysis
        if 'power_analysis' in statistical_results:
            power_analysis = statistical_results['power_analysis']
            
            power = power_analysis['power']
            sample_size = power_analysis['sample_size']
            
            axes[1, 1].bar(['Statistical Power'], [power], 
                          color=self.colors['best'] if power >= 0.8 else self.colors['worst'])
            axes[1, 1].set_title(f'Statistical Power\n(Sample Size: {sample_size})')
            axes[1, 1].set_ylabel('Power')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add power adequacy line
            axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Adequate Power')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig
    
    def create_comprehensive_report(self, all_results: Dict[str, Any], 
                                  title: str = "Comprehensive Experiment Report") -> List[plt.Figure]:
        """Create a comprehensive visual report"""
        figures = []
        
        # Training curves
        if 'training_history' in all_results:
            fig = self.plot_training_curves(all_results['training_history'], 
                                          title="Training Progress", 
                                          save_name="comprehensive_training")
            figures.append(fig)
        
        # Adaptive metrics
        if 'adaptive_history' in all_results:
            fig = self.plot_adaptive_metrics(all_results['adaptive_history'], 
                                           title="Adaptive Loss Analysis", 
                                           save_name="comprehensive_adaptive")
            figures.append(fig)
        
        # Comparison results
        if 'comparison_results' in all_results:
            fig = self.plot_comparison_results(all_results['comparison_results'], 
                                             title="Method Comparison", 
                                             save_name="comprehensive_comparison")
            figures.append(fig)
        
        # Ablation analysis
        if 'ablation_results' in all_results:
            fig = self.plot_ablation_analysis(all_results['ablation_results'], 
                                            title="Ablation Study", 
                                            save_name="comprehensive_ablation")
            figures.append(fig)
        
        # Statistical analysis
        if 'statistical_results' in all_results:
            fig = self.plot_statistical_analysis(all_results['statistical_results'], 
                                               title="Statistical Validation", 
                                               save_name="comprehensive_statistical")
            figures.append(fig)
        
        # Create summary figure
        summary_fig = self._create_summary_figure(all_results)
        figures.append(summary_fig)
        
        return figures
    
    def _create_summary_figure(self, all_results: Dict[str, Any]) -> plt.Figure:
        """Create a summary figure with key metrics"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract key metrics
        summary_data = {
            'Metric': [],
            'Value': [],
            'Type': []
        }
        
        # Add metrics from different sources
        if 'final_metrics' in all_results:
            metrics = all_results['final_metrics']
            for key, value in metrics.items():
                summary_data['Metric'].append(key.replace('_', ' ').title())
                summary_data['Value'].append(value)
                summary_data['Type'].append('Final')
        
        if summary_data['Metric']:
            df = pd.DataFrame(summary_data)
            
            # Create grouped bar plot
            metrics_to_plot = df['Metric'].unique()[:8]  # Top 8 metrics
            
            bars = ax.bar(range(len(metrics_to_plot)), 
                         [df[df['Metric'] == m]['Value'].iloc[0] for m in metrics_to_plot])
            
            ax.set_title('Experiment Summary - Key Metrics')
            ax.set_xticks(range(len(metrics_to_plot)))
            ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Color bars
            for bar in bars:
                bar.set_color(self.colors['adaptive'])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / "summary_report.png"
        plt.savefig(save_path, **self.plot_configs)
        
        return fig

if __name__ == "__main__":
    # Test the visualization system
    print("Testing Experiment Visualizer...")
    
    # Create mock data
    training_history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1],
        'val_loss': [0.85, 0.65, 0.45, 0.35, 0.25, 0.18, 0.15, 0.12],
        'train_acc': [0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94],
        'val_acc': [0.58, 0.68, 0.78, 0.83, 0.86, 0.88, 0.89, 0.91],
        'gradient_norms': [2.1, 1.8, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6]
    }
    
    adaptive_history = {
        'rewards': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'actions': [0, 1, 2, 1, 0, 1, 2, 1],
        'main_weights': [1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9],
        'aux_weights': [0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.2, 0.1]
    }
    
    # Create visualizer
    visualizer = ExperimentVisualizer(output_dir="./test_plots")
    
    # Test training curves
    fig1 = visualizer.plot_training_curves(training_history, "Test Training Curves")
    
    # Test adaptive metrics
    fig2 = visualizer.plot_adaptive_metrics(adaptive_history, "Test Adaptive Metrics")
    
    # Test comparison results
    comparison_results = {
        'baseline': {'final_val_acc': 0.85, 'final_val_loss': 0.15},
        'adaptive': {'final_val_acc': 0.91, 'final_val_loss': 0.12}
    }
    fig3 = visualizer.plot_comparison_results(comparison_results, "Test Comparison")
    
    # Test interactive dashboard
    dashboard_data = {**training_history, **adaptive_history}
    interactive_fig = visualizer.create_interactive_dashboard(dashboard_data, "Test Dashboard")
    
    print("Visualization test completed successfully!")
    print(f"Plots saved to: {visualizer.output_dir}")
