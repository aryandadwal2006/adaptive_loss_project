# File: src/meta_controller.py (enhanced version)

import numpy as np
import torch
from src.config import META_CONFIG

class MetaLearningController:
    """
    Enhanced meta-learning controller that can work independently or with RL monitor
    """
    
    def __init__(self):
        self.task_history = []
        self.performance_history = []
        self.meta_params = {
            'adaptation_rate': META_CONFIG.get('adaptation_rate', 0.1),
            'task_embedding_dim': META_CONFIG.get('task_embedding_dim', 16)
        }
        self.learned_strategies = {}
    
    def adapt_params(self, base_params, task_metadata):
        """
        Adapt loss parameters based on task metadata and learned strategies
        """
        # Extract task features
        epoch = task_metadata.get('epoch', 0)
        batch = task_metadata.get('batch', 0)
        
        # Create task signature
        task_phase = 'early' if epoch < 10 else 'mid' if epoch < 20 else 'late'
        
        # Apply learned adaptations
        adapted_params = base_params.copy()
        
        if task_phase == 'early':
            # Early training: be more conservative
            adapted_params['main_weight'] *= 0.9
            adapted_params['aux_weight'] = adapted_params.get('aux_weight', 0.0) + 0.1
        elif task_phase == 'mid':
            # Mid training: standard adaptation
            adapted_params['main_weight'] *= 1.0
        else:
            # Late training: focus on main loss
            adapted_params['main_weight'] *= 1.1
            adapted_params['aux_weight'] = max(0, adapted_params.get('aux_weight', 0.0) - 0.05)
        
        # Ensure bounds
        adapted_params['main_weight'] = np.clip(adapted_params['main_weight'], 0.5, 2.0)
        adapted_params['aux_weight'] = np.clip(adapted_params.get('aux_weight', 0.0), 0.0, 0.5)
        
        return adapted_params
    
    def update_strategy(self, task_metadata, performance_metrics):
        """
        Update meta-learning strategies based on performance feedback
        """
        self.task_history.append(task_metadata)
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
            self.performance_history = self.performance_history[-100:]
    
    def get_state(self):
        """
        Get current meta-controller state
        """
        return {
            'n_tasks_seen': len(self.task_history),
            'avg_performance': np.mean([p.get('accuracy', 0) for p in self.performance_history[-10:]]) if self.performance_history else 0,
            'meta_params': self.meta_params.copy()
        }
