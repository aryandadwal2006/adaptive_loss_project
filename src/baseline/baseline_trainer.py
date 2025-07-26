"""
Comprehensive Baseline Training Framework
Multiple baseline implementations for comparison with adaptive loss system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy

from ..config import TRAINING_CONFIG, DEVICE
from ..utils.logging_utils import ExperimentLogger

class BaselineTrainer:
    """
    Standard baseline trainer with static loss functions
    Serves as primary comparison for adaptive loss system
    """
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module, 
                 optimizer: optim.Optimizer, experiment_name: str = "baseline"):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.experiment_name = experiment_name
        self.device = DEVICE
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'gradient_norms': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Logger
        self.logger = ExperimentLogger(experiment_name)
        
        # Best model tracking
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient norm computation
            grad_norm = self._compute_gradient_norm()
            gradient_norms.append(grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            if batch_idx % 50 == 0:
                self.logger.log_batch(epoch, batch_idx, loss.item(), 
                                    correct/total, grad_norm)
        
        epoch_time = time.time() - start_time
        
        # Epoch metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total,
            'gradient_norm': np.mean(gradient_norms),
            'epoch_time': epoch_time
        }
        
        return metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs: int = None) -> Dict[str, List[float]]:
        """Complete training loop"""
        num_epochs = num_epochs or TRAINING_CONFIG['num_epochs']
        
        self.logger.log_info(f"Starting {self.experiment_name} training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['gradient_norms'].append(train_metrics['gradient_norm'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(train_metrics['epoch_time'])
            
            # Best model tracking
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Logging
            self.logger.log_epoch(epoch, train_metrics, val_metrics)
            
            # Console output
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Train Acc: {train_metrics['accuracy']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Final logging
        self.logger.log_info(f"Training completed. Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        return self.history
    
    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def get_best_model(self) -> nn.Module:
        """Return model with best validation performance"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics"""
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history['train_loss']),
            'avg_epoch_time': np.mean(self.history['epoch_times']),
            'convergence_speed': self._calculate_convergence_speed()
        }
    
    def _calculate_convergence_speed(self) -> int:
        """Calculate convergence speed (epochs to 90% of best performance)"""
        if len(self.history['val_loss']) < 10:
            return len(self.history['val_loss'])
        
        target_loss = self.best_val_loss * 1.1  # 90% of best performance
        
        for i, loss in enumerate(self.history['val_loss']):
            if loss <= target_loss:
                return i
        
        return len(self.history['val_loss'])

class RandomBaselineTrainer(BaselineTrainer):
    """Baseline that makes random predictions"""
    
    def __init__(self, num_classes: int, experiment_name: str = "random_baseline"):
        # Dummy model for consistency
        class DummyModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
                
            def forward(self, x):
                batch_size = x.size(0)
                return torch.randn(batch_size, self.num_classes).to(x.device)
        
        model = DummyModel(num_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        super().__init__(model, loss_fn, optimizer, experiment_name)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Random predictions - no actual training"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in train_loader:
                batch_size = data.size(0)
                # Random predictions
                predicted = torch.randint(0, self.model.num_classes, (batch_size,))
                total += target.size(0)
                correct += (predicted == target.cpu()).sum().item()
        
        return {
            'loss': -np.log(1.0 / self.model.num_classes),  # Theoretical random loss
            'accuracy': correct / total,
            'gradient_norm': 0.0,
            'epoch_time': 0.01  # Minimal time
        }

class MajorityClassBaselineTrainer(BaselineTrainer):
    """Baseline that always predicts the majority class"""
    
    def __init__(self, num_classes: int, experiment_name: str = "majority_baseline"):
        class DummyModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
                self.majority_class = 0  # Will be set during training
                
            def forward(self, x):
                batch_size = x.size(0)
                output = torch.zeros(batch_size, self.num_classes).to(x.device)
                output[:, self.majority_class] = 1.0
                return output
        
        model = DummyModel(num_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        super().__init__(model, loss_fn, optimizer, experiment_name)
        self.class_counts = defaultdict(int)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Count classes in first epoch, then predict majority"""
        if epoch == 0:
            # Count classes
            for _, target in train_loader:
                for t in target:
                    self.class_counts[t.item()] += 1
            
            # Set majority class
            self.model.majority_class = max(self.class_counts.keys(), 
                                          key=lambda x: self.class_counts[x])
        
        # Evaluate performance
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _, target in train_loader:
                predicted = torch.full_like(target, self.model.majority_class)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        majority_prob = self.class_counts[self.model.majority_class] / sum(self.class_counts.values())
        
        return {
            'loss': -np.log(majority_prob),
            'accuracy': correct / total,
            'gradient_norm': 0.0,
            'epoch_time': 0.01
        }

class LinearDecayLossTrainer(BaselineTrainer):
    """Baseline with linearly decaying loss weights"""
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module, 
                 optimizer: optim.Optimizer, decay_rate: float = 0.95,
                 experiment_name: str = "linear_decay_baseline"):
        super().__init__(model, loss_fn, optimizer, experiment_name)
        self.decay_rate = decay_rate
        self.current_weight = 1.0
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train with linearly decaying loss weights"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        # Update loss weight
        self.current_weight *= self.decay_rate
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target) * self.current_weight
            
            loss.backward()
            
            grad_norm = self._compute_gradient_norm()
            gradient_norms.append(grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total,
            'gradient_norm': np.mean(gradient_norms),
            'epoch_time': epoch_time
        }
