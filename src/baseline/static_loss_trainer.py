"""
Static Loss Trainer
Implements training with fixed loss schedules for comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from ..config import TRAINING_CONFIG, DEVICE
from ..utils.logging_utils import ExperimentLogger

class StaticLossTrainer:
    """
    Trainer that uses a static weighted sum of base and auxiliary losses
    with no adaptation.
    """
    def __init__(self, model: nn.Module, 
                 base_loss_fn: nn.Module, 
                 aux_loss_fn: nn.Module,
                 base_weight: float = 1.0,
                 aux_weight: float = 0.0,
                 experiment_name: str = "static_loss"):
        self.model = model.to(DEVICE)
        self.base_loss_fn = base_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=TRAINING_CONFIG['learning_rate'],
                                    weight_decay=TRAINING_CONFIG['weight_decay'])
        self.logger = ExperimentLogger(experiment_name)
        self.history = {'loss': [], 'acc': []}

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        start = time.time()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.base_weight * self.base_loss_fn(preds, y)
            if self.aux_weight > 0:
                loss += self.aux_weight * self.aux_loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, p = preds.max(1)
            correct += (p == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss/len(loader)
        acc = correct/total
        self.history['loss'].append(avg_loss)
        self.history['acc'].append(acc)
        self.logger.log_epoch(epoch, {'loss': avg_loss, 'accuracy': acc}, {'loss':0,'accuracy':0})
        return avg_loss, acc, time.time()-start

    def train(self, train_loader, val_loader):
        for e in range(TRAINING_CONFIG['num_epochs']):
            self.train_epoch(train_loader, e)
        return self.history
