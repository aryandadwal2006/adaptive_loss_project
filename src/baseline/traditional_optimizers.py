"""
Traditional Optimizers Comparison
Provides trainer variants for SGD, RMSProp, and Adagrad
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ..config import TRAINING_CONFIG, DEVICE
from ..utils.logging_utils import ExperimentLogger

class OptimizerComparison:
    """
    Trainer that can switch between traditional optimizers to compare.
    """
    def __init__(self, model: nn.Module, loss_fn: nn.Module,
                 optimizer_name: str = "SGD", lr: float = 0.01,
                 experiment_name: str = "optimizer_compare"):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.logger = ExperimentLogger(f"{experiment_name}_{optimizer_name.lower()}")
        optimizers = {
            "SGD": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
            "RMSProp": optim.RMSprop(model.parameters(), lr=lr),
            "Adagrad": optim.Adagrad(model.parameters(), lr=lr)
        }
        self.optimizer = optimizers[optimizer_name]
        self.history = {'loss': [], 'acc': []}

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for x,y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, p = out.max(1)
            correct += (p==y).sum().item()
            total += y.size(0)
        avg_loss = total_loss/len(loader)
        acc = correct/total
        self.history['loss'].append(avg_loss)
        self.history['acc'].append(acc)
        self.logger.log_epoch(epoch, {'loss': avg_loss, 'accuracy': acc}, {'loss':0,'accuracy':0})
        return avg_loss, acc

    def train(self, train_loader, val_loader):
        for e in range(TRAINING_CONFIG['num_epochs']):
            self.train_epoch(train_loader, e)
        return self.history
