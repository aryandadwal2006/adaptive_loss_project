import torch
import torch.nn as nn
import torch.optim as optim

class BaselineTrainer:
    """
    Standard trainer without adaptive loss.
    """
    def __init__(self, model: nn.Module, loss_fn, lr: float, weight_decay: float):
        self.model = model.to(model.device if hasattr(model, 'device') else torch.device('cpu'))
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {'loss': [], 'val_acc': []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.history['loss'].append(total_loss / len(loader))

    def validate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                preds = self.model(x)
                _, p = preds.max(1)
                correct += (p == y).sum().item()
                total += y.size(0)
        acc = correct / total
        self.history['val_acc'].append(acc)
        return acc

    def train(self, train_loader, val_loader, epochs, log_interval):
        for epoch in range(1, epochs+1):
            self.train_epoch(train_loader)
            acc = self.validate(val_loader)
            if epoch % log_interval == 0:
                print(f"[Baseline] Epoch {epoch} | Loss {self.history['loss'][-1]:.4f} | Val Acc {acc*100:.2f}%")
