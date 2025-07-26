"""
Adaptive Loss Trainer
Orchestrates model training with RL-based loss adaptation and meta-learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .loss_monitor import LossMonitorAgent
from .meta_controller import MetaLearningController
from .loss_executor import DynamicLossExecutor
from .config import TRAINING_CONFIG, DEVICE, DATASET_CONFIG

class AdaptiveLossTrainer:
    def __init__(self, model: nn.Module, base_loss_fn, aux_loss_fn=None):
        self.model = model.to(DEVICE)
        self.device = DEVICE
        self.base_loss_fn = base_loss_fn
        self.aux_loss_fn = aux_loss_fn

        # Components
        self.loss_monitor = LossMonitorAgent()
        self.meta_controller = MetaLearningController()
        self.loss_executor = DynamicLossExecutor(base_loss_fn, aux_loss_fn)

        # Optimizer
        cfg = TRAINING_CONFIG
        self.optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                    weight_decay=cfg['weight_decay'])

        # History
        self.history = {'loss': [], 'reward': [], 'grad_norm': []}
        self.prev_metrics = None

    def train_epoch(self, dataloader, epoch_idx):
        cfg = TRAINING_CONFIG
        self.model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            iteration = epoch_idx * len(dataloader) + batch_idx
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            preds = self.model(x)

            # State & actions
            state = self.loss_monitor.get_state(
                self.loss_executor.loss_history,
                self.loss_executor.grad_stats,
                {'iteration': iteration, 'epoch': epoch_idx}
            )
            task_enc = self.meta_controller.encode_task(
                model_type=0,
                dataset_size=x.size(0),
                complexity=0.5
            )
            meta_strat = self.meta_controller.predict(task_enc, state)
            action = self.loss_monitor.act(state, training=True)
            rl_params = self.loss_monitor.action_to_params(action)

            # Combine
            m = float(rl_params['main_weight'])
            a = float(rl_params['aux_weight'])
            sf = float(rl_params['schedule_factor'])
            meta = meta_strat.squeeze().detach().cpu().numpy()
            combined = {
                'main_weight': m * (1 + 0.1 * meta[0]),
                'aux_weight' : a * (1 + 0.1 * meta[1]),
                'schedule'   : sf * (1 + 0.05 * meta[2])
            }

            # Compute adaptive loss
            loss = self.loss_executor.compute(preds, y, combined)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          cfg.get('max_grad_norm', 1.0))
            self.optimizer.step()

            # Grad stats
            grad_stats = self.loss_executor.compute_grad_stats(self.model)

            # Reward
            curr_metrics = {
                'loss': loss.item(),
                'gradient_norm': grad_stats['norm'],
                'stable': self.loss_executor.is_stable()
            }
            reward = self.loss_monitor.compute_reward(
                curr_metrics, self.prev_metrics or curr_metrics, combined
            )
            self.prev_metrics = curr_metrics

            # RL update
            next_state = self.loss_monitor.get_state(
                self.loss_executor.loss_history,
                self.loss_executor.grad_stats,
                {'iteration': iteration+1, 'epoch': epoch_idx}
            )
            self.loss_monitor.remember(state, action, reward, next_state, False)
            self.loss_monitor.replay()

            # Logging
            self.history['loss'].append(loss.item())
            self.history['reward'].append(reward)
            self.history['grad_norm'].append(grad_stats['norm'])

    def validate(self, dataloader):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                _, pred_labels = preds.max(1)
                total += y.size(0)
                correct += (pred_labels == y).sum().item()
        return correct / total

    def train(self, train_loader, val_loader):
        cfg = TRAINING_CONFIG
        for epoch in range(cfg['num_epochs']):
            self.train_epoch(train_loader, epoch)
            if (epoch + 1) % cfg['log_interval'] == 0:
                acc = self.validate(val_loader)
                print(f"Epoch {epoch+1} | Loss {self.history['loss'][-1]:.4f} "
                      f"| Reward {self.history['reward'][-1]:.4f} "
                      f"| Val Acc {acc*100:.2f}%")
            if (epoch + 1) % cfg['save_interval'] == 0:
                self.loss_monitor.update_target_network()
