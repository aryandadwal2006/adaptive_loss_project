# File: src/trainer/train_ablation.py

import os
import sys

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data.cifar10_loader import get_cifar10_loaders
from src.models.simple_cnn import SimpleCNN
from src.loss_monitor import LossMonitorAgent
from src.meta_controller import MetaLearningController
from src.loss_executor import DynamicLossExecutor
from src.utils.logging_utils import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study: Component Analysis")
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=["baseline", "monitor_only", "controller_only", "full_adaptive"],
                        help="Ablation mode to run")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs/ablation",
                        help="Log directory")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_adaptive_components(mode: str, criterion):
    """
    Create adaptive components based on ablation mode
    """
    monitor = None
    controller = None
    executor = None
    
    if mode == "baseline":
        # No adaptive components
        return None, None, None
        
    elif mode == "monitor_only":
        # Only RL monitor, no meta-controller
        monitor = LossMonitorAgent()
        executor = DynamicLossExecutor(criterion)
        executor.monitor = monitor
        return monitor, None, executor
        
    elif mode == "controller_only":
        # Only meta-controller, no RL monitor
        controller = MetaLearningController()
        executor = DynamicLossExecutor(criterion)
        executor.controller = controller
        # Create a dummy monitor that just returns default params
        class DummyMonitor:
            def get_state(self, loss_history, grad_stats, metadata):
                return np.zeros(8)  # Default state
            def act(self, state):
                return 0  # Default action
            def action_to_params(self, action):
                return {'main_weight': 1.0, 'aux_weight': 0.0}
        executor.monitor = DummyMonitor()
        return None, controller, executor
        
    elif mode == "full_adaptive":
        # Both components
        monitor = LossMonitorAgent()
        controller = MetaLearningController()
        executor = DynamicLossExecutor(criterion)
        executor.monitor = monitor
        executor.controller = controller
        return monitor, controller, executor

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, 
                monitor=None, controller=None, executor=None, mode="baseline"):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)

        if mode == "baseline":
            # Static loss
            loss = criterion(outputs, y)
        elif mode == "monitor_only":
            # Only RL-based adaptation
            state = executor.monitor.get_state(
                executor.loss_history,
                executor.grad_stats,
                {"iteration": batch_idx, "epoch": epoch}
            )
            action = executor.monitor.act(state)
            params = executor.monitor.action_to_params(action)
            loss = executor.compute(outputs, y, params)
        elif mode == "controller_only":
            # Only meta-learning adaptation
            params = {'main_weight': 1.0, 'aux_weight': 0.0}  # Default params
            if controller:
                # Meta-controller modifies params based on task context
                task_metadata = {"epoch": epoch, "batch": batch_idx}
                params = controller.adapt_params(params, task_metadata)
            loss = executor.compute(outputs, y, params)
        elif mode == "full_adaptive":
            # Both RL monitor and meta-controller
            state = executor.monitor.get_state(
                executor.loss_history,
                executor.grad_stats,
                {"iteration": batch_idx, "epoch": epoch}
            )
            action = executor.monitor.act(state)
            params = executor.monitor.action_to_params(action)
            
            # Meta-controller can further refine params
            if controller:
                task_metadata = {"epoch": epoch, "batch": batch_idx}
                params = controller.adapt_params(params, task_metadata)
            
            loss = executor.compute(outputs, y, params)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", avg_acc, epoch)
    return avg_loss, avg_acc

def validate(model, loader, criterion, device, epoch, writer,
             monitor=None, controller=None, executor=None, mode="baseline"):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            if mode == "baseline":
                loss = criterion(outputs, y)
            elif mode == "monitor_only":
                state = executor.monitor.get_state(
                    executor.loss_history,
                    executor.grad_stats,
                    {"iteration": batch_idx, "epoch": epoch}
                )
                action = executor.monitor.act(state)
                params = executor.monitor.action_to_params(action)
                loss = executor.compute(outputs, y, params)
            elif mode == "controller_only":
                params = {'main_weight': 1.0, 'aux_weight': 0.0}
                if controller:
                    task_metadata = {"epoch": epoch, "batch": batch_idx}
                    params = controller.adapt_params(params, task_metadata)
                loss = executor.compute(outputs, y, params)
            elif mode == "full_adaptive":
                state = executor.monitor.get_state(
                    executor.loss_history,
                    executor.grad_stats,
                    {"iteration": batch_idx, "epoch": epoch}
                )
                action = executor.monitor.act(state)
                params = executor.monitor.action_to_params(action)
                
                if controller:
                    task_metadata = {"epoch": epoch, "batch": batch_idx}
                    params = controller.adapt_params(params, task_metadata)
                
                loss = executor.compute(outputs, y, params)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_acc, epoch)
    return avg_loss, avg_acc

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    logger = ExperimentLogger("ablation", args.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=args.batch_size)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create components based on ablation mode
    monitor, controller, executor = create_adaptive_components(args.ablation_mode, criterion)

    logger.log_info(f"Running ablation mode: {args.ablation_mode}")
    logger.log_info(f"Monitor: {monitor is not None}, Controller: {controller is not None}")

    best_val_acc, best_epoch = 0.0, 0
    start_time = time.time()

    # Store per-epoch metrics for analysis
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            monitor, controller, executor, args.ablation_mode
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer,
            monitor, controller, executor, args.ablation_mode
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.log_info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}; "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), os.path.join(args.logdir, "best_model.pth"))

    total_time = time.time() - start_time
    logger.log_info(f"Completed in {total_time:.1f}s. Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}.")

    # Save detailed metrics for analysis
    metrics = {
        'ablation_mode': args.ablation_mode,
        'seed': args.seed,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    
    # Save metrics in multiple formats
    import json
    with open(os.path.join(args.logdir, "detailed_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save best metrics for experiment runner
    with open(os.path.join(args.logdir, "best_metrics.txt"), "w") as f:
        f.write(f"{best_epoch},{best_val_acc:.4f},{total_time:.2f}")

    writer.close()

if __name__ == "__main__":
    main()
