# File: src/trainer/train_phase1.py

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
    parser = argparse.ArgumentParser(description="Phase 1: Baseline vs Adaptive training")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training and validation")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive loss (RL + meta-learning)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs/phase1",
                        help="TensorBoard log directory")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, executor=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        print(f"[DEBUG] loader batch sizes → x: {x.size()}, y: {y.size()}")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        print(f"[DEBUG] outputs shape → {outputs.size()}")

        if executor:
            # 1) gather state
            state = executor.monitor.get_state(
                executor.loss_history,
                executor.grad_stats,
                {"iteration": batch_idx, "epoch": epoch}
            )
            # 2) select action and map to params
            action = executor.monitor.act(state)
            params = executor.monitor.action_to_params(action)
            # 3) compute adaptive loss
            loss = executor.compute(outputs, y, params)
        else:
            loss = criterion(outputs, y)

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

def validate(model, loader, criterion, device, epoch, writer, executor=None):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            print(f"[DEBUG] loader batch sizes → x: {x.size()}, y: {y.size()}")
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            print(f"[DEBUG] outputs shape → {outputs.size()}")

            if executor:
                state = executor.monitor.get_state(
                    executor.loss_history,
                    executor.grad_stats,
                    {"iteration": batch_idx, "epoch": epoch}
                )
                action = executor.monitor.act(state)
                params = executor.monitor.action_to_params(action)
                loss = executor.compute(outputs, y, params)
            else:
                loss = criterion(outputs, y)

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
    logger = ExperimentLogger("phase1", args.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=args.batch_size)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    executor = None
    if args.adaptive:
        monitor = LossMonitorAgent()
        controller = MetaLearningController()
        # Executor uses the base loss function; monitor/controller handle adaptation
        executor = DynamicLossExecutor(criterion)
        executor.monitor = monitor
        executor.controller = controller

    best_val_acc, best_epoch = 0.0, 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, executor
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer, executor
        )

        logger.log_info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}; "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), os.path.join(args.logdir, "best_model.pth"))

    total_time = time.time() - start_time
    logger.log_info(f"Completed in {total_time:.1f}s. Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}.")

    # Save best metrics for experiment runner
    with open(os.path.join(args.logdir, "best_metrics.txt"), "w") as f:
        f.write(f"{best_epoch},{best_val_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
