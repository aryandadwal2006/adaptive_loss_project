# Experiment Plan

## Objective
Validate the real-time adaptive loss system on a simple CNN classification task.

## Dataset
- Synthetic 32×32 grayscale images (1,000 samples, 5 classes)

## Model
- SimpleCNN (3 conv blocks + classifier)

## Losses
- Base: CrossEntropy
- Auxiliary: MSE

## Procedure
1. Generate train/val split (80/20).
2. Train with adaptive loss for 50 epochs.
3. Log loss, reward, gradient norms every batch.
4. Validate accuracy every 10 epochs.
5. Compare to static baseline.

## Metrics
- Final validation accuracy
- Loss convergence speed
- Reward trajectory
- Gradient stability
