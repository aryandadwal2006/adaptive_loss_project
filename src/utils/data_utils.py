"""
Utility functions for data handling, preprocessing, and augmentations.
"""

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
from .logging_utils import ExperimentLogger

logger = ExperimentLogger("data_utils")

def get_transforms(task_type: str, image_size: int) -> T.Compose:
    """
    Return torchvision transforms based on task type.
    """
    if task_type == 'classification':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    elif task_type == 'generation':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
    else:
        return T.Compose([T.ToTensor()])

def wrap_dataloader(dataset: Dataset,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    """
    Wrap a dataset into DataLoader with common defaults.
    """
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=False)
    logger.log_info(f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}")
    return loader

def split_dataset(images: torch.Tensor,
                  targets: torch.Tensor,
                  train_split: float = 0.8
                  ) -> Tuple[Dataset, Dataset]:
    """
    Split tensors into train & val subsets.
    """
    n = len(images)
    idx = int(n * train_split)
    train_ds = torch.utils.data.TensorDataset(images[:idx], targets[:idx])
    val_ds   = torch.utils.data.TensorDataset(images[idx:], targets[idx:])
    logger.log_info(f"Split dataset: train={idx}, val={n-idx}")
    return train_ds, val_ds
