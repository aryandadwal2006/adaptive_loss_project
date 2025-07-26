"""
Performance optimization utilities: profiling, memory checking, parallelism helpers.
"""

import psutil
import time
import torch
from .logging_utils import ExperimentLogger

logger = ExperimentLogger("performance_utils")

def profile_function(fn, *args, **kwargs):
    """
    Profile execution time and memory usage of a function.
    Returns result, metrics dict.
    """
    start_mem = psutil.Process().memory_info().rss
    start_time = time.time()
    result = fn(*args, **kwargs)
    duration = time.time() - start_time
    end_mem = psutil.Process().memory_info().rss
    used_mb = (end_mem - start_mem) / (1024**2)
    metrics = {"duration_s": duration, "mem_delta_mb": used_mb}
    logger.log_info(f"Profiled {fn.__name__}: {metrics}")
    return result, metrics

def optimize_dataloader(loader):
    """
    Attempt to set pin_memory and prefetch settings where possible.
    """
    try:
        loader.pin_memory = True
        logger.log_info("Enabled pin_memory on DataLoader")
    except:
        pass
    return loader

def set_parallelism(num_threads: int):
    """
    Configure torch and numpy threading.
    """
    torch.set_num_threads(num_threads)
    logger.log_info(f"Set torch threads to {num_threads}")
