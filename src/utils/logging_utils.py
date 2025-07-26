"""
Advanced Logging Utilities for Adaptive Loss Experiments
Comprehensive logging system with multiple output formats and real-time monitoring
"""

import logging
import json
import csv
import time
import datetime
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import threading
from collections import defaultdict, deque
import sys

class ExperimentLogger:
    """
    Advanced experiment logger with multiple output formats and real-time monitoring
    Supports console, file, JSON, and CSV outputs with configurable verbosity
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "./logs", 
                 console_level: int = logging.INFO, file_level: int = logging.DEBUG):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        
        # Initialize logging system
        self.logger = self._setup_logger(console_level, file_level)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.batch_metrics = defaultdict(deque)
        self.epoch_metrics = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 10  # seconds
        
        # File handlers
        self.csv_files = {}
        self.json_log_file = None
        self._setup_file_handlers()
        
        # Performance tracking
        self.timing_data = {}
        
        self.log_info(f"Experiment Logger initialized for: {experiment_name}")
        self.log_info(f"Session ID: {self.session_id}")
        self.log_info(f"Output directory: {self.output_dir}")
    
    def _setup_logger(self, console_level: int, file_level: int) -> logging.Logger:
        """Setup the main logger with console and file handlers"""
        logger = logging.getLogger(f"experiment_{self.experiment_name}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"{self.experiment_name}_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_file_handlers(self):
        """Setup CSV and JSON file handlers"""
        # JSON log file for structured data
        json_log_path = self.output_dir / f"{self.experiment_name}_{self.session_id}_structured.json"
        self.json_log_file = open(json_log_path, 'w')
        self.json_log_file.write('[\n')  # Start JSON array
        
        # CSV files for different metric types
        csv_files_config = {
            'training': ['epoch', 'batch', 'loss', 'accuracy', 'learning_rate', 'gradient_norm'],
            'validation': ['epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1_score'],
            'adaptive': ['epoch', 'batch', 'action', 'reward', 'epsilon', 'main_weight', 'aux_weight'],
            'system': ['timestamp', 'cpu_usage', 'memory_usage', 'gpu_usage', 'temperature']
        }
        
        for file_type, headers in csv_files_config.items():
            csv_path = self.output_dir / f"{self.experiment_name}_{self.session_id}_{file_type}.csv"
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)
            self.csv_files[file_type] = {'file': csv_file, 'writer': csv_writer}
    
    def log_info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message with optional structured data"""
        self.logger.info(message)
        if extra_data:
            self._log_structured_data('INFO', message, extra_data)
    
    def log_warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message with optional structured data"""
        self.logger.warning(message)
        if extra_data:
            self._log_structured_data('WARNING', message, extra_data)
    
    def log_error(self, message: str, extra_data: Optional[Dict] = None):
        """Log error message with optional structured data"""
        self.logger.error(message)
        if extra_data:
            self._log_structured_data('ERROR', message, extra_data)
    
    def _log_structured_data(self, level: str, message: str, data: Dict):
        """Log structured data to JSON file"""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'level': level,
            'message': message,
            'data': data,
            'session_id': self.session_id
        }
        
        json.dump(log_entry, self.json_log_file, indent=2)
        self.json_log_file.write(',\n')
        self.json_log_file.flush()
    
    def log_batch(self, epoch: int, batch: int, loss: float, 
                  accuracy: float, gradient_norm: float):
        """Log batch-level training metrics"""
        # Store in memory for real-time monitoring
        self.batch_metrics['loss'].append(loss)
        self.batch_metrics['accuracy'].append(accuracy)
        self.batch_metrics['gradient_norm'].append(gradient_norm)
        
        # Keep only last 100 batch metrics for memory efficiency
        if len(self.batch_metrics['loss']) > 100:
            self.batch_metrics['loss'].popleft()
            self.batch_metrics['accuracy'].popleft()
            self.batch_metrics['gradient_norm'].popleft()
        
        # Log to CSV
        if 'training' in self.csv_files:
            self.csv_files['training']['writer'].writerow([
                epoch, batch, loss, accuracy, 0.001, gradient_norm  # learning_rate hardcoded for now
            ])
            self.csv_files['training']['file'].flush()
        
        # Log structured data
        self._log_structured_data('BATCH', f"Batch {epoch}.{batch}", {
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'accuracy': accuracy,
            'gradient_norm': gradient_norm
        })
        
        # Console log every 50 batches
        if batch % 50 == 0:
            self.log_info(f"Epoch {epoch}, Batch {batch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch-level metrics"""
        # Store in memory
        self.epoch_metrics['train_loss'].append(train_metrics.get('loss', 0))
        self.epoch_metrics['val_loss'].append(val_metrics.get('loss', 0))
        self.epoch_metrics['train_acc'].append(train_metrics.get('accuracy', 0))
        self.epoch_metrics['val_acc'].append(val_metrics.get('accuracy', 0))
        
        # Log to CSV
        if 'validation' in self.csv_files:
            self.csv_files['validation']['writer'].writerow([
                epoch, val_metrics.get('loss', 0), val_metrics.get('accuracy', 0),
                val_metrics.get('precision', 0), val_metrics.get('recall', 0), 
                val_metrics.get('f1_score', 0)
            ])
            self.csv_files['validation']['file'].flush()
        
        # Log structured data
        self._log_structured_data('EPOCH', f"Epoch {epoch} completed", {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
        # Console log
        self.log_info(
            f"Epoch {epoch}: Train Loss={train_metrics.get('loss', 0):.4f}, "
            f"Val Loss={val_metrics.get('loss', 0):.4f}, "
            f"Val Acc={val_metrics.get('accuracy', 0):.4f}"
        )
    
    def log_adaptive_metrics(self, epoch: int, batch: int, action: int, 
                           reward: float, epsilon: float, main_weight: float, 
                           aux_weight: float):
        """Log adaptive loss specific metrics"""
        # Store in memory
        self.metrics['actions'].append(action)
        self.metrics['rewards'].append(reward)
        self.metrics['main_weights'].append(main_weight)
        self.metrics['aux_weights'].append(aux_weight)
        
        # Log to CSV
        if 'adaptive' in self.csv_files:
            self.csv_files['adaptive']['writer'].writerow([
                epoch, batch, action, reward, epsilon, main_weight, aux_weight
            ])
            self.csv_files['adaptive']['file'].flush()
        
        # Log structured data
        self._log_structured_data('ADAPTIVE', f"Adaptive metrics for {epoch}.{batch}", {
            'epoch': epoch,
            'batch': batch,
            'action': action,
            'reward': reward,
            'epsilon': epsilon,
            'main_weight': main_weight,
            'aux_weight': aux_weight
        })
    
    def start_monitoring(self):
        """Start real-time monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.log_info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring thread"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1)
            self.log_info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Log system metrics
                self._log_system_metrics()
                
                # Log training progress summary
                self._log_progress_summary()
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.log_error(f"Monitoring loop error: {e}")
    
    def _log_system_metrics(self):
        """Log system performance metrics"""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU usage (if available)
            gpu_usage = 0
            temperature = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    temperature = gpus[0].temperature
            except:
                pass
            
            # Log to CSV
            if 'system' in self.csv_files:
                self.csv_files['system']['writer'].writerow([
                    datetime.datetime.now().isoformat(),
                    cpu_usage, memory.percent, gpu_usage, temperature
                ])
                self.csv_files['system']['file'].flush()
        
        except ImportError:
            pass  # psutil not available
    
    def _log_progress_summary(self):
        """Log training progress summary"""
        if self.batch_metrics['loss']:
            avg_loss = sum(list(self.batch_metrics['loss'])) / len(self.batch_metrics['loss'])
            avg_acc = sum(list(self.batch_metrics['accuracy'])) / len(self.batch_metrics['accuracy'])
            
            self.log_info(f"Recent progress: Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.4f}")
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        summary = {
            'session_id': self.session_id,
            'experiment_name': self.experiment_name,
            'duration': time.time() - self.start_time,
            'total_epochs': len(self.epoch_metrics['train_loss']),
            'total_batches': len(self.metrics.get('actions', [])),
            'current_metrics': {
                'latest_train_loss': self.epoch_metrics['train_loss'][-1] if self.epoch_metrics['train_loss'] else 0,
                'latest_val_loss': self.epoch_metrics['val_loss'][-1] if self.epoch_metrics['val_loss'] else 0,
                'latest_train_acc': self.epoch_metrics['train_acc'][-1] if self.epoch_metrics['train_acc'] else 0,
                'latest_val_acc': self.epoch_metrics['val_acc'][-1] if self.epoch_metrics['val_acc'] else 0,
            },
            'adaptive_metrics': {
                'avg_reward': sum(self.metrics.get('rewards', [0])) / max(len(self.metrics.get('rewards', [1])), 1),
                'action_distribution': self._get_action_distribution(),
                'avg_main_weight': sum(self.metrics.get('main_weights', [0])) / max(len(self.metrics.get('main_weights', [1])), 1),
                'avg_aux_weight': sum(self.metrics.get('aux_weights', [0])) / max(len(self.metrics.get('aux_weights', [1])), 1),
            }
        }
        
        return summary
    
    def _get_action_distribution(self) -> Dict:
        """Get distribution of actions taken"""
        actions = self.metrics.get('actions', [])
        if not actions:
            return {}
        
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
        
        total_actions = len(actions)
        return {k: v / total_actions for k, v in action_counts.items()}
    
    def save_final_report(self):
        """Save final experiment report"""
        summary = self.get_metrics_summary()
        
        # Save summary as JSON
        summary_path = self.output_dir / f"{self.experiment_name}_{self.session_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed metrics
        detailed_path = self.output_dir / f"{self.experiment_name}_{self.session_id}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump({
                'epoch_metrics': dict(self.epoch_metrics),
                'batch_metrics': {k: list(v) for k, v in self.batch_metrics.items()},
                'all_metrics': dict(self.metrics)
            }, f, indent=2)
        
        self.log_info(f"Final report saved to {summary_path}")
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        self.stop_monitoring()
        
        # Close JSON file
        if self.json_log_file:
            self.json_log_file.write('\n]')  # Close JSON array
            self.json_log_file.close()
        
        # Close CSV files
        for file_info in self.csv_files.values():
            file_info['file'].close()

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: ExperimentLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.log_info(f"{self.operation_name} completed in {duration:.3f}s")
        
        # Store timing data
        if not hasattr(self.logger, 'timing_data'):
            self.logger.timing_data = {}
        self.logger.timing_data[self.operation_name] = duration

if __name__ == "__main__":
    # Test the logger
    logger = ExperimentLogger("test_experiment")
    
    # Test batch logging
    for epoch in range(2):
        for batch in range(5):
            logger.log_batch(epoch, batch, 0.5 - batch * 0.1, 0.8 + batch * 0.02, 0.5)
        
        # Test epoch logging
        logger.log_epoch(epoch, 
                        {'loss': 0.3, 'accuracy': 0.85}, 
                        {'loss': 0.35, 'accuracy': 0.82})
    
    # Test adaptive metrics
    logger.log_adaptive_metrics(0, 0, 1, 0.5, 0.9, 1.0, 0.2)
    
    # Test timing
    with PerformanceTimer(logger, "test_operation"):
        time.sleep(0.1)
    
    # Get summary
    summary = logger.get_metrics_summary()
    print(f"Summary: {summary}")
    
    # Save final report
    logger.save_final_report()
    
    print("Logger test completed successfully!")
