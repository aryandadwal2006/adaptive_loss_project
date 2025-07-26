"""
Advanced Checkpoint Management for Adaptive Loss Systems
Comprehensive checkpointing with versioning, compression, and recovery features
"""

import torch
import os
import json
import shutil
import time
import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pickle
import hashlib
from dataclasses import dataclass
from enum import Enum

class CheckpointType(Enum):
    """Types of checkpoints"""
    BEST = "best"
    LATEST = "latest"
    EPOCH = "epoch"
    MANUAL = "manual"
    EMERGENCY = "emergency"

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    timestamp: datetime.datetime
    epoch: int
    batch: int
    metrics: Dict[str, float]
    file_size: int
    hash_value: str
    experiment_name: str
    model_architecture: str
    notes: str = ""

class CheckpointManager:
    """
    Advanced checkpoint management system with versioning, compression, and recovery
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 max_checkpoints: int = 10, 
                 compression: bool = True,
                 auto_save_interval: int = 5):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            compression: Whether to compress checkpoints
            auto_save_interval: Auto-save interval in epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.auto_save_interval = auto_save_interval
        
        # Checkpoint registry
        self.registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        self.registry = self._load_registry()
        
        # Current session info
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance tracking
        self.save_times = []
        self.load_times = []
        
        print(f"Checkpoint Manager initialized: {checkpoint_dir}")
        print(f"Session ID: {self.session_id}")
    
    def _load_registry(self) -> Dict[str, CheckpointMetadata]:
        """Load checkpoint registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to CheckpointMetadata objects
                registry = {}
                for key, value in data.items():
                    metadata = CheckpointMetadata(
                        checkpoint_id=value['checkpoint_id'],
                        checkpoint_type=CheckpointType(value['checkpoint_type']),
                        timestamp=datetime.datetime.fromisoformat(value['timestamp']),
                        epoch=value['epoch'],
                        batch=value['batch'],
                        metrics=value['metrics'],
                        file_size=value['file_size'],
                        hash_value=value['hash_value'],
                        experiment_name=value['experiment_name'],
                        model_architecture=value['model_architecture'],
                        notes=value.get('notes', '')
                    )
                    registry[key] = metadata
                
                return registry
                
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {}
        
        return {}
    
    def _save_registry(self):
        """Save checkpoint registry to file"""
        # Convert CheckpointMetadata objects to dict
        data = {}
        for key, metadata in self.registry.items():
            data[key] = {
                'checkpoint_id': metadata.checkpoint_id,
                'checkpoint_type': metadata.checkpoint_type.value,
                'timestamp': metadata.timestamp.isoformat(),
                'epoch': metadata.epoch,
                'batch': metadata.batch,
                'metrics': metadata.metrics,
                'file_size': metadata.file_size,
                'hash_value': metadata.hash_value,
                'experiment_name': metadata.experiment_name,
                'model_architecture': metadata.model_architecture,
                'notes': metadata.notes
            }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_checkpoint_id(self, checkpoint_type: CheckpointType, 
                               epoch: int, batch: int = 0) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{checkpoint_type.value}_{epoch:04d}_{batch:04d}_{timestamp}"
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_monitor: Any,
                       meta_controller: Any,
                       epoch: int,
                       batch: int,
                       metrics: Dict[str, float],
                       checkpoint_type: CheckpointType = CheckpointType.LATEST,
                       experiment_name: str = "default",
                       notes: str = "") -> str:
        """
        Save a comprehensive checkpoint
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            loss_monitor: Loss monitor agent
            meta_controller: Meta-learning controller
            epoch: Current epoch
            batch: Current batch
            metrics: Current metrics
            checkpoint_type: Type of checkpoint
            experiment_name: Name of experiment
            notes: Additional notes
            
        Returns:
            Checkpoint ID
        """
        start_time = time.time()
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(checkpoint_type, epoch, batch)
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_monitor_state': self._get_loss_monitor_state(loss_monitor),
            'meta_controller_state': self._get_meta_controller_state(meta_controller),
            'epoch': epoch,
            'batch': batch,
            'metrics': metrics,
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'session_id': self.session_id,
            'pytorch_version': torch.__version__,
            'notes': notes
        }
        
        # Save checkpoint
        if self.compression:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pth.gz"
            torch.save(checkpoint_data, checkpoint_file, _use_new_zipfile_serialization=False)
        else:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pth"
            torch.save(checkpoint_data, checkpoint_file)
        
        # Calculate file hash and size
        file_hash = self._calculate_file_hash(checkpoint_file)
        file_size = checkpoint_file.stat().st_size
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            timestamp=datetime.datetime.now(),
            epoch=epoch,
            batch=batch,
            metrics=metrics,
            file_size=file_size,
            hash_value=file_hash,
            experiment_name=experiment_name,
            model_architecture=str(model.__class__.__name__),
            notes=notes
        )
        
        # Update registry
        self.registry[checkpoint_id] = metadata
        
        # Handle checkpoint limits
        self._cleanup_old_checkpoints(checkpoint_type)
        
        # Save registry
        self._save_registry()
        
        # Track performance
        save_time = time.time() - start_time
        self.save_times.append(save_time)
        
        print(f"Checkpoint saved: {checkpoint_id} ({file_size / 1024 / 1024:.1f} MB, {save_time:.2f}s)")
        
        return checkpoint_id
    
    def _get_loss_monitor_state(self, loss_monitor: Any) -> Dict:
        """Extract state from loss monitor"""
        try:
            if hasattr(loss_monitor, 'q_network') and hasattr(loss_monitor, 'target_network'):
                return {
                    'q_network_state': loss_monitor.q_network.state_dict(),
                    'target_network_state': loss_monitor.target_network.state_dict(),
                    'optimizer_state': loss_monitor.optimizer.state_dict(),
                    'epsilon': loss_monitor.epsilon,
                    'total_steps': getattr(loss_monitor, 'total_steps', 0),
                    'memory_size': len(getattr(loss_monitor, 'memory', [])),
                    'loss_history': getattr(loss_monitor, 'loss_history', [])
                }
        except Exception as e:
            print(f"Warning: Could not save loss monitor state: {e}")
        
        return {}
    
    def _get_meta_controller_state(self, meta_controller: Any) -> Dict:
        """Extract state from meta-controller"""
        try:
            if hasattr(meta_controller, 'net') and hasattr(meta_controller, 'opt'):
                return {
                    'network_state': meta_controller.net.state_dict(),
                    'optimizer_state': meta_controller.opt.state_dict(),
                    'history': getattr(meta_controller, 'history', [])
                }
        except Exception as e:
            print(f"Warning: Could not save meta-controller state: {e}")
        
        return {}
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint by ID
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Loaded checkpoint data
        """
        start_time = time.time()
        
        if checkpoint_id not in self.registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry")
        
        metadata = self.registry[checkpoint_id]
        
        # Determine file path
        if self.compression:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pth.gz"
        else:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pth"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Verify file integrity
        current_hash = self._calculate_file_hash(checkpoint_file)
        if current_hash != metadata.hash_value:
            raise ValueError(f"Checkpoint file corrupted: hash mismatch")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
        
        # Track performance
        load_time = time.time() - start_time
        self.load_times.append(load_time)
        
        print(f"Checkpoint loaded: {checkpoint_id} ({load_time:.2f}s)")
        
        return checkpoint_data
    
    def restore_training(self, checkpoint_id: str, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, 
                        loss_monitor: Any, meta_controller: Any) -> Dict[str, Any]:
        """
        Restore training from a checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint to restore from
            model: Model to restore
            optimizer: Optimizer to restore
            loss_monitor: Loss monitor to restore
            meta_controller: Meta-controller to restore
            
        Returns:
            Restored training info
        """
        checkpoint_data = self.load_checkpoint(checkpoint_id)
        
        # Restore model
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Restore optimizer
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Restore loss monitor
        self._restore_loss_monitor(loss_monitor, checkpoint_data.get('loss_monitor_state', {}))
        
        # Restore meta-controller
        self._restore_meta_controller(meta_controller, checkpoint_data.get('meta_controller_state', {}))
        
        training_info = {
            'epoch': checkpoint_data['epoch'],
            'batch': checkpoint_data['batch'],
            'metrics': checkpoint_data['metrics'],
            'experiment_name': checkpoint_data['experiment_name'],
            'session_id': checkpoint_data['session_id']
        }
        
        print(f"Training restored from checkpoint {checkpoint_id}")
        print(f"Resuming from epoch {training_info['epoch']}, batch {training_info['batch']}")
        
        return training_info
    
    def _restore_loss_monitor(self, loss_monitor: Any, state: Dict):
        """Restore loss monitor state"""
        try:
            if 'q_network_state' in state:
                loss_monitor.q_network.load_state_dict(state['q_network_state'])
            if 'target_network_state' in state:
                loss_monitor.target_network.load_state_dict(state['target_network_state'])
            if 'optimizer_state' in state:
                loss_monitor.optimizer.load_state_dict(state['optimizer_state'])
            if 'epsilon' in state:
                loss_monitor.epsilon = state['epsilon']
            if 'total_steps' in state:
                loss_monitor.total_steps = state['total_steps']
            if 'loss_history' in state:
                loss_monitor.loss_history = state['loss_history']
        except Exception as e:
            print(f"Warning: Could not restore loss monitor state: {e}")
    
    def _restore_meta_controller(self, meta_controller: Any, state: Dict):
        """Restore meta-controller state"""
        try:
            if 'network_state' in state:
                meta_controller.net.load_state_dict(state['network_state'])
            if 'optimizer_state' in state:
                meta_controller.opt.load_state_dict(state['optimizer_state'])
            if 'history' in state:
                meta_controller.history = state['history']
        except Exception as e:
            print(f"Warning: Could not restore meta-controller state: {e}")
    
    def _cleanup_old_checkpoints(self, checkpoint_type: CheckpointType):
        """Clean up old checkpoints to maintain limits"""
        # Get checkpoints of same type
        same_type_checkpoints = [
            (id, metadata) for id, metadata in self.registry.items()
            if metadata.checkpoint_type == checkpoint_type
        ]
        
        # Sort by timestamp (newest first)
        same_type_checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
        
        # Remove excess checkpoints
        if len(same_type_checkpoints) > self.max_checkpoints:
            for checkpoint_id, metadata in same_type_checkpoints[self.max_checkpoints:]:
                self._delete_checkpoint(checkpoint_id)
    
    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint and its files"""
        if checkpoint_id not in self.registry:
            return
        
        metadata = self.registry[checkpoint_id]
        
        # Delete files
        for extension in ['.pth', '.pth.gz']:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{extension}"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        
        # Remove from registry
        del self.registry[checkpoint_id]
        
        print(f"Deleted checkpoint: {checkpoint_id}")
    
    def list_checkpoints(self, checkpoint_type: Optional[CheckpointType] = None) -> List[CheckpointMetadata]:
        """List available checkpoints"""
        checkpoints = list(self.registry.values())
        
        if checkpoint_type:
            checkpoints = [cp for cp in checkpoints if cp.checkpoint_type == checkpoint_type]
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def get_best_checkpoint(self, metric_name: str, maximize: bool = True) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint based on a metric"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Filter checkpoints that have the metric
        valid_checkpoints = [cp for cp in checkpoints if metric_name in cp.metrics]
        
        if not valid_checkpoints:
            return None
        
        # Find best checkpoint
        best_checkpoint = max(valid_checkpoints, 
                            key=lambda x: x.metrics[metric_name] if maximize 
                            else -x.metrics[metric_name])
        
        return best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def export_checkpoint(self, checkpoint_id: str, export_path: str):
        """Export a checkpoint to a different location"""
        if checkpoint_id not in self.registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        metadata = self.registry[checkpoint_id]
        
        # Find source file
        source_file = None
        for extension in ['.pth', '.pth.gz']:
            potential_file = self.checkpoint_dir / f"{checkpoint_id}{extension}"
            if potential_file.exists():
                source_file = potential_file
                break
        
        if not source_file:
            raise FileNotFoundError(f"Checkpoint file not found for {checkpoint_id}")
        
        # Copy to export location
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, export_file)
        
        # Export metadata
        metadata_file = export_file.with_suffix('.json')
        metadata_dict = {
            'checkpoint_id': metadata.checkpoint_id,
            'checkpoint_type': metadata.checkpoint_type.value,
            'timestamp': metadata.timestamp.isoformat(),
            'epoch': metadata.epoch,
            'batch': metadata.batch,
            'metrics': metadata.metrics,
            'file_size': metadata.file_size,
            'hash_value': metadata.hash_value,
            'experiment_name': metadata.experiment_name,
            'model_architecture': metadata.model_architecture,
            'notes': metadata.notes
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Checkpoint exported to: {export_file}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics"""
        checkpoints = self.list_checkpoints()
        total_size = sum(cp.file_size for cp in checkpoints)
        
        type_counts = {}
        for cp in checkpoints:
            type_counts[cp.checkpoint_type.value] = type_counts.get(cp.checkpoint_type.value, 0) + 1
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size / (1024 * 1024),
            'checkpoint_types': type_counts,
            'avg_save_time': sum(self.save_times) / len(self.save_times) if self.save_times else 0,
            'avg_load_time': sum(self.load_times) / len(self.load_times) if self.load_times else 0,
            'oldest_checkpoint': min(checkpoints, key=lambda x: x.timestamp).timestamp.isoformat() if checkpoints else None,
            'newest_checkpoint': max(checkpoints, key=lambda x: x.timestamp).timestamp.isoformat() if checkpoints else None
        }
        
        return stats

if __name__ == "__main__":
    # Test the checkpoint manager
    from ..models import SimpleCNN
    from ..loss_monitor import LossMonitorAgent
    from ..meta_controller import MetaLearningController
    
    print("Testing Checkpoint Manager...")
    
    # Create test objects
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters())
    loss_monitor = LossMonitorAgent()
    meta_controller = MetaLearningController()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        max_checkpoints=5,
        compression=True
    )
    
    # Save a checkpoint
    checkpoint_id = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        loss_monitor=loss_monitor,
        meta_controller=meta_controller,
        epoch=10,
        batch=100,
        metrics={'loss': 0.5, 'accuracy': 0.85},
        checkpoint_type=CheckpointType.BEST,
        experiment_name="test_experiment",
        notes="Test checkpoint"
    )
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    # Get stats
    stats = checkpoint_manager.get_stats()
    print(f"Stats: {stats}")
    
    # Load checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id)
    print(f"Loaded checkpoint epoch: {loaded_data['epoch']}")
    
    # Restore training
    training_info = checkpoint_manager.restore_training(
        checkpoint_id, model, optimizer, loss_monitor, meta_controller
    )
    print(f"Training restored: {training_info}")
    
    print("Checkpoint Manager test completed successfully!")
