"""
Toy dataset generator for testing adaptive loss functions
Creates synthetic datasets optimized for CPU training
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from .config import DATASET_CONFIG, DEVICE

class ToyDatasetGenerator:
    """Generate synthetic datasets for different tasks"""
    
    def __init__(self, task_type='classification', **kwargs):
        self.task_type = task_type
        self.num_samples = kwargs.get('num_samples', DATASET_CONFIG['num_samples'])
        self.image_size = kwargs.get('image_size', DATASET_CONFIG['image_size'])
        self.num_classes = kwargs.get('num_classes', DATASET_CONFIG['num_classes'])
        
    def generate_classification_data(self):
        """Generate synthetic image classification dataset"""
        # Create base random images
        images = np.random.rand(self.num_samples, 1, self.image_size, self.image_size) * 0.3
        labels = np.random.randint(0, self.num_classes, self.num_samples)
        
        # Add distinctive patterns for each class
        for i in range(self.num_samples):
            label = labels[i]
            image = images[i, 0]
            
            if label == 0:  # Circles
                self._add_circle(image)
            elif label == 1:  # Squares
                self._add_square(image)
            elif label == 2:  # Triangles
                self._add_triangle(image)
            elif label == 3:  # Lines
                self._add_lines(image)
            else:  # Random noise (class 4)
                self._add_noise(image)
        
        # Convert to tensors
        images_tensor = torch.FloatTensor(images).to(DEVICE)
        labels_tensor = torch.LongTensor(labels).to(DEVICE)
        
        return images_tensor, labels_tensor
    
    def generate_regression_data(self):
        """Generate synthetic regression dataset"""
        # Generate input images
        inputs = np.random.rand(self.num_samples, 1, self.image_size, self.image_size)
        
        # Generate target images with simple transformations
        targets = np.zeros_like(inputs)
        
        for i in range(self.num_samples):
            # Apply simple transformations
            input_img = inputs[i, 0]
            target_img = targets[i, 0]
            
            # Simple blur effect
            target_img[:] = self._apply_blur(input_img)
        
        inputs_tensor = torch.FloatTensor(inputs).to(DEVICE)
        targets_tensor = torch.FloatTensor(targets).to(DEVICE)
        
        return inputs_tensor, targets_tensor
    
    def generate_vae_data(self):
        """Generate data for VAE training"""
        # For VAE, we use the same data as input and target
        images = np.random.rand(self.num_samples, self.image_size * self.image_size) * 0.8
        
        # Add some structure to make reconstruction meaningful
        for i in range(self.num_samples):
            if i % 3 == 0:
                # Add horizontal stripes
                images[i] = self._add_stripes(images[i].reshape(self.image_size, self.image_size)).flatten()
            elif i % 3 == 1:
                # Add vertical stripes
                images[i] = self._add_stripes(images[i].reshape(self.image_size, self.image_size), vertical=True).flatten()
            # else: keep random
        
        images_tensor = torch.FloatTensor(images).to(DEVICE)
        return images_tensor, images_tensor  # Same as input and target for VAE
    
    def _add_circle(self, image):
        """Add circle pattern to image"""
        center = self.image_size // 2
        radius = self.image_size // 4
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        image[mask] = 1.0
    
    def _add_square(self, image):
        """Add square pattern to image"""
        start = self.image_size // 4
        end = 3 * self.image_size // 4
        image[start:end, start:end] = 1.0
    
    def _add_triangle(self, image):
        """Add triangle pattern to image"""
        center = self.image_size // 2
        for i in range(self.image_size):
            for j in range(self.image_size):
                if (i >= center and 
                    j >= center - (i - center) and 
                    j <= center + (i - center)):
                    image[i, j] = 1.0
    
    def _add_lines(self, image):
        """Add line patterns to image"""
        # Vertical lines
        for i in range(0, self.image_size, 6):
            if i + 2 < self.image_size:
                image[:, i:i+2] = 1.0
    
    def _add_noise(self, image):
        """Add random noise"""
        noise = np.random.rand(*image.shape) * 0.5
        image += noise
        np.clip(image, 0, 1, out=image)
    
    def _add_stripes(self, image, vertical=False):
        """Add stripe patterns"""
        if vertical:
            for i in range(0, self.image_size, 4):
                if i + 1 < self.image_size:
                    image[:, i:i+2] = 1.0
        else:
            for i in range(0, self.image_size, 4):
                if i + 1 < self.image_size:
                    image[i:i+2, :] = 1.0
        return image
    
    def _apply_blur(self, image):
        """Apply simple blur effect"""
        # Simple 3x3 average filter
        blurred = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                blurred[i, j] = np.mean(image[i-1:i+2, j-1:j+2])
        return blurred
    
    def generate(self):
        """Generate dataset based on task type"""
        if self.task_type == 'classification':
            return self.generate_classification_data()
        elif self.task_type == 'regression':
            return self.generate_regression_data()
        elif self.task_type == 'vae':
            return self.generate_vae_data()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def visualize_samples(self, num_samples=5):
        """Visualize generated samples"""
        images, labels = self.generate()
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i in range(min(num_samples, len(images))):
            if images[i].dim() == 3:  # (C, H, W)
                img = images[i].squeeze().cpu().numpy()
            else:  # Flattened
                img = images[i].cpu().numpy().reshape(self.image_size, self.image_size)
            
            axes[i].imshow(img, cmap='gray')
            if self.task_type == 'classification':
                axes[i].set_title(f'Class: {labels[i].item()}')
            else:
                axes[i].set_title(f'Sample {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_data.png')
        plt.show()

class AdaptiveDataset(Dataset):
    """PyTorch Dataset wrapper for our generated data"""
    
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

def create_dataloaders(task_type='classification', batch_size=16, train_split=0.8):
    """Create train and validation dataloaders"""
    generator = ToyDatasetGenerator(task_type=task_type)
    images, targets = generator.generate()
    
    # Split into train and validation
    num_train = int(len(images) * train_split)
    
    train_images = images[:num_train]
    train_targets = targets[:num_train]
    val_images = images[num_train:]
    val_targets = targets[num_train:]
    
    # Create datasets
    train_dataset = AdaptiveDataset(train_images, train_targets)
    val_dataset = AdaptiveDataset(val_images, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset generator
    print("Testing dataset generation...")
    
    generator = ToyDatasetGenerator(task_type='classification')
    images, labels = generator.generate()
    
    print(f"Generated {len(images)} images with shape {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    # Visualize samples
    generator.visualize_samples()
