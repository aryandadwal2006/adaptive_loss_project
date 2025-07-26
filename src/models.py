"""
Neural network models optimized for CPU training
Lightweight architectures suitable for limited hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DEVICE, DATASET_CONFIG

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for classification tasks
    Optimized for CPU training with minimal parameters
    """
    
    def __init__(self, input_channels=1, num_classes=5, hidden_dim=64):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Third convolutional block
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to 4x4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class SimpleVAE(nn.Module):
    """
    Lightweight Variational Autoencoder
    Optimized for CPU training with minimal parameters
    """
    
    def __init__(self, input_dim=1024, hidden_dim=128, latent_dim=20):
        super(SimpleVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Latent space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        # Flatten input if needed
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Reshape back to original if needed
        if len(original_shape) > 2:
            recon = recon.view(original_shape)
        
        return recon, mu, logvar
    
    def sample(self, num_samples=10):
        """Generate samples from the model"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(DEVICE)
            samples = self.decode(z)
            return samples
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class SimpleGenerator(nn.Module):
    """
    Simple generator for GAN-style training
    Lightweight design for CPU efficiency
    """
    
    def __init__(self, latent_dim=100, output_channels=1, output_size=32):
        super(SimpleGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Calculate initial feature map size
        self.init_size = output_size // 4  # 8x8 for 32x32 output
        
        # Dense layer to project latent vector
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        
        # Transposed convolutions for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, output_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def forward(self, z):
        # Project and reshape
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        return img
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

class SimpleDiscriminator(nn.Module):
    """
    Simple discriminator for GAN-style training
    Lightweight design for CPU efficiency
    """
    
    def __init__(self, input_channels=1, input_size=32):
        super(SimpleDiscriminator, self).__init__()
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second block
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third block
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final convolution
            nn.Conv2d(128, 1, 4, 1, 0),  # 4x4 -> 1x1
            nn.Flatten(),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def forward(self, img):
        validity = self.conv_blocks(img)
        return validity
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction and KL divergence
    """
    # Flatten inputs for loss computation
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
        recon_x = recon_x.view(recon_x.size(0), -1)
    
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_shape):
    """Print a summary of the model architecture"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create dummy input
    if len(input_shape) == 3:  # (C, H, W)
        dummy_input = torch.randn(1, *input_shape)
    else:  # (features,)
        dummy_input = torch.randn(1, input_shape[0])
    
    # Forward pass to get output shape
    with torch.no_grad():
        if isinstance(model, SimpleVAE):
            output, _, _ = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        else:
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...")
    
    # Test CNN
    print("\n" + "="*50)
    cnn = SimpleCNN(input_channels=1, num_classes=5)
    model_summary(cnn, (1, 32, 32))
    
    # Test VAE
    print("\n" + "="*50)
    vae = SimpleVAE(input_dim=1024, hidden_dim=128, latent_dim=20)
    model_summary(vae, (1024,))
    
    # Test Generator
    print("\n" + "="*50)
    generator = SimpleGenerator(latent_dim=100, output_channels=1, output_size=32)
    model_summary(generator, (100,))
    
    # Test Discriminator
    print("\n" + "="*50)
    discriminator = SimpleDiscriminator(input_channels=1, input_size=32)
    model_summary(discriminator, (1, 32, 32))
