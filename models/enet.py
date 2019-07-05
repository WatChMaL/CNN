"""
enet.py

Block implementation of different autoencoder models i.e. encoders, decoders, bottlenecks etc.

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import randn, randn_like, tensor, zeros
from torch import device

# Global variables
valid_variants = ["AE", "VAE"]

# Enet class
class Enet(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels=38, num_latent_dims=64, variant="AE"):
        assert variant in valid_variants
        super(Enet, self).__init__()
        
        # Class attributes
        self.variant = variant
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Add the layer blocks
        self.encoder = Encoder(num_input_channels)
        
        # Add the desired bottleneck
        if variant is "AE":
            self.bottleneck = AEBottleneck()
        elif variant is "VAE":
            self.bottleneck = VAEBottleneck()
            
        self.decoder = Decoder(num_input_channels, None)
            
    # Forward
    def forward(self, X, mode):
        x = self.encoder(X)
        x = self.bottleneck(x)
        self.decoder.unflat_size = self.encoder.unflat_size
        return self.decoder(x)
        
        
# Encoder class
class Encoder(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels):
        super(Encoder, self).__init__()
        self.unflat_size = None
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Feature extraction
        self.en_conv1a  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature extraction
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
    # Forward
    def forward(self, X):
        x = self.en_conv1a(X)
        x = self.relu(self.en_conv1b(x))
        
        x = self.relu(self.en_conv2(x))
        
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        
        x = self.en_conv4(x)
        
        self.unflat_size = x.size()

        return x.view(-1, x.size(1)*x.size(2)*x.size(3))
    
# Decoder class
class Decoder(nn.Module):
    
    # Initialize
    def __init__(self, num_output_channels, unflat_size):
        super(Decoder, self).__init__()
        self.unflat_size = unflat_size
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Upsampling de-convolution
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        # Upsampling de-convolution
        self.de_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv1b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1a = nn.ConvTranspose2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward
    def forward(self, X):
        x = X.view(self.unflat_size)            
        
        x = self.relu(self.de_conv4(x))
        
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        
        x = self.relu(self.de_conv2(x))
        
        x = self.de_conv1b(x)
        x = self.relu(self.de_conv1a(x))

        return x
    
# AEBottleneck
class AEBottleneck(nn.Module):
    
    # Initialize
    def __init__(self):
        super(AEBottleneck, self).__init__()
        
    # Forward
    def forward(self, X):
        return X
    
# VAEBottleneck
class VAEBottleneck(nn.Module):
    
    # Initialize
    def __init__(self):
        super(VAEBottleneck, self).__init__()
        
    # Forward
    def forward(self, X):
        return X