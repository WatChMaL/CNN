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
variant_dict = {0:"AE", 1:"VAE"}

# Enet class
class ENet(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels=38, num_latent_dims=64, variant_key=0):
        assert variant_key in variant_dict.keys()
        super(ENet, self).__init__()
        
        # Class attributes
        self.variant = variant_dict[variant_key]
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Add the layer blocks
        self.encoder = Encoder(num_input_channels)
        
        # Set require_grad = False for encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Add the desired bottleneck
        if self.variant is "AE":
            self.bottleneck = AEBottleneck()
        elif self.variant is "VAE":
            self.bottleneck = VAEBottleneck()
            
        self.decoder = Decoder(num_input_channels, None)
        
        # Set require_grad = False for decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    # Forward
    def forward(self, X, mode):
        x = self.encoder(X)
        
        if self.variant is "AE":
            x = self.bottleneck(x)
        elif self.variant is "VAE":
            z, mu, logvar = self.bottleneck(x)
            
        self.decoder.unflat_size = self.encoder.unflat_size
        
        if self.variant is "AE":
            return self.decoder(x)
        elif self.variant is "VAE":
            return self.decoder(x), z, mu, logvar
        
        
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
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        
        return x
    
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
        self.flat_size = None
        
    # Forward
    def forward(self, X):
        return X
    
# VAEBottleneck
class VAEBottleneck(nn.Module):
    
    # Initialize
    def __init__(self):
        super(VAEBottleneck, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Linear layers
        self.en_fc1 = nn.Linear(5120, 5120)
        self.en_fc2 = nn.Linear(5120, 5120)
        
        # Initialize the weights and biases of the layers
        nn.init.eye_(self.en_fc1.weight)
        nn.init.zeros_(self.en_fc1.bias)
        
        nn.init.zeros_(self.en_fc2.weight)
        nn.init.constant_(self.en_fc2.bias, 1e-3)
        
    # Forward
    def forward(self, X):
        
        mu, logvar = self.en_fc1(X), self.en_fc2(X)
        
        # Reparameterization trick
        std = logvar.mul(0.5).exp()
        eps = std.new(std.size()).normal_()
        
        return eps.mul(std).add(mu), mu, logvar