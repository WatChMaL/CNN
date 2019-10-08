"""
elenet.py

PyTorch implementation of LeNet-style architecture to be used as an encoder and decoder
in the variational inference models.
"""

import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels, num_latent_dims):
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
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(5120, 1024)
        self.en_fc2 = nn.Linear(1024, 512)
        self.en_fc3 = nn.Linear(512, num_latent_dims)
        
    # Forward
    def forward(self, X):
        x = self.relu(self.en_conv1a(X))
        x = self.relu(self.en_conv1b(x))
        x = self.relu(self.en_conv2(x))
        x = self.relu(self.en_conv3a(x))
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_conv4(x))
        
        self.unflat_size = x.size()
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        
        return self.relu(self.en_fc3(x))
    
# Decoder class
class Decoder(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels, num_latent_dims):
        super(Decoder, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.de_fc3 = nn.Linear(num_latent_dims, 512)
        self.de_fc2 = nn.Linear(512, 1024)
        self.de_fc1 = nn.Linear(1024, 5120)
        
        # Upsampling de-convolution
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        # Upsampling de-convolution
        self.de_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv1b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1a = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward
    def forward(self, X, unflat_size):
        
        x = self.relu(self.de_fc3(X))
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc1(x))
        
        x = x.view(unflat_size) if unflat_size is not None else x.view(-1, 128, 4, 10)
        
        x = self.relu(self.de_conv4(x))
        x = self.relu(self.de_conv3b(x))
        x = self.relu(self.de_conv3a(x))
        x = self.relu(self.de_conv2(x))
        x = self.relu(self.de_conv1b(x))
        x = self.relu(self.de_conv1a(x))

        return x
    
#-------------------------------------------------
# Initializer for the models with various depths
#-------------------------------------------------

def elenet18(**kwargs):
    """Constructs a LeNet style model.
    """
    return Encoder(**kwargs), Decoder(**kwargs)