"""
cnaenet.py

PyTorch implementation of a VAE impelemented using pure convolutional neural networks

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn
import torch.tensor as tensor
from torch import zeros
from torch import randn_like
from torch import randn
from torch import device

# ConvNet class
class CnaeNet(nn.Module):
    
    """ V1 VAE
    # Initializer
    
    def __init__(self, num_input_channels=19, num_classes=3, num_latent_dims=64, train=True):
        
        # Initialize the superclass
        super(CnaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # User-defined parameters
        self.num_latent_dims = num_latent_dims
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions
        self.en_conv1  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        self.en_fc1    = nn.Linear(5120, 5120)
        self.en_fc2    = nn.Linear(5120, 5120)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        self.de_fc     = nn.Linear(5120, 5120)
        
        # De-convolutions
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2b = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.de_conv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1 = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward pass
    def forward(self, X, mode="train"):
        if mode == "train" or mode == "validate":
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar, shots=1)

            # Return the output image, mean and covariance matrix
            return z, self.decode(z), mu, logvar
        
        elif mode == "generate":
            
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar)

            return z, mu, logvar
        
        elif mode == "sample":
            return self.sample()
        
    # Encoder
    
    def encode(self, X):
        
        # Convolutions
        x = self.relu(self.en_conv1(X))
        x = self.en_conv2a(x)
        x = self.relu(self.en_conv2b(x))
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_conv4(x))
        
        # Save the size of the input
        self.unflat_size = x.size()

        x = x.view(-1, 5120)
        
        return self.en_fc1(x), self.en_fc2(x)
    
    # Decoder
    
    def decode(self, X):
        
        x = self.de_fc(X)

        # Unflattening
        x = x.view(self.unflat_size)            
        
        # Deconvolutions
        x = self.relu(self.de_conv4(x))
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        x = self.relu(self.de_conv2b(x))
        x = self.de_conv2a(x)
        x = self.relu(self.de_conv1(x))

        return x
    
    # Reparameterization
    
    def reparameterize(self, mu, logvar, shots=1):
        
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul_(std).add_(mu)
"""

    # Initializer
    
    def __init__(self, num_input_channels=19, num_classes=3, num_latent_dims=64, train=True):
        
        # Initialize the superclass
        super(CnaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # User-defined parameters
        self.num_latent_dims = num_latent_dims
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions
        self.en_conv1  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        self.en_fc1    = nn.Linear(5120, 5120)
        self.en_fc2    = nn.Linear(5120, 5120)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        self.de_fc     = nn.Linear(5120, 5120)
        
        # De-convolutions
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2b = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.de_conv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1 = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward pass
    def forward(self, X, mode="train"):
        if mode == "train" or mode == "validate":
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar, shots=1)

            # Return the output image, mean and covariance matrix
            return z, self.decode(z), mu, logvar
        
        elif mode == "generate":
            
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar)

            return z, mu, logvar
        
        elif mode == "sample":
            return self.sample()
        
    # Encoder
    
    def encode(self, X):
        
        # Convolutions
        x = self.relu(self.en_conv1(X))
        x = self.en_conv2a(x)
        x = self.relu(self.en_conv2b(x))
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_conv4(x))
        
        # Save the size of the input
        self.unflat_size = x.size()

        x = x.view(-1, 5120)
        
        # Generate the random log variance vector
        eps = 0.001
        logvar = x.new_full(x.size(), eps).log_()
        logvar_noise = x.new(x.size()).normal_(std=eps)
        
        return x, logvar.add_(logvar_noise)
    
    # Decoder
    
    def decode(self, X):

        # Unflattening
        x = X.view(self.unflat_size)            
        
        # Deconvolutions
        x = self.relu(self.de_conv4(x))
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        x = self.relu(self.de_conv2b(x))
        x = self.de_conv2a(x)
        x = self.relu(self.de_conv1(x))

        return x
    
    # Reparameterization
    
    def reparameterize(self, mu, logvar, shots=1):
        
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul_(std).add_(mu)
        
"""
    def __init__(self, num_input_channels=19, num_classes=3, num_latent_dims=64, train=True):
        
        # Initialize the superclass
        super(CnaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # User-defined parameters
        self.num_latent_dims = num_latent_dims
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions
        self.en_conv1  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # De-convolutions
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2b = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.de_conv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1 = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward pass
    def forward(self, X, mode="train"):
        return self.decode(self.encode(X))
        
    # Encoder
    
    def encode(self, X):
        
        # Convolutions
        x = self.relu(self.en_conv1(X))
        x = self.en_conv2a(x)
        x = self.relu(self.en_conv2b(x))
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_conv4(x))
        
        # Save the size of the input
        self.unflat_size = x.size()

        return x.view(-1, 5120)
    
    # Decoder
    
    def decode(self, X):

        # Unflattening
        x = X.view(self.unflat_size)            
        
        # Deconvolutions
        x = self.relu(self.de_conv4(x))
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        x = self.relu(self.de_conv2b(x))
        x = self.de_conv2a(x)
        x = self.relu(self.de_conv1(x))

        return x
"""