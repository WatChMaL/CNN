"""
    convtwovaenet.py
    
    PyTorch implementation of the VAE network corresponding to the convtwoNet classifier
    
    The model is supposed to be the Convolutional Only alternative to KvaeNet
"""

# PyTorch imports
import torch.nn as nn
import torch.tensor as tensor
from torch import zeros

class ConvtwovaeNet(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=19, num_latent_dims=64, num_classes=3, train=True):
        
        # Initialize the superclass
        super(ConvtwovaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Variables for model architecture
        self.unflat_size = None
        
        #--
        self.indices = None
        self.max_input_size = None
        #--
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Feature extraction convolutions
        self.en_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Feature extraction convolutions
        self.en_conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.en_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Flattening
        
        # Fully connected layers
        self.en_fc1 = nn.Linear(20480, 512)
        
        # Classifier output layer
        self.en_fc4 = nn.Linear(512, num_classes)
        
        # Encoder output layers
        self.en_fc41 = nn.Linear(512, num_latent_dims)
        self.en_fc42 = nn.Linear(512, num_latent_dims)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # Fully connected layers
        self.de_fc4 = nn.Linear(num_latent_dims, 512)
        self.de_fc1 = nn.Linear(512, 20480)
        
        # Unflattening
        
        # Feature extraction deconvolution
        self.de_conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.de_conv5 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Feature extraction convolution
        self.de_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
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
        
    # Encoder application
    
    def encode(self, X):
        
        # Feature extraction convolutions
        x = self.relu(self.en_conv1(X))
        x = self.en_conv2(x)
        x = self.en_conv3(x)
        x = self.relu(self.en_conv4(x))

        # Feature extraction convolutions
        x = self.relu(self.en_conv5(x))
        x = self.relu(self.en_conv6(x))
        
        # Save the unflattened size
        self.unflat_size = x.size()
        
        # Flattening
        x = x.view(-1, 20480)
        
        # Fully connected layers
        x = self.relu(self.en_fc1(x))
        
        # Encoder output layer
        return self.en_fc41(x), self.en_fc42(x)
        
        
    # Reparameterization trick
    
    def reparameterize(self, mu, logvar, shots=1024):
        
        # Calculate the square-root covariance matrix
        std = logvar.mul(0.5).exp_()
            
        # Initialize a torch tensor holding the sum with the same size as std
        eps_sum = std.new(std.size())
        
        # Compute and reparameterize multiple samples
        for shot in range(shots):
            
            # Sample a vector eps from the normal distribution
            eps = std.new(std.size()).normal_()
            
            # Add in-place to the 
            eps_sum.add_(eps.mul_(std).add_(mu))
            
        # Return the average of the sampled and reparameterized latent vectors
        denom = tensor((1.0/shots))
        
        return eps_sum.mul_(denom)
    
    # Decoder application
    
    def decode(self, X):
        
        # Fully connected layers
        x = self.de_fc4(X)
        x = self.relu(self.de_fc1(x))

        # Un-flattening
        x = x.view(self.unflat_size)
        
        # Feature extraction deconvolutions
        x = self.relu(self.de_conv6(x))
        x = self.relu(self.de_conv5(x))
        
        # Feature extraction convolutions
        x = self.relu(self.de_conv4(x))
        x = self.de_conv3(x)
        x = self.de_conv2(x)
        x = self.relu(self.de_conv1(x))
        
        return x