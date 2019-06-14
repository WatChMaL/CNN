"""
vaenet.py

PyTorch implementation of a Convolutional Neural Network Variational Autoencoder ( VaeNet )

Author : Abhishek .
"""

# PyTorch Imports
import torch.nn as nn

# VaeNet class
class VaeNet(nn.Module):
    
    # Initializer for the ConvNetVAE model
    def __init__(self, num_input_channels=19, num_latent_dimensions=1024, train=True):
        
        # Initialize the superclass
        super(VaeNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.en_fc1 = nn.Linear(9 * 3 * 32, num_latent_dimensions)
        self.en_fc2 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        self.en_fc31 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        self.en_fc32 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        
        # Decoder
        self.de_fc1 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        self.de_fc2 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        self.de_fc3 = nn.Linear(num_latent_dimensions, num_latent_dimensions)
        self.de_conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=3, padding=0, bias=False)
        self.de_conv2 = nn.Conv2d(32, 32, kernel_size=(4,2), stride=(3,1), padding=(0,0), bias=False)
        self.de_conv3 = nn.Conv2d(32, num_input_channels, kernel_size=3, stride=1, padding=0, bias=False)
        
        # Boolean to determine the reparametrization mode
        self.training = train 
            
    # Method for the forward pass
    def forward(self, X):
        mu, covar = self.encode(X)
        z = self.reparameterize(mu, covar)
        return self.decode(z), mu, covar
    
    # Method for the encoder layers
    def encode(self, X):
        x = self.maxpool(self.relu(self.bn1(self.relu(self.conv1(X)))))
        x = self.maxpool(self.relu(self.bn2(self.relu(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        return self.en_fc31(x), self.en_fc32(x)
        
    # Method for the reparameterization
    def reparameterize(self, mu, covar):
        if self.training: 
            std = covar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    # Method for the decoder layers
    def decode(self, X):
        x = self.relu(self.de_fc1(X))
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc3(x))
        x = x.view(x.size(0), 1, 1, 1, x.size(1))
        x = self.relu(nn.functional.interpolate(x, size=(1, 32, 32), mode="nearest"))
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(4))
        x = self.relu(self.bn2(self.relu(self.de_conv1(x))))
        x = self.relu(self.bn2(self.relu(self.de_conv2(x))))
        x = x.view(x.size(0), x.size(1), 1, x.size(2), x.size(3))
        x = self.relu(nn.functional.interpolate(x, size=(1, 8, 20), mode="nearest"))
        x = self.relu(nn.functional.interpolate(x, size=(1, 12, 30), mode="nearest"))
        x = self.relu(nn.functional.interpolate(x, size=(1, 18, 42), mode="nearest"))
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(4))
        x = self.relu(self.de_conv3(x))
        return x