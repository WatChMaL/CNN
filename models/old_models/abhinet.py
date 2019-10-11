"""
vaenet.py

PyTorch implementation of a Convolutional Neural Network Variational Autoencoder ( VaeNet )

Author : Abhishek .
"""

# PyTorch Imports
import torch.nn as nn

# VaeNet class
class AbhiNet(nn.Module):
    
    # Initializer for the ConvNetVAE model
    def __init__(self, num_input_channels=19, num_latent_dimensions=32, num_classes=3, train=True):
        
        # Initialize the superclass
        super(AbhiNet, self).__init__()
        
        # Non-linear activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Encoder
        
        self.en_conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.en_bn1 = nn.BatchNorm2d(32, affine=True)
        self.en_maxpool1 = nn.MaxPool2d(kernel_size=(3,9), stride=(1,1), padding=(1,0))
        self.en_conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.en_bn2 = nn.BatchNorm2d(64, affine=True)
        self.en_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.en_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.en_fc1 = nn.Linear(1024, 256)
        self.en_fc2 = nn.Linear(256, 64)
        self.en_fc3 = nn.Linear(64, 16)
        self.en_fc4 = nn.Linear(16, num_classes)
        
        # Decoder
        
        self.de_fc1   = nn.Linear(num_latent_dimensions, 64)
        self.de_fc2   = nn.Linear(64, 256)
        self.de_fc3   = nn.Linear(256, 1024)
        self.de_conv1 = nn.ConvTranspose2d(1024, 128, kernel_size=(4,8), stride=(1,1), padding=(0,0))
        self.de_bn1   = nn.BatchNorm2d(128, affine=True)
        self.de_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=(5,9), stride=(1,1), padding=(0,0))
        self.de_bn2   = nn.BatchNorm2d(64, affine=True)
        self.de_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.de_bn3   = nn.BatchNorm2d(32, affine=True)
        self.de_conv4 = nn.ConvTranspose2d(32, num_input_channels, kernel_size=(3,9), stride=(1,1), padding=(1,0))
        
        # Boolean to determine the reparametrization mode
        self.training = train 
            
    # Method for the forward pass
    def forward(self, X):
        """
        mu, covar = self.encode(X)
        z = self.reparameterize(mu, covar)
        return self.decode(z), mu, covar"""
        return self.encode(X)
        
    
    # Method for the encoder layers
    def encode(self, X):
        x = self.en_maxpool1(self.relu(self.en_conv1(X)))
        x = self.en_maxpool2(self.relu(self.en_conv2(x)))
        x = self.relu(self.en_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.relu(self.en_fc3(x))
        return self.softmax(self.en_fc4(x))
        
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
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.de_bn1(self.relu(self.de_conv1(x)))
        x = self.de_bn2(self.relu(self.de_conv2(x)))
        x = self.de_bn3(self.relu(self.de_conv3(x)))
        x = self.de_conv4(x)
        return x