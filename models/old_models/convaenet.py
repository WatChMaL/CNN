"""
convaenet.py

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
class ConvaeNet(nn.Module):
    
    # Initializer
    def __init__():
            
        # Initialize the superclass
        super(ConvaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # User-defined parameters
        self.num_latent_dims = num_latent_dims
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions
        self.en_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=0)
        self.en_maxconv1  = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_maxconv2  = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_maxconv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)
        
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
         # Flattening
        self.en_conv6 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=1)
        self.en_conv7 = nn.Conv2d(128, 128, kernel_size=(3,9), stride=1, padding=0)
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(128, 128)
        self.en_fc2 = nn.Linear(128, 128)
        
        # Classifier output layer
        self.en_fc3 = nn.Linear(128, num_classes)
        
        # Encoder parameter layer
        self.en_fc31 = nn.Linear(128, self.num_latent_dims)
        self.en_fc32 = nn.Linear(128, self.num_latent_dims)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # Fully-connected layers
        self.de_fc3 = nn.Linear(self.num_latent_dims, 128)
        self.de_fc2 = nn.Linear(128, 128)
        self.de_fc1 = nn.Linear(128, 128)
        
        # Unflattening
        self.de_conv7 = nn.ConvTranspose2d(128, 128, kernel_size=(3,9), stride=1, padding=0)
        self.de_conv6 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=1)
        
        # De-convolutions
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.de_maxconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)
        
        self.de_maxconv2  = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        self.de_maxconv1  = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.de_conv1 = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=0)
        
        # Boolean to determine the reparametrization mode
        self.training = train
        
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
        
    # Classifier
    
    def classify(self, X):
        
        # Convolutions
        x = self.relu(self.en_conv1(X))
        x = self.relu(self.en_maxconv1(x))
        
        x = self.en_conv2a(x)
        x = self.relu(self.en_conv2b(x))
        x = self.relu(self.en_maxconv2(x))
        
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_maxconv3(x))
        
        x = self.relu(self.en_conv4(x))
        
        # Flattening
        
        x = self.relu(self.en_conv6(x))
        x = self.en_conv7(x)
        
        x = x.view(-1, 128)
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.en_fc3(x)
        
        return x
        
    # Encoder
    
    def encode(self, X):
        
        # Convolutions
        x = self.relu(self.en_conv1(X))
        x = self.relu(self.en_maxconv1(x))
        
        x = self.en_conv2a(x)
        x = self.relu(self.en_conv2b(x))
        x = self.relu(self.en_maxconv2(x))
        
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        x = self.relu(self.en_maxconv3(x))
        
        x = self.relu(self.en_conv4(x))
        
        # Flattening
        
        x = self.relu(self.en_conv6(x))
        x = self.en_conv7(x)
        
        x = x.view(-1, 128)
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        
        return self.en_fc31(x), self.en_fc32(x)
                              
    # Reparameterization
    
    def reparameterize(self, mu, logvar, shots=1):
        
        if self.training:
            # Calculate the square-root covariance matrix
            std = logvar.mul(0.5).exp_()

            # Sample a vector eps from the normal distribution
            eps = std.new(std.size()).normal_()

            # Return the latent vector
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    # Decoder
    
    def decode(self, X):
        
        # Fully-connected layers
        x = self.de_fc3(X)
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc1(x))
        
        # Unflattening
        x = x.view(-1, 128, 1, 1)
                            
        x = self.de_conv7(x)
        x = self.relu(self.de_conv6(x))
        
        # Deconvolutions
        x = self.relu(self.de_conv4(x))
        
        x = self.relu(self.de_maxconv3(x))
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        
        x = self.relu(self.de_maxconv2(x))
        x = self.relu(self.de_conv2b(x))
        x = self.de_conv2a(x)
        
        x = self.relu(self.de_maxconv1(x))
        x = self.relu(self.de_conv1(x))

        return x
    
    # Sampler
    
    def sample(self):
        # Sample a vector from the normal distribution
        eps = randn(1, self.num_latent_dims, device=device('cuda'))
        
        # Decode the latent vector to generate an event
        return eps, self.decode(eps)