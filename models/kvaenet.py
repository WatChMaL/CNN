"""
kvaenet.py

PyTorch implementation of the KazuNet as a classifier for input of size Batch_Size x Input_Channels x 16 x 40
Details on the model architecture can be found here : 
    https://github.com/WatChMaL/ExampleNotebooks/blob/maste/HKML%20CNN%20Image%20Classification.ipynb

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn

# KVAENet class
class KVaeNet(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=38, num_classes=3, num_latent_dimensions=64):
        
        # Initialize the superclass
        super(KVaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions and max-pooling
        self.en_conv1  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_max1   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.en_max1_indices = None
        self.en_max1_output_size = None
        
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_max2   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.en_max2_indices = None
        self.en_max2_output_size = None
        
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_max3   = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        
        self.en_max3_indices = None
        self.en_max3_output_size = None
        
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Flattening
        self.en_maxflat_indices = None
        self.en_maxflat_output_size = None
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(128, 128)
        self.en_fc2 = nn.Linear(128, 128)
        
        # Classifier output layer
        self.en_fc3  = nn.Linear(128, num_classes)
            
        # Encoder parameters layer
        self.en_fc31 = nn.Linear(128, num_latent_dimensions)
        self.en_fc32 = nn.Linear(128, num_latent_dimensions)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # Fully-connected layers
        self.de_fc3 = nn.Linear(num_latent_dimensions, 128)
        self.de_fc2 = nn.Linear(128, 128)
        self.de_fc1 = nn.Linear(128, 128)
        
        # Unflattening
        
        # De-convolutions and (un)max-pooling
        self.de_deconv4  = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.de_unmax3   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.de_deconv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_deconv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.de_unmax2   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.de_deconv2b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_deconv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.de_unmax1   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.de_deconv1  = nn.ConvTranspose2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        
    
    
    # Variational Auto-Encoder forward pass
    
    def forward(self, X):
        
        # Encoder to get the parameters for the distribution
        mu, logvar = self.encode(X)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Return the output image, mean and covariance matrix
        return self.decode(z), mu, logvar
    
    """
    
    # Classifier forward pass
    
    def forward(self, X):
        return self.classify(X)
        
    """
    
    # Classifier
    
    def classify(self, X):
        
        # Convolutions and max-pooling
        x = self.en_max1(self.en_conv1(X))
        x = self.en_max2(self.en_conv2b(self.en_conv2a(x)))
        x = self.en_max3(self.en_conv3b(self.en_conv3a(x)))
        
        x = self.en_conv4(x)
        
        # Flattening
        x, self.en_maxflat_indices = nn.MaxPool2d(x.size()[2:], return_indices=True)(x)
        x = x.view(-1, 128)
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.en_fc3(x)
        
        return x
        
    # Encoder
    
    def encode(self, X):
        
        # Convolutions and max-pooling
        x, self.en_max1_indices = self.en_max1(self.en_conv1(X))
        self.en_max1_output_size = x.size()
        
        x, self.en_max2_indices = self.en_max2(self.en_conv2b(self.en_conv2a(x)))
        self.en_max2_output_size = x.size()
        
        x, self.en_max3_indices = self.en_max3(self.en_conv3b(self.en_conv3a(x)))
        self.en_max3_output_size = x.size()
        
        x = self.en_conv4(x)
        
        # Flattening
        x, self.en_maxflat_indices = nn.MaxPool2d(x.size()[2:], return_indices=True)(x)
        self.en_maxflat_output_size = x.size()
        x = x.view(-1, 128)
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        
        return self.en_fc31(x), self.en_fc32(x)
    
    # Reparameterization
    
    def reparameterize(self, mu, logvar):
        
        # Calculate the square-root covariance matrix
        std = logvar.mul(0.5).exp_()
        
        # Sample a vector eps from the normal distribution
        eps = std.data.new(std.size()).normal_()
        
        # Return the latent vector
        return eps.mul_(std).add_(mu)
    
    # Decoder
    
    def decode(self, X):
        
        # Fully-connected layers
        x = self.de_fc1(X)
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc3(x))
        
        # Unflattening
        x = x.view(128, 1, 1)
        x = nn.MaxUnpool2d(x.size()[2:])(x, self.en_maxflat_indices, self.en_maxflat_output_size) 
        
        # De-convolutions and (un)max-pooling
        x = self.de_deconv4(x)
        
        x = self.de_unmax3(x, self.en_max3_indices, self.en_max3_output_size)
        x = self.de_deconv3a(self.de_deconv3b(x))
        
        x = self.de_unmax2(x, self.en_max2_indices, self.en_max2_output_size)
        x = self.de_deconv2a(self.de_deconv2b(x))
        
        x = self.de_unmax1(x, self.en_max1_indices, self.en_max1_output_size)
        x = self.de_deconv1(x)
        
        return x