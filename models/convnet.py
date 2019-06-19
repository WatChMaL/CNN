"""
convnet.py

PyTorch implementation of the ConvNet as a classifier for input of size : 
    Batch_Size x Input_Channels x 16 x 40
The model uses only convolutional layers in the architecture removing all max pooling layers from the KazuNet.

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn

# ConvNet class
class ConvNet(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=38, num_classes=3, train=True):
        
        # Initialize the superclass
        super(ConvNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
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
        self.en_conv5a = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv5b = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.en_conv5c = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        
        self.en_conv6 = nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0)
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(256, 128)
        self.en_fc2 = nn.Linear(128, 128)
        self.en_fc3 = nn.Linear(128, num_classes)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # De-convolutions and (un)max-pooling
        
    # Forward pass
    
    def forward(self, X):
        return self.classify(X)
        
    # Classifier
    
    def classify(self, X):
        
        # Convolutions
        x = self.en_maxconv1(self.en_conv1(X))
        x = self.en_maxconv2(self.en_conv2b(self.en_conv2a(x)))
        x = self.en_maxconv3(self.en_conv3b(self.en_conv3a(x)))
        
        x = self.en_conv4(x)
        
        # Flattening
        x = self.en_conv5c(self.en_conv5b(self.en_conv5a(x)))
        x = self.en_conv6(x)
        
        x = x.view(-1, 256)
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.en_fc3(x)
        
        return x
        
    # Encoder
    
    def encode(self, X):
        return X
    
    # Reparameterization
    
    def reparameterize(self, X):
        return X
    
    # Decoder
    
    def decode(self, X):
        return X