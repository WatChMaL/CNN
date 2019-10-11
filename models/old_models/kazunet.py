"""
kazunet.py

PyTorch implementation of the KazuNet as a classifier for input of size Batch_Size x Input_Channels x 16 x 40
Details on the model architecture can be found here : 
    https://github.com/WatChMaL/ExampleNotebooks/blob/master/HKML%20CNN%20Image%20Classification.ipynb

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn

# KazuNet class
class KazuNet(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=38, num_classes=3, train=True):
        
        # Initialize the superclass
        super(KazuNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions and max-pooling
        self.en_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_max1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_max2   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.en_max3   = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Flattening
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(128, 128)
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
        
        # Convolutions and max-pooling
        x = self.en_conv1(X)
        x = self.en_max1(x)
                      
        x = self.en_conv2a(x)
        x = self.en_conv2b(x)
        x = self.en_max2(x)
        
        x = self.en_conv3a(x)
        x = self.en_conv3b(x)
        x = self.en_max3(x)
        
        x = self.en_conv4(x)
        
        # Flattening
        x = nn.MaxPool2d(x.size()[2:])(x)
        x = x.view(-1, 128)
        
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