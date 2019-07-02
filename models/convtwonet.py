"""
convtwonet.py

PyTorch implementation of the ConvtwoNet as a classifier for IWCD detector response

The model uses only convolutional layers in the architecture removing all max pooling layers from the KazuNet.

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn

# ConvNet class
class ConvtwoNet(nn.Module):
    
    # Initializer
    
    def __init__(self, num_input_channels=19, num_classes=3, train=True):
        
        # Initialize the superclass
        super(ConvtwoNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Feature extraction convolutions
        self.en_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1)
        self.en_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.en_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.en_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Downsampling convolution
        self.en_maxconv1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        
        # Feature extraction convolutions
        self.en_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.en_conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.en_fc1 = nn.Linear(1024, 512)
        self.en_fc2 = nn.Linear(512, 256)
        self.en_fc3 = nn.Linear(256, 128)
        
        # Classifier output layer
        self.en_fc4 = nn.Linear(128, num_classes)
        
    # Forward
    
    def forward(self, X):
        return self.classify(X)
    
    # Classifier
    
    def classify(self, X):
        
        # Feature extraction convolutions
        x = self.relu(self.en_conv1(X))
        x = self.relu(self.en_conv2(x))
        x = self.relu(self.en_conv3(x))
        x = self.relu(self.en_conv4(x))
        
        # Downsampling convolution
        x = self.relu(self.en_maxconv1(x))
        
        # Feature extraction convolutions
        x = self.relu(self.en_conv5(x))
        x = self.relu(self.en_conv6(x))
        x = self.relu(self.en_conv7(x))
        
        # Flattening
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.relu(self.en_fc3(x))
        
        # Classifier output layer
        x = self.en_fc4(x)
        
        return x