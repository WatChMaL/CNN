"""
clnet.py

Derived class implementing a fully supervised classifier using the encoder from basemodel and a latent vector
classifier from bottlenecks
"""

# WatChMaL imports
from models.bottlenecks import LatentClassifier
from models.basemodel import BaseModel

# PyTorch imports
from torch.nn import Module

class ClNet(Module, BaseModel):
    
    def __init__(self, num_input_channels, num_latent_dims, num_classes, arch_key, arch_depth, train_all):
        Module.__init__(self)
        BaseModel.__init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth)
        
        self.classifier = LatentClassifier(num_latent_dims, num_classes)
        
        if not train_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """
        x = self.encoder(X)
        return self.classifier(x)