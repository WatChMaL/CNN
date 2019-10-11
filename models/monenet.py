"""
monenet.py

Derived class implementing a semi-supervised classifier using the encoder from basemodel.py and, VAEBottleneck and
LatentClassiifer from bottlenecks.py
"""

# WatChMaL imports
from models.bottlenecks import VAEBottleneck, LatentClassifer
from models.basemodel import BaseModel

# PyTorch imports
from torch.nn import Module

class MoneNet(Module, BaseModel):
    
    def __init__(self, num_input_channels, num_latent_dims, num_classes, arch_key, arch_depth):
        Module.__init__(self)
        BaseModel.__init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth)
        
        # MoneNet attributes
        self.bottleneck = VAEBottleneck(num_latent_dims)
        self.classifier = LatentClassifier(num_latent_dims, num_classes)
        
    def forward(self, X):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """
        z_prime = self.encoder(X)
        z, mu, logvar = self.bottleneck(z_prime, None)
        return self.classifier(z), z, mu, logvar