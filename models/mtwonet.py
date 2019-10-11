"""
mtwonet.py

Derived class implementing a semi-supervised VAE using the encoder from basemodel.py and, M2Bottleneck from bottlenecks.py
"""

# WatChMaL imports
from models import edlenet, edresnet
from models.bottlenecks import M2Bottleneck
from models.basemodel import BaseModel

# PyTorch imports
from torch.nn import Module

# Global variables
_ARCH_DICT_DEC = {0:"dlenet", 1:"dresnet"}

class MtwoNet(Module, BaseModel):
    
    def __init__(self, num_input_channels, num_latent_dims, num_classes, arch_key, arch_depth):
        Module.__init__(self)
        BaseModel.__init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth)
        
        # MtwoNet attributes
        self.arch_dec = _ARCH_DICT_DEC[arch_key]
        self.bottleneck = M2Bottleneck(num_latent_dims, num_classes)
        
        if self.arch == "edlenet":
            assert arch_depth == 9
            self.decoder = getattr(edlenet, self.arch_dec + str(arch_depth))(num_input_channels=num_input_channels,
                                                                             num_latent_dims=num_latent_dims + num_classes)
        elif self.arch == "eresnet":
            assert arch_depth in [18, 34, 50, 101, 152]
            self.decoder = getattr(edresnet, self.arch_dec + str(arch_depth))(num_input_channels=num_input_channels,
                                                                              num_latent_dims=num_latent_dim + num_classes)
        else:
            raise NotImplementedError
        
    def forward(self, X, mode, labels):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """
        z_prime = self.encoder(X)
        z_y, z, mu, logvar, pi = self.bottleneck(z_prime, None, labels=labels)
        return self.decoder(z_y, None), z, mu, logvar, z_prime, pi