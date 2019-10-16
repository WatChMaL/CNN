"""
vaenet.py

Derived class implementing a fully unsupervised VAE using the encoder from basemodel.py and VAEBottleneck
from bottlenecks.py
"""

# WatChMaL imports
from models import edlenet, edresnet
from models.bottlenecks import VAEBottleneck
from models.basemodel import BaseModel

# PyTorch imports
from torch.nn import Module

# Global variables
_ARCH_DICT_DEC = {0:"dlenet", 1:"dresnet"}

class VaeNet(Module, BaseModel):
    
    def __init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth):
        Module.__init__(self)
        BaseModel.__init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth)
        
        # VaeNet attributes
        self.arch_dec = _ARCH_DICT_DEC[arch_key]
        self.bottleneck = VAEBottleneck(num_latent_dims)
        
        if self.arch_dec == "dlenet":
            assert arch_depth == 9
            self.decoder = getattr(edlenet, self.arch_dec + str(arch_depth))(num_input_channels=num_input_channels,
                                                                             num_latent_dims=num_latent_dims)
        elif self.arch_dec == "dresnet":
            assert arch_depth in [18, 34, 50, 101, 152]
            self.decoder = getattr(edresnet, self.arch_dec + str(arch_depth))(num_input_channels=num_input_channels,
                                                                              num_latent_dims=num_latent_dims)
        else:
            raise NotImplementedError
        
    def forward(self, X, mode):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """
        if mode in ["train","validation"]:
            z_prime = self.encoder(X)
            z, mu, logvar = self.bottleneck(z_prime, None)
            return self.decoder(z, None), z, mu, logvar
        elif mode == "sample":
            z = self.bottleneck(X, mode)
            return self.decoder(z, None), z
        elif mode == "decode":
            return self.decoder(X, None)
            