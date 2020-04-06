"""
basemodel.py

Abstract Base Class to be used to implement the unsupervised and semi-supervised
deep generative models and their extensions in the VAE framework
"""

# Python standard imports
from abc import ABC, abstractmethod

# WatChMaL imports
from models import edlenet, edresnet, edtworesnet, GeneratorDiscriminator

# Global variables
_ARCH_DICT_ENC = {0:"elenet", 1:"eresnet", 2:"etworesnet", 3:"gan"}

class BaseModel(ABC):
    
    def __init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth):
        super().__init__()
        
        # BaseModel attributes
        self.arch_enc = _ARCH_DICT_ENC[arch_key]
        
        if self.arch_enc == "elenet":
            assert arch_depth == 9
            self.encoder = getattr(edlenet, self.arch_enc + str(arch_depth))(num_input_channels=num_input_channels,
                                                                             num_latent_dims=num_latent_dims)
        elif self.arch_enc == "eresnet":
            assert arch_depth in [18, 34, 50, 101, 152]
            self.encoder = getattr(edresnet, self.arch_enc + str(arch_depth))(num_input_channels=num_input_channels,
                                                                              num_latent_dims=num_latent_dims)
        elif self.arch_enc == "etworesnet":
            assert arch_depth in [18, 34, 50, 101, 152]
            self.encoder = getattr(edtworesnet, self.arch_enc + str(arch_depth))(num_input_channels=num_input_channels,
                                                                                 num_latent_dims=num_latent_dims)
        elif self.arch_enc == "gan":
            assert arch_depth in [18, 34, 50, 101, 152]
            self.generator = getattr(GeneratorDiscriminator, "genresnet" + str(arch_depth))(num_input_channels=num_input_channels,
                                                                                 num_latent_dims=num_latent_dims)
            self.discriminator = getattr(GeneratorDiscriminator, "disresnet" + str(arch_depth))(num_input_channels=num_input_channels,
                                                                                 num_latent_dims=num_latent_dims)
        else:
            raise NotImplementedError

