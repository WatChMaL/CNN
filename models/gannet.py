"""
gannet.py

Derived class implementing a GAN using the generator and discriminator from basemodel
"""

# WatChMaL imports
from models.bottlenecks import LatentClassifier
from models.basemodel import BaseModel

# PyTorch imports
from torch.nn import Module, init
from torch import Tensor
import numpy as np
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class GanNet(Module, BaseModel):
    
    def __init__(self, num_input_channels, num_latent_dims, num_classes, arch_key, arch_depth, train_all):
        Module.__init__(self)
        BaseModel.__init__(self, num_input_channels, num_latent_dims, arch_key, arch_depth)
        
        #self.classifier = LatentClassifier(num_latent_dims, num_classes)
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)
        
        if not train_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X, Z):
        """Overrides the generic forward() method in torch.nn.Module
        
        Args:
        X -- input minibatch tensor of size (mini_batch, *)
        """

        # Generate a batch of images
        gen_imgs = self.generator(Z)

        # Run discriminator on real and generated images
        dis_genresults = self.discriminator(gen_imgs)
        dis_realresults = self.discriminator(X)
        
        return {'genresults': dis_genresults, 'realresults': dis_realresults, 'genimgs': gen_imgs}

