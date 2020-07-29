"""
GeneratorDiscrimantor.py

PyTorch implementation of Generator and Disciminator models for GAN using ResNet-style architecture.

End caps 'pasted' into data.
"""
# For debugging
import pdb

# PyTorch imports
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

# WatChMaL imports
from models import resnetblocks

# Global variables
__all__ = ['genresnet18', 'genresnet34', 'genresnet50', 'genresnet101', 'genresnet152',
           'disresnet18', 'disresnet34', 'disresnet50', 'disresnet101', 'disresnet152']
# _RELU = ReLU()
# _LeakyRELU = LeakyReLU(0.2, True)
# _Sigmoid = Sigmoid()
# _Tanh = Tanh()

# -------------------------------
# Generator architecture layers
# -------------------------------

nz = 128
nc = 3
ngf = 64
ndf = 64    

class Generator(nn.Module):
    def __init__(self, block1, block2, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        super(Generator, self).__init__()
        self.ngpu = 2
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


    # def _make_layer_up(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         if planes < 128:
    #             downsample = Sequential(
    #                 resnetblocks.conv4x4(self.inplanes, planes * block.expansion, stride),
    #                 BatchNorm2d(planes * block.expansion),
    #             )
    #         else:
    #             downsample = Sequential(
    #                 resnetblocks.conv2x2(self.inplanes, planes * block.expansion),
    #                 BatchNorm2d(planes * block.expansion),
    #             )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return Sequential(*layers)
    
    # def _make_layer_down(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         if planes < 128:
    #             downsample = Sequential(
    #                 resnetblocks.convtranspose4x4(planes * block.expansion, self.inplanes, stride),
    #                 BatchNorm2d(self.inplanes),
    #             )
    #         else:
    #             downsample = Sequential(
    #                 resnetblocks.convtranspose2x2(planes * block.expansion, self.inplanes),
    #                 BatchNorm2d(self.inplanes),
    #             )
             
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #     layers.reverse()
        
    #     return Sequential(*layers)

#-------------------------------
# Discriminator architecture layers
#-------------------------------
class Discriminator(nn.Module):
    def __init__(self, block1, block2, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        super(Discriminator, self).__init__()
        self.ngpu = 2
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# -------------------------------------------------------
# Initializers for model encoders with various depths
# -------------------------------------------------------

def genresnet18(**kwargs):
    """Constructs a generator based on a ResNet-18 model.
    """
    return Generator(resnetblocks.EresNetBasicBlock, resnetblocks.DresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def genresnet34(**kwargs):
    """Constructs a generator based on a ResNet-34 model.
    """
    return EresNet(resnetblocks.EresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def genresnet50(**kwargs):
    """Constructs a generator based on a ResNet-50 model.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 4, 6, 3], **kwargs)

def genresnet101(**kwargs):
    """Constructs a generator based on a ResNet-101 model.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 4, 23, 3], **kwargs)

def genresnet152(**kwargs):
    """Constructs a generator based on a ResNet-152 model.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 8, 36, 3], **kwargs)

# -------------------------------------------------------
# Initializers for model decoders with various depths
# -------------------------------------------------------

def disresnet18(**kwargs):
    """Constructs a discriminator based on a ResNet-18 model decoder.
    """
    return Discriminator(resnetblocks.EresNetBasicBlock, resnetblocks.DresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def disresnet34(**kwargs):
    """Constructs a discriminator based on a ResNet-34 model decoder.
    """
    return Discriminator(resnetblocks.DresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def disresnet50(**kwargs):
    """Constructs a discriminator based on a ResNet-50 model decoder.
    """
    return Discriminator(resnetblocks.DresNetBottleneck, [3, 4, 6, 3], **kwargs)

def disresnet101(**kwargs):
    """Constructs a discriminator based on a ResNet-101 model decoder.
    """
    return Discriminator(resnetblocks.DresNetBottleneck, [3, 4, 23, 3], **kwargs)

def disresnet152(**kwargs):
    """Constructs a discriminator based on a ResNet-152 model decoder.
    """
    return Discriminator(resnetblocks.DresNetBottleneck, [3, 8, 36, 3], **kwargs)
