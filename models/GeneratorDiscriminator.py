"""
GeneratorDiscrimantor.py

PyTorch implementation of Generator and Disciminator models for GAN using ResNet-style architecture.

End caps 'pasted' into data.
"""
# For debugging
import pdb

# PyTorch imports
from torch.nn import Module, Sequential, Linear, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh
from torch.nn.init import kaiming_normal_, constant_

# WatChMaL imports
from models import resnetblocks

# Global variables
__all__ = ['genresnet18', 'genresnet34', 'genresnet50', 'genresnet101', 'genresnet152',
           'disresnet18', 'disresnet34', 'disresnet50', 'disresnet101', 'disresnet152']
_RELU = ReLU()
_LeakyRELU = LeakyReLU(0.2, True)
_Sigmoid = Sigmoid()
_Tanh = Tanh()

# -------------------------------
# Generator architecture layers
# -------------------------------

class Generator(Module):

    def __init__(self, block1, block2, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        
        # downsampling
        '''
        self.inplanes = 64
        
        self.conv1 = Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(64)
        
        self.layer0 = resnetblocks.EresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer_up(block1, 64, layers[0], stride=2)
        self.layer2 = self._make_layer_up(block1, 64, layers[1], stride=2)
        self.layer3 = self._make_layer_up(block1, 64, layers[2], stride=2)
        self.layer4 = self._make_layer_up(block1, 128, layers[3], stride=2)
        
        self.unroll_size = 128 * block1.expansion
        self.bool_deep = False
        
        self.conv3a = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(4,4), stride=(1,1))
        #self.conv3b = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        #self.conv3c = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(2,2), stride=(1,1))
        self.bn3    = BatchNorm2d(self.unroll_size)
        
        self.fc1 = Linear(self.unroll_size, num_latent_dims)
        
        # upsampling
        
        self.inplanes = 64
        
        self.dconv1 = ConvTranspose2d(16, num_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.dconv2 = ConvTranspose2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.dbn2   = BatchNorm2d(16)
        
        self.dlayer0 = resnetblocks.DresNetBasicBlock(64, 64)
        self.dlayer1 = self._make_layer_down(block2, 64, layers[0], stride=2)
        self.dlayer2 = self._make_layer_down(block2, 64, layers[1], stride=2)
        self.dlayer3 = self._make_layer_down(block2, 64, layers[2], stride=2)
        self.dlayer4 = self._make_layer_down(block2, 128, layers[3], stride=2)
        self.unroll_size = 128 * block2.expansion

        self.dconv3 = ConvTranspose2d(self.unroll_size, self.unroll_size, kernel_size=(4,4), stride=(1,1))
        self.dbn3   = BatchNorm2d(self.unroll_size)
                
        self.dfc1 = Linear(num_latent_dims, self.unroll_size)

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnetblocks.EresNetBasicBlock):
                    constant_(m.bn2.weight, 0)
                elif isinstance(m, resnetblocks.DresNetBasicBlock):
                    constant_(m.bn2.weight, 0)
        '''
         
        ngf = 64
        nc = 19
        
        self.conv1 = ConvTranspose2d(128, ngf * 8, 4, 2, 0, bias=False)
        self.bn1 = BatchNorm2d(ngf * 8)
        
        self.conv2 = ConvTranspose2d(ngf * 8, ngf * 4, 3, 3, 1, bias=False)
        self.bn2 = BatchNorm2d(ngf * 4)
        
        self.conv3 = ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm2d(ngf * 2)
        
        self.conv4 = ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm2d(ngf)
        
        self.conv5 = ConvTranspose2d( ngf, nc, 3, 1, 1, bias=False)
        
        ''' 
        
        ngf = 64
        nc = 3
        self.main = Sequential(
            # input is Z, going into a convolution
            ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            BatchNorm2d(ngf * 8),
            ReLU(True),
            # state size. (ngf*8) x 4 x 4
            ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            BatchNorm2d(ngf * 4),
            ReLU(True),
            # state size. (ngf*4) x 8 x 8
            ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            BatchNorm2d(ngf * 2),
            ReLU(True),
            # state size. (ngf*2) x 16 x 16
            ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            BatchNorm2d(ngf),
            ReLU(True),
            # state size. (ngf) x 32 x 32
            ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            Tanh()
            # state size. (nc) x 64 x 64
        )
        '''
        


    def _make_layer_up(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 128:
                downsample = Sequential(
                    resnetblocks.conv4x4(self.inplanes, planes * block.expansion, stride),
                    BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = Sequential(
                    resnetblocks.conv2x2(self.inplanes, planes * block.expansion),
                    BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)
    
    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 128:
                downsample = Sequential(
                    resnetblocks.convtranspose4x4(planes * block.expansion, self.inplanes, stride),
                    BatchNorm2d(self.inplanes),
                )
            else:
                downsample = Sequential(
                    resnetblocks.convtranspose2x2(planes * block.expansion, self.inplanes),
                    BatchNorm2d(self.inplanes),
                )
             
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        layers.reverse()
        
        return Sequential(*layers)

    def forward(self, X):
        
        # downsampling
        '''
        x = self.conv1(X)
        x = self.bn1(x)
        x = _RELU(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = _RELU(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if x.size()[-2:] == (4,4):
            x = self.conv3a(x)
        elif x.size()[-2:] == (1,1):
            x = self.conv3b(x)
        elif x.size()[-2:] == (2,2):
            x = self.conv3c(x)
       
        x = self.bn3(x)
        x = _RELU(x)

        x = x.view(x.size(0), -1)
        
        x = _RELU(self.fc1(x))
        
        
        # upsampling
        
        x = _RELU(self.dfc1(X))

        #x = x.view(unflat_size) if unflat_size is not None else x.view(x.size(0), -1, 1, 1)
        x = x.view(x.size(0), -1, 1, 1)

        x = self.dconv3(x)
        x = self.dbn3(x)
        x = _RELU(x)
        
        x = self.dlayer4(x)
        x = self.dlayer3(x)
        x = self.dlayer2(x)
        x = self.dlayer1(x)
        x = self.dlayer0(x)
        x = self.dconv2(x)
        x = self.dbn2(x)
        x = _RELU(x)
        
        x = _RELU(self.dconv1(x))
        '''
        
        x = self.conv1(X)
        x = self.bn1(x)
        x = _RELU(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = _RELU(x)
       
        x = self.conv3(x)
        x = self.bn3(x)
        x = _RELU(x)
     
        x = self.conv4(x)
        x = self.bn4(x)
        x = _RELU(x)

        x = self.conv5(x)
        #x = _Tanh(x)
        '''
        x = self.main(X)
        '''
        return x

#-------------------------------
# Discriminator architecture layers
#-------------------------------
class Discriminator(Module):

    def __init__(self, block1, block2, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        '''
        self.inplanes = 64
        
        self.conv1 = Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(64)
        
        self.layer0 = resnetblocks.EresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer_up(block1, 64, layers[0], stride=2)
        self.layer2 = self._make_layer_up(block1, 64, layers[1], stride=2)
        self.layer3 = self._make_layer_up(block1, 64, layers[2], stride=2)
        self.layer4 = self._make_layer_up(block1, 128, layers[3], stride=2)
        
        self.unroll_size = 128 * block1.expansion
        self.bool_deep = False
        
        self.conv3a = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(4,4), stride=(1,1))
        #self.conv3b = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        #self.conv3c = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(2,2), stride=(1,1))
        self.bn3    = BatchNorm2d(self.unroll_size)
        
        self.fc1 = Linear(self.unroll_size, num_latent_dims)
        
        self.cl_fc1 = Linear(num_latent_dims, int(num_latent_dims/2))
        self.cl_fc2 = Linear(int(num_latent_dims/2), int(num_latent_dims/4))
        self.cl_fc3 = Linear(int(num_latent_dims/4), int(num_latent_dims/8))
        self.cl_fc4 = Linear(int(num_latent_dims/8), 1)
        '''
        
        nc = 19
        ndf = 64
        
        #self.conv1 = Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        #self.conv2 = Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias = False)
        #self.bn2 = BatchNorm2d(ndf*2)
        #self.conv3 = Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias = False)
        #self.bn3 = BatchNorm2d(ndf*4)
        #self.conv4 = Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias = False)
        #self.bn4 = BatchNorm2d(ndf*8)
        #self.conv5 = Conv2d(ndf*8, 1, kernel_size=2, stride=1, padding=0, bias=False)
        
        self.main = Sequential(
            # input is (nc) x 64 x 64
            Conv2d(nc, ndf, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 2),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 4),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 8),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            Sigmoid()
        )
        
        '''
        nc = 19
        ndf = 64
        
        self.main = Sequential(
            # input is (nc) x 64 x 64
            Conv2d(nc, ndf, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 2),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 4),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            BatchNorm2d(ndf * 8),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            Sigmoid()
        )
        '''
            
        '''
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnetblocks.EresNetBottleneck):
                    constant_(m.bn3.weight, 0)
                elif isinstance(m, resnetblocks.EresNetBasicBlock):
                    constant_(m.bn2.weight, 0)
        '''
        
    def _make_layer_up(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 128:
                downsample = Sequential(
                    resnetblocks.conv4x4(self.inplanes, planes * block.expansion, stride),
                    BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = Sequential(
                    resnetblocks.conv2x2(self.inplanes, planes * block.expansion),
                    BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)    

    def forward(self, X):
        
        # downsampling
        '''
        x = self.conv1(X)
        x = self.bn1(x)
        x = _RELU(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = _RELU(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if x.size()[-2:] == (4,4):
            x = self.conv3a(x)
        elif x.size()[-2:] == (1,1):
            x = self.conv3b(x)
        elif x.size()[-2:] == (2,2):
            x = self.conv3c(x)
       
        x = self.bn3(x)
        x = _RELU(x)
        
        x = x.view(x.size(0), -1)
        
        x = _RELU(self.fc1(x))
        
        # Fully-connected layers
        x = _RELU(self.cl_fc1(x))
        x = _RELU(self.cl_fc2(x))
        x = _RELU(self.cl_fc3(x))
        x = self.cl_fc4(x)
        
        '''
        '''
       
        x = self.conv1(X)
        _LeakyRELU(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = _LeakyRELU(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = _LeakyRELU(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = _LeakyRELU(x)
        
        x = self.conv5(x)
        
        x = _Sigmoid(x)
        '''
        #x = x.view(x.size(0), -1)
        x = self.main(X)
        
        return x


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
