"""
edtworesnet.py

PyTorch implementation of ResNet-style architecture to be used as an encoder and decoder in the variational
inference models with a corresponding symmetric decoder.

Model run separately on the data for each end cap and the barrel and 'stitched' together.
"""

# +
# For debugging
import pdb

# PyTorch imports
from torch import cat
from torch.nn import Module, Sequential, Linear, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU
from torch.nn.init import kaiming_normal_, constant_
# -

# WatChMaL imports
from models import resnetblocks

# Global variables
__all__ = ['etworesnet18', 'etworesnet34', 'etworesnet50', 'etworesnet101', 'etworesnet152',
           'dtworesnet18', 'dtworesnet34', 'dtworesnet50', 'dtworesnet101', 'dtworesnet152']
_RELU = ReLU()

# -------------------------------
# Encoder architecture layers
# -------------------------------

class EtworesNet(Module):

    def __init__(self, block, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        
        self.inplanes = 64
        
        self.conv1 = Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(16)
        
        self.conv2 = Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(64)
        
        self.layer0 = resnetblocks.EresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion * 2
        self.bool_deep = False
        
        self.conv3barrel = Conv2d(int(self.unroll_size/2), int(self.unroll_size/2), kernel_size=(1,4), stride=(1,1))
        self.conv3endcap = Conv2d(int(self.unroll_size/4), int(self.unroll_size/4), kernel_size=(1,1), stride=(1,1))
        
        self.bn3barrel = BatchNorm2d(int(self.unroll_size/2))
        self.bn3endcap = BatchNorm2d(int(self.unroll_size/4))
        
        for m in self.modules():
            if isinstance(m, resnetblocks.EresNetBottleneck):
                self.fc1 = Linear(self.unroll_size, int(self.unroll_size/2))
                self.fc2 = Linear(int(self.unroll_size/2), int(self.unroll_size/4))
                self.fc3 = Linear(int(self.unroll_size/4), num_latent_dims)
                self.bool_deep = True
                break
                
        if not self.bool_deep:
            self.fc1 = Linear(self.unroll_size, int(self.unroll_size/2))
            self.fc2 = Linear(int(self.unroll_size/2), num_latent_dims)
        
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 512:
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
        
        # -------------------------------------------
        # Split the input into the 3 components
        # -------------------------------------------
        X_endcap_top = X[:, :, 0:12, 14:26]
        X_barrel = X[:, :, 12:28, :]
        X_endcap_bottom = X[:, :, 28:40, 14:26]
        
        # -------------------------------------------
        # Apply the operations on the top endcap data
        # -------------------------------------------
        x = self.conv1(X_endcap_top)
        #x = self.bn1(x)
        x = _RELU(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = _RELU(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv3endcap(x)
        #x = self.bn3endcap(x)
        x = _RELU(x)
        
        x_endcap_top = x.view(x.size(0), -1)
            
        # -------------------------------------------
        # Apply the operations on the central barrel
        # -------------------------------------------
        x = self.conv1(X_barrel)
        #x = self.bn1(x)
        x = _RELU(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = _RELU(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv3barrel(x)
        #x = self.bn3barrel(x)
        x = _RELU(x)
        
        x_barrel = x.view(x.size(0), -1)
        
        # -------------------------------------------
        # Apply the operations on the bottom endcap
        # -------------------------------------------
        x = self.conv1(X_endcap_bottom)
        #x = self.bn1(x)
        x = _RELU(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = _RELU(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv3endcap(x)
        #x = self.bn3endcap(x)
        x = _RELU(x)
        
        x_endcap_bottom = x.view(x.size(0), -1)
        
        # -------------------------------------------
        # Concatenate the 1d features extracted
        # -------------------------------------------
        x_tank = cat((x_endcap_top, x_barrel, x_endcap_bottom), dim=1)
        
        x = _RELU(self.fc1(x_tank))
        x = _RELU(self.fc2(x))
        if self.bool_deep:
            x = _RELU(self.fc3(x))
            
        return x

#-------------------------------
# Decoder architecture layers
#-------------------------------
class DtworesNet(Module):

    def __init__(self, block, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        
        self.inplanes = 64
        
        self.conv1 = ConvTranspose2d(16, num_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv2 = ConvTranspose2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(16)
        
        self.layer0 = resnetblocks.DresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion
        self.bool_deep = False
        
        self.conv3 = ConvTranspose2d(self.unroll_size, self.unroll_size, kernel_size=(4,4), stride=(1,1))
        self.bn3   = BatchNorm2d(self.unroll_size)
        
        for m in self.modules():
            if isinstance(m, resnetblocks.DresNetBottleneck):
                self.fc2 = Linear(num_latent_dims, int(self.unroll_size/2))
                self.fc1 = Linear(int(self.unroll_size/2), self.unroll_size)
                self.bool_deep = True
                break
                
        if not self.bool_deep:
            self.fc1 = Linear(num_latent_dims, self.unroll_size)

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
                if isinstance(m, resnetblocks.DresNetBottleneck):
                    constant_(m.bn3.weight, 0)
                elif isinstance(m, resnetblocks.DresNetBasicBlock):
                    constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or planes:
            if planes < 512:
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

    def forward(self, X, unflat_size):
        if self.bool_deep:
            x = _RELU(self.fc2(X))
        else:
            x = X
            
        x = _RELU(self.fc1(x))
        
        x = x.view(unflat_size) if unflat_size is not None else x.view(x.size(0), -1, 1, 1)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = _RELU(x)
        
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.layer0(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = _RELU(x)
        
        x = _RELU(self.conv1(x))

        return x


# -------------------------------------------------------
# Initializers for model encoders with various depths
# -------------------------------------------------------

def etworesnet18(**kwargs):
    """Constructs a EresNet-18 model encoder.
    """
    return EtworesNet(resnetblocks.EresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def etworesnet34(**kwargs):
    """Constructs a EresNet-34 model encoder.
    """
    return EtworesNet(resnetblocks.EresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def etworesnet50(**kwargs):
    """Constructs a EresNet-50 model encoder.
    """
    return EtworesNet(resnetblocks.EresNetBottleneck, [3, 4, 6, 3], **kwargs)

def etworesnet101(**kwargs):
    """Constructs a EresNet-101 model encoder.
    """
    return EtworesNet(resnetblocks.EresNetBottleneck, [3, 4, 23, 3], **kwargs)

def etworesnet152(**kwargs):
    """Constructs a ErsNet-152 model encoder.
    """
    return EtworesNet(resnetblocks.EresNetBottleneck, [3, 8, 36, 3], **kwargs)

# -------------------------------------------------------
# Initializers for model decoders with various depths
# -------------------------------------------------------

def dtworesnet18(**kwargs):
    """Constructs a DresNet-18 model decoder.
    """
    return DtworesNet(resnetblocks.DresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def dtworesnet34(**kwargs):
    """Constructs a DresNet-34 model decoder.
    """
    return DtworesNet(resnetblocks.DresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def dtworesnet50(**kwargs):
    """Constructs a DresNet-50 model encoder.
    """
    return DtworesNet(resnetblocks.DresNetBottleneck, [3, 4, 6, 3], **kwargs)

def dtworesnet101(**kwargs):
    """Constructs a DresNet-101 model decoder.
    """
    return DtworesNet(resnetblocks.DresNetBottleneck, [3, 4, 23, 3], **kwargs)

def dtworesnet152(**kwargs):
    """Constructs a DresNet-152 model decoder.
    """
    return DtworesNet(resnetblocks.DresNetBottleneck, [3, 8, 36, 3], **kwargs)
