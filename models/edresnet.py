"""
edresnet.py

PyTorch implementation of ResNet-style architecture to be used as an encoder and decoder in the variational
inference models with a corresponding symmetric decoder.

End caps 'pasted' into data.
"""
# For debugging
import pdb

# PyTorch imports
from torch.nn import Module, Sequential, Linear, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU
from torch.nn.init import kaiming_normal_, constant_

# WatChMaL imports
from models import resnetblocks

# Global variables
__all__ = ['eresnet18', 'eresnet34', 'eresnet50', 'eresnet101', 'eresnet152',
           'dresnet18', 'dresnet34', 'dresnet50', 'dresnet101', 'dresnet152']
_RELU = ReLU()

# -------------------------------
# Encoder architecture layers
# -------------------------------

class EresNet(Module):

    def __init__(self, block, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        
        self.inplanes = 64
        
        #self.conv1 = Conv2d(num_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bn1   = BatchNorm2d(64)
        
        # added resnet blocks that act on single pixel
        #self.minires0 = resnetblocks.EresNetMiniBlock(num_input_channels, 64)
        #self.minires1 = self._make_mini_layer(resnetblocks.EresNetMiniBlock, 64, layers[0], stride=1)
        #self.minires2 = self._make_mini_layer(resnetblocks.EresNetMiniBlock, 64, layers[1], stride=1)
        #self.minires3 = self._make_mini_layer(resnetblocks.EresNetMiniBlock, 64, layers[2], stride=1)
        #self.minires4 = self._make_mini_layer(resnetblocks.EresNetMiniBlock, 64, layers[3], stride=1)
       
        self.conv1 = Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(64)
        
        self.layer0 = resnetblocks.EresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        
        self.unroll_size = 128 * block.expansion
        self.bool_deep = False
        
        self.conv3a = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(4,4), stride=(1,1))
        self.conv3b = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        self.conv3c = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(2,2), stride=(1,1))
        self.bn3    = BatchNorm2d(self.unroll_size)
        
        
        for m in self.modules():
            if isinstance(m, resnetblocks.EresNetBottleneck):
                self.fc1 = Linear(self.unroll_size, int(self.unroll_size/2))
                self.fc2 = Linear(int(self.unroll_size/2), num_latent_dims)
                self.bool_deep = True
                break
                
        if not self.bool_deep:
            self.fc1 = Linear(self.unroll_size, num_latent_dims)

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
    
    def _make_mini_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 512:
                downsample = Sequential(
                    resnetblocks.conv1x1(self.inplanes, planes * block.expansion, stride),
                    BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = Sequential(
                    resnetblocks.conv1x1(self.inplanes, planes * block.expansion),
                    BatchNorm2d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, X):
        
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
        #pdb.set_trace()
        x = x.view(x.size(0), -1)
        #pdb.set_trace()
        x = _RELU(self.fc1(x))
        if self.bool_deep:
            x = _RELU(self.fc2(x))
        #pdb.set_trace()
        return x

#-------------------------------
# Decoder architecture layers
#-------------------------------
class DresNet(Module):

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

def eresnet18(**kwargs):
    """Constructs a EresNet-18 model encoder.
    """
    return EresNet(resnetblocks.EresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def eresnet34(**kwargs):
    """Constructs a EresNet-34 model encoder.
    """
    return EresNet(resnetblocks.EresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def eresnet50(**kwargs):
    """Constructs a EresNet-50 model encoder.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 4, 6, 3], **kwargs)

def eresnet101(**kwargs):
    """Constructs a EresNet-101 model encoder.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 4, 23, 3], **kwargs)

def eresnet152(**kwargs):
    """Constructs a ErsNet-152 model encoder.
    """
    return EresNet(resnetblocks.EresNetBottleneck, [3, 8, 36, 3], **kwargs)

# -------------------------------------------------------
# Initializers for model decoders with various depths
# -------------------------------------------------------

def dresnet18(**kwargs):
    """Constructs a DresNet-18 model decoder.
    """
    return DresNet(resnetblocks.DresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def dresnet34(**kwargs):
    """Constructs a DresNet-34 model decoder.
    """
    return DresNet(resnetblocks.DresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def dresnet50(**kwargs):
    """Constructs a DresNet-50 model encoder.
    """
    return DresNet(resnetblocks.DresNetBottleneck, [3, 4, 6, 3], **kwargs)

def dresnet101(**kwargs):
    """Constructs a DresNet-101 model decoder.
    """
    return DresNet(resnetblocks.DresNetBottleneck, [3, 4, 23, 3], **kwargs)

def dresnet152(**kwargs):
    """Constructs a DresNet-152 model decoder.
    """
    return DresNet(resnetblocks.DresNetBottleneck, [3, 8, 36, 3], **kwargs)
