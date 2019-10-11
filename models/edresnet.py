"""
eresnet.py

PyTorch implementation of ResNet-style architecture to be used as an encoder and decoder in the variational
inference models with a corresponding symmetric decoder.
"""

from torch.nn import Module, Sequential, Linear, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU
from torch.nn.init import kaiming_normal_, constant_

# Global variables
__all__ = ['eresnet18', 'eresnet34', 'eresnet50', 'eresnet101', 'eresnet152',
           'dresnet18', 'dresnet34', 'dresnet50', 'dresnet101', 'dresnet152']
_RELU = ReLU()

#-------------------------------
# Encoder Conv2d layers
#-------------------------------

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv2x2(in_planes, out_planes, stride=1):
    """2x2 convoltion"""
    return Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv4x4(in_planes, out_planes, stride=1):
    """4x4 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)

#-------------------------------
# Decoder ConvTranspose2d layers
#-------------------------------

def convtranspose1x1(in_planes, out_planes, stride=1):
    """1x1 transposed convolution"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convtranspose2x2(in_planes, out_planes, stride=1):
    """2x2 transposed convoltion"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)
    
def convtranspose3x3(in_planes, out_planes, stride=1):
    """3x3 transposed convolution with padding"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def convtranspose4x4(in_planes, out_planes, stride=1):
    """4x4 transposed convolution with padding"""
    return ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)


#-------------------------------
# ResNet encoder block layers
#-------------------------------

class EresNetBasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        if downsample is None:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            if planes < 512:
                self.conv1 = conv4x4(inplanes, planes, stride=2)
            else:
                self.conv1 = conv2x2(inplanes, planes)
            
        self.bn1        = BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = _RELU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity        
        out = _RELU(out)

        return out


class EresNetBottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = conv1x1(inplanes, planes)            
        self.bn1   = BatchNorm2d(planes)
        
        if downsample is None:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            if planes < 512:
                self.conv2 = conv4x4(planes, planes, stride=2)
            else:
                self.conv2 = conv2x2(planes, planes)
                
        self.bn2 = BatchNorm2d(planes)
        
        self.conv3      = conv1x1(planes, planes * self.expansion)
        self.bn3        = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = _RELU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = _RELU(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity
        out = _RELU(out)

        return out
    
#-------------------------------
# ResNet decoder block layers
#-------------------------------

class DresNetBasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        if downsample is None:
            self.conv1 = convtranspose3x3(planes, inplanes, stride)
        else:
            if planes < 512:
                self.conv1 = convtranspose4x4(planes, inplanes, stride=2)
            else:
                self.conv1 = convtranspose2x2(planes, inplanes)
            
        self.bn1 = BatchNorm2d(inplanes)    
        
        self.conv2      = convtranspose3x3(planes, planes)
        self.bn2        = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, X):
        out = self.conv2(X)
        out = self.bn2(out)
        out = _RELU(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity        
        out = _RELU(out)

        return out


class DresNetBottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = convtranspose1x1(planes, inplanes)            
        self.bn1   = BatchNorm2d(inplanes)
        
        if downsample is None:
            self.conv2 = convtranspose3x3(planes, planes, stride)
        else:
            if planes < 512:
                self.conv2 = convtranspose4x4(planes, planes, stride=2)
            else:
                self.conv2 = convtranspose2x2(planes, planes)
                
        self.bn2 = BatchNorm2d(planes)
        
        self.conv3      = convtranspose1x1(planes * self.expansion, planes)
        self.bn3        = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, X):
        
        out = self.conv3(X)
        out = self.bn3(out)
        out = _RELU(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = _RELU(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity
        out = _RELU(out)

        return out

#-------------------------------
# Encoder architecture layers
#-------------------------------

class EresNet(Module):

    def __init__(self, block, layers, num_input_channels, num_latent_dims, zero_init_residual=False):
        
        super().__init__()
        
        self.inplanes = 64
        
        self.conv1 = Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = BatchNorm2d(16)
        
        self.conv2 = Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = BatchNorm2d(64)
        
        self.layer0 = EresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion
        self.bool_deep = False
        
        self.conv3 = Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        self.bn3   = BatchNorm2d(self.unroll_size)
        
        for m in self.modules():
            if isinstance(m, EresNetBottleneck):
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
                if isinstance(m, EresNetBottleneck):
                    constant_(m.bn3.weight, 0)
                elif isinstance(m, EresNetBasicBlock):
                    constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 512:
                downsample = Sequential(
                    conv4x4(self.inplanes, planes * block.expansion, stride),
                    BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = Sequential(
                    conv2x2(self.inplanes, planes * block.expansion),
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
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = _RELU(x)
        
        x = x.view(x.size(0), -1)
        
        x = _RELU(self.fc1(x))
        if self.bool_deep:
            x = _RELU(self.fc2(x))
        
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
        
        self.layer0 = DresNetBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion
        self.bool_deep = False
        
        self.conv3 = ConvTranspose2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        self.bn3   = BatchNorm2d(self.unroll_size)
        
        for m in self.modules():
            if isinstance(m, DresNetBottleneck):
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
                if isinstance(m, DresNetBottleneck):
                    constant_(m.bn3.weight, 0)
                elif isinstance(m, DresNetBasicBlock):
                    constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or planes:
            if planes < 512:
                downsample = Sequential(
                    convtranspose4x4(planes * block.expansion, self.inplanes, stride),
                    BatchNorm2d(self.inplanes),
                )
            else:
                downsample = Sequential(
                    convtranspose2x2(planes * block.expansion, self.inplanes),
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


#-------------------------------------------------------
# Initializers for model encoders with various depths
#-------------------------------------------------------

def eresnet18(**kwargs):
    """Constructs a EresNet-18 model encoder.
    """
    return EresNet(EresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def eresnet34(**kwargs):
    """Constructs a EresNet-34 model encoder.
    """
    return EresNet(EresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def eresnet50(**kwargs):
    """Constructs a EresNet-50 model encoder.
    """
    return EresNet(EresNetBottleneck, [3, 4, 6, 3], **kwargs)

def eresnet101(**kwargs):
    """Constructs a EresNet-101 model encoder.
    """
    return EresNet(EresNetBottleneck, [3, 4, 23, 3], **kwargs)

def eresnet152(**kwargs):
    """Constructs a ErsNet-152 model encoder.
    """
    return EresNet(EresNetBottleneck, [3, 8, 36, 3], **kwargs)

#-------------------------------------------------------
# Initializers for model decoders with various depths
#-------------------------------------------------------

def dresnet18(**kwargs):
    """Constructs a EresNet-18 model decoder.
    """
    return DresNet(DresNetBasicBlock, [2, 2, 2, 2], **kwargs)

def dresnet34(**kwargs):
    """Constructs a EresNet-34 model decoder.
    """
    return DresNet(DresNetBasicBlock, [3, 4, 6, 3], **kwargs)

def dresnet50(**kwargs):
    """Constructs a EresNet-50 model encoder.
    """
    return DresNet(DresNetBottleneck, [3, 4, 6, 3], **kwargs)

def dresnet101(**kwargs):
    """Constructs a EresNet-101 model decoder.
    """
    return DresNet(DresNetBottleneck, [3, 4, 23, 3], **kwargs)

def dresnet152(**kwargs):
    """Constructs a ErsNet-152 model decoder.
    """
    return DresNet(DresNetBottleneck, [3, 8, 36, 3], **kwargs)