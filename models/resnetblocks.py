"""
resnetblocks.py

PyTorch implementation of ResNet-style architecture layers to be used by the encoder and decoder
modules in the discriminative and variational inference models
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU
from torch.nn.init import kaiming_normal_, constant_
import pdb

# Global variables
_RELU = ReLU()

# -------------------------------w
# Encoder Conv2d layers
# -------------------------------

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

# -------------------------------
# Decoder ConvTranspose2d layers
# -------------------------------

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

# -------------------------------
# ResNet encoder block layers
# -------------------------------

class EresNetBasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        if downsample is None:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            if planes < 128:
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

# ResNet block that performs only 1x1 convolution
class EresNetMiniBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        if downsample is None:
            self.conv1 = conv1x1(inplanes, planes, stride)
            if inplanes != planes:
                downsample = Sequential(
                    conv1x1(inplanes, planes, stride),
                    BatchNorm2d(planes),
                )
        else:
            if planes < 512:
                self.conv1 = conv1x1(inplanes, planes, stride=1)
            else:
                self.conv1 = conv1x1(inplanes, planes)
            
        self.bn1        = BatchNorm2d(planes)
        self.conv2      = conv1x1(planes, planes)
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


# -------------------------------
# ResNet decoder block layers
# -------------------------------

class DresNetBasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        
        if downsample is None:
            self.conv1 = convtranspose3x3(planes, inplanes, stride)
        else:
            if planes < 128:
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
