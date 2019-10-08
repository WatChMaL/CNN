"""
eresnet.py

PyTorch implementation of ResNet-style architecture to be used as an encoder and decoder in the variational
inference models with a corresponding symmetric decoder.
"""

import torch.nn as nn

__all__ = ['Encoder', 'Decoder', 'encoder18', 'encoder34',
           'encoder50', 'encoder101', 'encoder152',
           'decoder18', 'decoder34', 'decoder50',
           'decoder101', 'decoder152']

#-------------------------------
# Encoder Conv2d layers
#-------------------------------

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv2x2(in_planes, out_planes, stride=1):
    """2x2 convoltion"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv4x4(in_planes, out_planes, stride=1):
    """4x4 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride,
                     padding=1, bias=False)

#-------------------------------
# Decoder ConvTranspose2d layers
#-------------------------------

def convtranspose1x1(in_planes, out_planes, stride=1):
    """1x1 transposed convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convtranspose2x2(in_planes, out_planes, stride=1):
    """2x2 transposed convoltion"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)
    
def convtranspose3x3(in_planes, out_planes, stride=1):
    """3x3 transposed convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)

def convtranspose4x4(in_planes, out_planes, stride=1):
    """4x4 transposed convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                              padding=1, bias=False)


#-------------------------------
# ResNet encoder block layers
#-------------------------------

class EncoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBasicBlock, self).__init__()
        
        if downsample is None:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            if planes < 512:
                self.conv1 = conv4x4(inplanes, planes, stride=2)
            else:
                self.conv1 = conv2x2(inplanes, planes)
            
        self.bn1 = nn.BatchNorm2d(planes)    
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity        
        out = self.relu(out)

        return out


class EncoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBottleneck, self).__init__()
        
        self.conv1 = conv1x1(inplanes, planes)            
        self.bn1 = nn.BatchNorm2d(planes)
        
        if downsample is None:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            if planes < 512:
                self.conv2 = conv4x4(planes, planes, stride=2)
            else:
                self.conv2 = conv2x2(planes, planes)
                
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity
        out = self.relu(out)

        return out
    
#-------------------------------
# ResNet decoder block layers
#-------------------------------

class DecoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DecoderBasicBlock, self).__init__()
        
        if downsample is None:
            self.conv1 = convtranspose3x3(planes, inplanes, stride)
        else:
            if planes < 512:
                self.conv1 = convtranspose4x4(planes, inplanes, stride=2)
            else:
                self.conv1 = convtranspose2x2(planes, inplanes)
            
        self.bn1 = nn.BatchNorm2d(inplanes)    
        self.relu = nn.ReLU()
        
        self.conv2 = convtranspose3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, X):
        out = self.conv2(X)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity        
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DecoderBottleneck, self).__init__()
        
        self.conv1 = convtranspose1x1(planes, inplanes)            
        self.bn1 = nn.BatchNorm2d(inplanes)
        
        if downsample is None:
            self.conv2 = convtranspose3x3(planes, planes, stride)
        else:
            if planes < 512:
                self.conv2 = convtranspose4x4(planes, planes, stride=2)
            else:
                self.conv2 = convtranspose2x2(planes, planes)
                
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = convtranspose1x1(planes * self.expansion, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, X):
        
        out = self.conv3(X)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        
        if self.downsample is not None:
            identity = self.downsample(X)
        else:
            identity = X

        out += identity
        out = self.relu(out)

        return out

#-------------------------------
# Encoder architecture layers
#-------------------------------

class Encoder(nn.Module):

    def __init__(self, block, layers, num_input_channels=19, num_latent_dims=64, zero_init_residual=False):
        
        super(Encoder, self).__init__()
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        
        self.layer0 = EncoderBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion
        self.bool_deep = False
        
        self.conv3 = nn.Conv2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(self.unroll_size)
        
        for m in self.modules():
            if isinstance(m, EncoderBottleneck):
                self.fc1 = nn.Linear(self.unroll_size, int(self.unroll_size/2))
                self.fc2 = nn.Linear(int(self.unroll_size/2), num_latent_dims)
                self.bool_deep = True
                break
                
        if not self.bool_deep:
            self.fc1 = nn.Linear(self.unroll_size, num_latent_dims)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, EncoderBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, EncoderBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes < 512:
                downsample = nn.Sequential(
                    conv4x4(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv2x2(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        if self.bool_deep:
            x = self.relu(self.fc2(x))
        
        return x
    
#-------------------------------
# Decoder architecture layers
#-------------------------------
class Decoder(nn.Module):

    def __init__(self, block, layers, num_input_channels=19, num_latent_dims=64, zero_init_residual=False):
        
        super(Decoder, self).__init__()
        
        self.inplanes = 64
        
        self.conv1 = nn.ConvTranspose2d(16, num_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv2 = nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.relu = nn.ReLU()
        
        self.layer0 = DecoderBasicBlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.unroll_size = 512 * block.expansion
        self.bool_deep = False
        
        self.conv3 = nn.ConvTranspose2d(self.unroll_size, self.unroll_size, kernel_size=(1,4), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(self.unroll_size)
        
        for m in self.modules():
            if isinstance(m, DecoderBottleneck):
                self.fc2 = nn.Linear(num_latent_dims, int(self.unroll_size/2))
                self.fc1 = nn.Linear(int(self.unroll_size/2), self.unroll_size)
                self.bool_deep = True
                break
                
        if not self.bool_deep:
            self.fc1 = nn.Linear(num_latent_dims, self.unroll_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DecoderBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, DecoderBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or planes:
            if planes < 512:
                downsample = nn.Sequential(
                    convtranspose4x4(planes * block.expansion, self.inplanes, stride),
                    nn.BatchNorm2d(self.inplanes),
                )
            else:
                downsample = nn.Sequential(
                    convtranspose2x2(planes * block.expansion, self.inplanes),
                    nn.BatchNorm2d(self.inplanes),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        layers.reverse()
        return nn.Sequential(*layers)

    def forward(self, x, unflat_size):
        if self.bool_deep:
            x = self.relu(self.fc2(x))
            
        x = self.relu(self.fc1(x))
        
        # Figure out how to properly reshape
        x = x.view(unflat_size) if unflat_size is not None else x.view(x.size(0), -1, 1, 1)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.layer0(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.relu(self.conv1(x))

        return x


#-------------------------------------------------
# Initializer for the models with various depths
#-------------------------------------------------

def eresnet18(**kwargs):
    """Constructs a EresNet-18 model.
    """
    return Encoder(EncoderBasicBlock, [2, 2, 2, 2], **kwargs), Decoder(DecoderBasicBlock, [2, 2, 2, 2], **kwargs)

def eresnet34(**kwargs):
    """Constructs a EresNet-34 model.
    """
    return Encoder(EncoderBasicBlock, [3, 4, 6, 3], **kwargs), Decoder(DecoderBasicBlock, [3, 4, 6, 3], **kwargs)


def eresnet50(**kwargs):
    """Constructs a EresNet-50 model.
    """
    return Encoder(EncoderBottleneck, [3, 4, 6, 3], **kwargs), Decoder(DecoderBottleneck, [3, 4, 6, 3], **kwargs)


def eresnet101(**kwargs):
    """Constructs a EresNet-101 model.
    """
    return Encoder(EncoderBottleneck, [3, 4, 23, 3], **kwargs), Decoder(DecoderBottleneck, [3, 4, 23, 3], **kwargs)

def eresnet152(**kwargs):
    """Constructs a ErsNet-152 model.
    """
    return Encoder(EncoderBottleneck, [3, 8, 36, 3], **kwargs), Decoder(DecoderBottleneck, [3, 8, 36, 3], **kwargs)