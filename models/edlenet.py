"""
edlenet.py

PyTorch implementation of LeNet-style architecture to be used as an encoder and decoder
in the variational inference models.
"""

from torch.nn import Linear, Conv2d, ConvTranspose2d, ReLU, Module

# Global variables
__all__ = ['elenet9', 'dlenet9']
_RELU = ReLU()

# Encoder class
class EleNet(Module):
    
    def __init__(self, num_input_channels, num_latent_dims):
        super().__init__()
        self.unflat_size = None
        
        # Feature extraction
        self.en_conv1a  = Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv1b  = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv2 = Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature extraction
        self.en_conv3a = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv4  = Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Fully-connected layers
        self.en_fc1 = Linear(5120, 1024)
        self.en_fc2 = Linear(1024, 512)
        """
        self.en_fc3 = Linear(512, num_latent_dims)
        """
        self.en_fc3 = Linear(512, 256)
        self.en_fc4 = Linear(256, num_latent_dims)
        
        
    def forward(self, X):
        """
        x = _RELU(self.en_conv1a(X))
        x = _RELU(self.en_conv1b(x))
        x = _RELU(self.en_conv2(x))
        x = _RELU(self.en_conv3a(x))
        x = _RELU(self.en_conv3b(x))
        x = _RELU(self.en_conv4(x))
        
        self.unflat_size = x.size()
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        
        x = _RELU(self.en_fc1(x))
        x = _RELU(self.en_fc2(x))
        x = _RELU(self.en_fc3(x))
        
        return _RELU(self.en_fc4(x))
        """
        x = self.en_conv1a(X)
        x = _RELU(self.en_conv1b(x))
        x = _RELU(self.en_conv2(x))
        x = self.en_conv3a(x)
        x = _RELU(self.en_conv3b(x))
        x = self.en_conv4(x)
        
        self.unflat_size = x.size()
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        
        x = _RELU(self.en_fc1(x))
        x = _RELU(self.en_fc2(x))
        x = _RELU(self.en_fc3(x))
        
        return _RELU(self.en_fc4(x))
        
    
# Decoder class
class DleNet(Module):
    
    # Initialize
    def __init__(self, num_input_channels, num_latent_dims):
        super().__init__()
        
        # Fully connected layers
        """
        self.de_fc3 = Linear(num_latent_dims, 512)
        """
        self.de_fc4 = Linear(num_latent_dims, 256)
        self.de_fc3 = Linear(256, 512)
        self.de_fc2 = Linear(512, 1024)
        self.de_fc1 = Linear(1024, 5120)
        
        # Upsampling de-convolution
        self.de_conv4  = ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv3b = ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        # Upsampling de-convolution
        self.de_conv2 = ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv1b = ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1a = ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward
    def forward(self, X, unflat_size):
        """
        x = _RELU(self.de_fc4(X))
        x = _RELU(self.de_fc3(x))
        x = _RELU(self.de_fc2(x))
        x = _RELU(self.de_fc1(x))
        
        x = x.view(unflat_size) if unflat_size is not None else x.view(-1, 128, 4, 10)
        
        x = _RELU(self.de_conv4(x))
        x = _RELU(self.de_conv3b(x))
        x = _RELU(self.de_conv3a(x))
        x = _RELU(self.de_conv2(x))
        x = _RELU(self.de_conv1b(x))
        x = _RELU(self.de_conv1a(x))

        return x
        """
        
        x = _RELU(self.de_fc4(X))
        x = _RELU(self.de_fc3(x))
        x = _RELU(self.de_fc2(x))
        x = _RELU(self.de_fc1(x))
        
        x = x.view(unflat_size) if unflat_size is not None else x.view(-1, 128, 4, 10)
        
        x = _RELU(self.de_conv4(x))
        x = _RELU(self.de_conv3b(x))
        x = self.de_conv3a(x)
        x = _RELU(self.de_conv2(x))
        x = self.de_conv1b(x)
        x = _RELU(self.de_conv1a(x))

        return x
    
#-------------------------------------------------
# Initializer for the models with various depths
#-------------------------------------------------

def elenet9(**kwargs):
    """Constructs a LeNet style encoder.
    """
    return EleNet(**kwargs)

def dlenet9(**kwargs):
    """Construct a LeNet style decoder.
    """
    return DleNet(**kwargs)