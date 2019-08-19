"""
enet.py

Block implementation of different autoencoder componentss i.e. encoders, decoders, bottlenecks etc.

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import randn, randn_like, tensor, zeros
from torch import device
from torch import mean

# WatChMaL imports
from models import resnet

# Global variables
variant_dict = {0:"AE", 1:"VAE"}
train_dict = {0:"train_all", 1:"train_ae_or_vae_only", 2:"train_bottleneck_only", 3:"train_cl_or_rg_only"}

# Enet class
class ENet(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels=38, num_latent_dims=64, num_classes=3, variant_key=0, train_key=1):
        assert variant_key in variant_dict.keys()
        assert train_key in train_dict.keys()
        
        super(ENet, self).__init__()
        
        # Class attributess
        self.variant = variant_dict[variant_key]
        self.train_type = train_dict[train_key]
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Add the layer blocks
        #self.encoder = Encoder(num_input_channels, num_latent_dims)
        self.encoder = resnet.resnet101(num_input_channels=num_input_channels,
                                        num_classes=num_latent_dims)
        self.decoder = Decoder(num_input_channels, num_latent_dims)
        self.classifier = Classifier(num_latent_dims, num_classes)
        self.regressor = Regressor(num_latent_dims, num_classes)
        
        # Add the desired bottleneck
        if self.variant is "AE":
            self.bottleneck = AEBottleneck(num_latent_dims)
        elif self.variant is "VAE":
            self.bottleneck = VAEBottleneck(num_latent_dims)
        
        # Set params.require_grad = False for the appropriate block of the model
        if self.train_type is not "train_all":
            
            if self.train_type is "train_ae_or_vae_only":
                
                # Set require_grad = False for classifier parameters
                for param in self.classifier.parameters():
                    param.requires_grad = False
                    
                # Set require_grad = False for Regressor parameters
                for param in self.regressor.parameters():
                    param.requires_grad = False
                
            else:
            
                # Set require_grad = False for encoder parameters
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
                # Set require_grad = False for decoder parameters
                for param in self.decoder.parameters():
                    param.requires_grad = False
                
                if self.train_type is "train_bottleneck_only":
                
                    # Set require_grad = False for classifier parameters
                    for param in self.classifier.parameters():
                        param.requires_grad = False
                        
                    # Set require_grad = False for Regressor parameters
                    for param in self.regressor.parameters():
                        param.requires_grad = False
                    
            
                elif self.train_type is "train_cl_or_rg_only":
                
                    # Set require_grad = False for bottleneck parameters
                    for param in self.bottleneck.parameters():
                        param.requires_grad = False
                        
    # Forward
    def forward(self, X, mode, device):
        if mode is "sample":
            assert self.variant is "VAE"
            z = self.bottleneck(X, mode, device)
            return self.decoder(z, None), self.classifier(z), self.regressor(z)
        elif mode is "decode":
            return self.decoder(X, None), self.classifier(X), self.regressor(X)
        else:
            z_prime = self.encoder(X)

            if self.variant is "AE":
                z = self.bottleneck(z_prime)
            elif self.variant is "VAE":
                z, mu, logvar = self.bottleneck(z_prime, None, device)
            if mode is "generate_latents":
                if self.variant is "AE":
                    return z
                elif self.variant is "VAE":
                    return z, mu, logvar
            elif mode is "cl_or_rg":
                return self.classifier(z), self.regressor(z)
            elif mode is "ae_or_vae":
                if self.variant is "AE":
                    return self.decoder(z, None)
                elif self.variant is "VAE":
                    return self.decoder(z, None), z, mu, logvar, z_prime
            elif mode is "all":
                if self.variant is "AE":
                    return self.decoder(z, None), self.classifier(z), self.regressor(z)
                elif self.variant is "VAE":
                    return self.decoder(z, None), z, mu, logvar, z_prime, self.classifier(z), self.regressor(z)
                
                
# Encoder class
class Encoder(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels, num_latent_dims):
        super(Encoder, self).__init__()
        self.unflat_size = None
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Feature extraction
        self.en_conv1a  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature extraction
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsampling
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(5120, 1024)
        self.en_fc2 = nn.Linear(1024, 512)
        self.en_fc3 = nn.Linear(512, 256)
        self.en_fc4 = nn.Linear(256, num_latent_dims)
        
    # Forward
    def forward(self, X):
        x = self.en_conv1a(X)
        x = self.relu(self.en_conv1b(x))
        
        x = self.relu(self.en_conv2(x))
        
        x = self.en_conv3a(x)
        x = self.relu(self.en_conv3b(x))
        
        x = self.en_conv4(x)
        
        self.unflat_size = x.size()
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        x = self.relu(self.en_fc3(x))
        x = self.relu(self.en_fc4(x))
        
        return x
    
# Decoder class
class Decoder(nn.Module):
    
    # Initialize
    def __init__(self, num_output_channels, num_latent_dims):
        super(Decoder, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.de_fc4 = nn.Linear(num_latent_dims, 256)
        self.de_fc3 = nn.Linear(256, 512)
        self.de_fc2 = nn.Linear(512, 1024)
        self.de_fc1 = nn.Linear(1024, 5120)
        
        # Upsampling de-convolution
        self.de_conv4  = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_conv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        # Upsampling de-convolution
        self.de_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Feature mapping de-convolution
        self.de_conv1b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_conv1a = nn.ConvTranspose2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)
        
    # Forward
    def forward(self, X, unflat_size):
        
        x = self.relu(self.de_fc4(X))
        x = self.relu(self.de_fc3(x))
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc1(x))
        
        x = x.view(unflat_size) if unflat_size is not None else x.view(-1, 128, 4, 10)
        
        x = self.relu(self.de_conv4(x))
        
        x = self.relu(self.de_conv3b(x))
        x = self.de_conv3a(x)
        
        x = self.relu(self.de_conv2(x))
        
        x = self.de_conv1b(x)
        x = self.relu(self.de_conv1a(x))

        return x
    
# AEBottleneck
class AEBottleneck(nn.Module):
    
    # Initialize
    def __init__(self, num_latent_dims):
        super(AEBottleneck, self).__init__()
        self.num_latent_dims = num_latent_dims
        
        # Activation functions
        self.relu = nn.ReLU()
        
    # Forward
    def forward(self, X):
        return X
        
    
# VAEBottleneck
class VAEBottleneck(nn.Module):
    
    # Initialize
    def __init__(self, num_latent_dims):
        super(VAEBottleneck, self).__init__()
        self.num_latent_dims = num_latent_dims
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # VAE distribution parameter layers
        self.en_mu = nn.Linear(num_latent_dims, num_latent_dims)
        self.en_var = nn.Linear(num_latent_dims, num_latent_dims)
        
    # Forward
    def forward(self, X, mode, device, shots=1):
        if mode is "sample":
            z_samples = randn((shots, X.size(0), self.num_latent_dims), device=device)
            return mean(z_samples, 0)
        else:
            mu, logvar = self.en_mu(X), self.en_var(X)
            
            # Reparameterization trick
            std = logvar.mul(0.5).exp()
            eps = std.new(std.size()).normal_()
            z = eps.mul(std).add(mu)

            return z, mu, logvar
        
# Latent Classifier
class Classifier(nn.Module):
    
    # Initializer
    def __init__(self, num_latent_dims=64, num_classes=3):
        super(Classifier, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Classifier fully connected layers
        self.cl_fc1 = nn.Linear(num_latent_dims, int(num_latent_dims/2))
        self.cl_fc2 = nn.Linear(int(num_latent_dims/2), int(num_latent_dims/4))
        self.cl_fc3 = nn.Linear(int(num_latent_dims/4), int(num_latent_dims/8))
        self.cl_fc4 = nn.Linear(int(num_latent_dims/8), num_classes)
        
    # Forward
    def forward(self, X):
        
        # Fully-connected layers
        x = self.relu(self.cl_fc1(X))
        x = self.relu(self.cl_fc2(x))
        x = self.relu(self.cl_fc3(x))
        x = self.cl_fc4(x)
        
        return x
    
# Latent regressor
class Regressor(nn.Module):
    
    # Initializer
    def __init__(self, num_latent_dims=64, num_classes=3):
        super(Regressor, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Classifier fully connected layers
        self.rg_fc1 = nn.Linear(num_latent_dims, int(num_latent_dims/2))
        self.rg_fc2 = nn.Linear(int(num_latent_dims/2), int(num_latent_dims/4))
        self.rg_fc3 = nn.Linear(int(num_latent_dims/4), int(num_latent_dims/8))
        self.rg_fc4 = nn.Linear(int(num_latent_dims/8), 1)
        
    # Forward
    def forward(self, X):
        
        # Fully-connected layers
        x = self.relu(self.rg_fc1(X))
        x = self.relu(self.rg_fc2(x))
        x = self.relu(self.rg_fc3(x))
        x = self.relu(self.rg_fc4(x))
        
        return x