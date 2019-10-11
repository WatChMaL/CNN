"""
enfnet.py

Block implementation of different autoencoder componentss i.e. encoders, decoders, bottlenecks etc.
in addition to normalizing flows - planar and radial

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import randn, randn_like, tensor, zeros
from torch import device
from torch import mean

# WatChMaL imports
from models import enet
from models import nf

# Global variables
variant_dict = {0:"AE", 1:"VAE", 2:"NF"}
train_dict = {0:"train_all", 1:"train_nf_only", 2:"train_bottleneck_only", 3:"train_cl_or_rg_only"}
flow_dict = {0:"planar", 1:"radial", 2:"sylvester"}

# EnfNet class
class EnfNet(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels=19, num_latent_dims=64, num_classes=3, variant_key=2, train_key=1, flow_depth=10, flow_key=0, device_id=0):
        assert variant_key in variant_dict.keys()
        assert train_key in train_dict.keys()
        assert device_id >= 0
        assert device_id <= 7
        
        super(EnfNet, self).__init__()
        
        # Initialization variables
        device="cuda:" + str(device_id)
        
        # Class attributess
        self.variant = variant_dict[variant_key]
        self.train_type = train_dict[train_key]
        
        # Add the layer blocks
        self.encoder = enet.Encoder(num_input_channels, num_latent_dims)
        self.decoder = enet.Decoder(num_input_channels, num_latent_dims)
        self.classifier = enet.Classifier(num_latent_dims, num_classes)
        self.regressor = enet.Regressor(num_latent_dims, num_classes)
        
        # Add the desired bottleneck
        if self.variant is "AE":
            self.bottleneck = enet.AEBottleneck(num_latent_dims)
        elif self.variant is "VAE":
            self.bottleneck = enet.VAEBottleneck(num_latent_dims)
        elif self.variant is "NF":
            self.vae_bottleneck = enet.VAEBottleneck(num_latent_dims)
            self.nf_bottleneck = NFBottleneck(num_latent_dims, flow_depth, flow_dict[flow_key], device) 
        
        # Set params.require_grad = False for the appropriate block of the model
        if self.train_type is not "train_all":
            
            if self.train_type is "train_nf_only":
                
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
                
                    # Set require_grad = False for encoder parameters
                    if self.bottleneck is not None:
                        for param in self.bottleneck.parameters():
                            param.requires_grad = False
                    else:
                        for param in self.vae_bottleneck.parameters():
                            param.requires_grad = False
                            
                        for param in self.nf_bottleneck.parameters():
                            param.requires_grad = False
                                          
    # Forward
    def forward(self, X, mode, device):
        # Sample from the latent space, pass the sample through the flow if specified
        # and decode to generate a reconstruction
        if mode is "sample":
            assert self.variant is "VAE" or self.variant is "NF"
            if self.variant is "VAE":
                z = self.bottleneck(X, mode, device)
                return self.decoder(z, None), self.classifier(z), self.regressor(z)
            elif self.variant is "NF":
                z = self.vae_bottleneck(X, mode, device)
                z_k, _ = self.nf_bottleneck(z, None, mode)
                return self.decoder(z_k, None), self.classifier(z_k), self.regressor(z_k)
        
        # Decode the latent vector passed and generate a reconstruction
        elif mode is "decode":
            return self.decoder(X, None), self.classifier(X), self.regressor(X)
        
        #  Forward pass using different modes
        else:
            # Compute the encoded latent vector
            z_prime = self.encoder(X)

            # Compute the stochastic latent vector
            if self.variant is "AE":
                z = self.bottleneck(z_prime)
            elif self.variant is "VAE":
                z, mu, logvar = self.bottleneck(z_prime, None, device)
            elif self.variant is "NF":
                z, mu, logvar = self.vae_bottleneck(z_prime, None, device)
                z_k, log_det = self.nf_bottleneck(z, z_prime, None)
            if mode is "generate_latents":
                if self.variant is "AE":
                    return z
                elif self.variant is "VAE":
                    return z, mu, logvar
                elif self.variant is "NF":
                    return z_k, z, mu, logvar
                
            elif mode is "cl_or_rg":
                if self.variant in ["AE", "VAE"]:
                    return self.classifier(z), self.regressor(z)
                elif self.variant is "NF":
                    return self.classifier(z_k), self.regressor(z_k)
            
            elif mode is "ae_or_vae":
                if self.variant is "AE":
                    return self.decoder(z, self.encoder.unflat_size)
                elif self.variant is "VAE":
                    return self.decoder(z, self.encoder.unflat_size), z, mu, logvar, z_prime
                
            elif mode is "nf":
                assert self.variant is "NF"
                return self.decoder(z_k, self.encoder.unflat_size), z_k, log_det, z, mu, logvar, z_prime
                
            elif mode is "all":
                if self.variant is "AE":
                    return self.decoder(z, self.encoder.unflat_size), self.classifier(z), self.regressor(z)
                elif self.variant is "VAE":
                    return self.decoder(z, self.encoder.unflat_size), z, mu, logvar, z_prime, self.classifier(z), self.regressor(z)
                elif self.variant is "NF":
                    return self.decoder(z_k, self.encoder.unflat_size), z_k, z, mu, logvar, z_prime, log_det, self.classifier(z_k), self.regressor(z_k)
                
# NFBottleneck - Normalizing flow bottleneck
class NFBottleneck(nn.Module):
   
    # Initializer
    def __init__(self, num_latent_dims, flow_depth, flow_type, device):
        super(NFBottleneck, self).__init__()
        
        # Initialize bottleneck attributes
        self.num_latent_dims = num_latent_dims
        self.flow_depth = flow_depth
        self.device = device
        self.flow_type = flow_type
        
        # Declare the flow specfied
        if flow_type is "planar":
            flow = nf.Planar
        elif flow_type is "radial":
            flow = nf.Radial
            
        # Initialize and add the flows to the bottleneck
        for k in range(self.flow_depth):
            flow_k = flow(self.num_latent_dims, device)
            self.add_module('flow_' + str(k), flow_k)
            
    # Forward
    def forward(self, X, h, mode=None, device=None):
        
        # Reshape the reparameterized latent vector X = z_0
        z = X.view(X.size(0), X.size(1), -1)
        
        # Initialize the reference point for radial flows
        if self.flow_type is "radial":
            z_0 = z
        
        # Initialize the logdet jacobians tensor
        log_det_jacobians = zeros(self.flow_depth, z.size(0), device=self.device)
        
        # Transform the latent vector through a series
        # of invertible flows and compute the log det
        # of the jacobian of the transformations
        for k in range(self.flow_depth):
            flow_k = getattr(self, 'flow_' + str(k))
            if self.flow_type is "planar":
                z, log_det_jacobians[k] = flow_k(z, h)
            elif self.flow_type is "radial":
                z, log_det_jacobians[k] = flow_k(z, z_0)
            
        # Reshape the transformed latent vector into 2 dims
        z = z.view(z.size(0), z.size(1))
        
        return z, log_det_jacobians