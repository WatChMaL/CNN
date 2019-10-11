"""
enet.py

Block implementation of different autoencoder componentss i.e. encoders, decoders, bottlenecks etc.

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import randn, randn_like, tensor, zeros, eye
from torch import mean, sum, cat

# WatChMaL imports
from models import elenet, eresnet

# Global dictionaries for model setup
variant_dict = {0:"AE", 1:"VAE", 2:"PURE_CL", 3:"M2", 4:"M1_M2"}
train_dict = {0:"train_all", 1:"train_ae_or_vae_only", 2:"train_bottleneck_only", 3:"train_pure_cl_only", 4:"train_M1", 5:"train_M2", 6:"train_M1_M2"}
model_dict = {0:"EleNet", 1:"EresNet"}

modules = ["encoder", "decoder", "bottleneck", "classifier", "regressor"]
weight_ignore_dict = {"train_all":modules,
                      "train_ae_or_vae_only":["classifier", "regressor"],
                      "train_bottleneck_only":["bottleneck", "classifier", "regressor"],
                      "train_pure_cl_only":modules,
                      "train_M1":["classifier", "regressor", "decoder"],
                      "train_M2":modules,
                      "train_M1_M2":["classifier", "regressor", "decoder", "bottleneck"]}

# Enet class
class ENet(nn.Module):
    
    # Initialize
    def __init__(self, num_input_channels=38, num_latent_dims=64, num_classes=3, variant_key=0, train_key=1, model_key=0):
        
        assert variant_key in variant_dict.keys()
        assert train_key in train_dict.keys()
        assert model_key in train_dict.keys()
        
        super(ENet, self).__init__()
        
        # Class attributess
        self.variant = variant_dict[variant_key]
        self.train_type = train_dict[train_key]
        self.model_type = model_dict[model_key]
        self.weight_ignore_dict = weight_ignore_dict
        
        # Add the layer blocks
        if self.model_type == "EleNet":
            self.encoder, self.decoder = lenet.lenet18(num_input_channels=num_input_channels, num_latent_dims=num_latent_dims)
        elif self.model_type == "EresNet":
            self.encoder, self.decoder = eresnet.eresnet34(num_input_channels=num_input_channels, num_latent_dims=num_latent_dims)
        
        self.regressor = Regressor(num_latent_dims, num_classes)
        self.classifier = Classifier(num_latent_dims, num_classes)
        
        # Add the desired bottleneck
        if self.variant == "AE":
            self.bottleneck = AEBottleneck(num_latent_dims)
        elif self.variant == "VAE":
            self.bottleneck = VAEBottleneck(num_latent_dims)
        elif self.variant == "PURE_CL":
            self.bottleneck = None
        elif (self.variant == "M2" or self.variant == "M1_M2"):
            self.bottleneck = M2Bottleneck(num_latent_dims, num_classes)
        
        # Set params.require_grad = False for the appropriate block of the model
        if self.train_type != "train_all":
            
            if self.train_type == "train_ae_or_vae_only" or self.train_type == "train_M2":
                
                # Set require_grad = False for classifier parameters
                for param in self.classifier.parameters():
                    param.requires_grad = False
                    
                # Set require_grad = False for Regressor parameters
                for param in self.regressor.parameters():
                    param.requires_grad = False
                    
            elif self.train_type == "train_pure_cl_only":
                
                # Set require_grad = False for decoder parameters
                for param in self.decoder.parameters():
                    param.requires_grad = False
                    
                # Set require_grad = False for Regressor parameters
                for param in self.regressor.parameters():
                    param.requires_grad = False
                    
            elif self.train_type == "train_M1_M2":
                
                # Set require_grad = False for encoder parameters
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                # Set require_grad = False for classifier parameters
                for param in self.classifier.parameters():
                    param.requires_grad = False
                    
                # Set require_grad = False for Regressor parameters
                for param in self.regressor.parameters():
                    param.requires_grad = False
                    
            elif self.train_type == "train_M2":
                
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
                
                if self.train_type == "train_bottleneck_only":
                    
                    assert self.variant != "PURE_CL"
                
                    # Set require_grad = False for classifier parameters
                    for param in self.classifier.parameters():
                        param.requires_grad = False
                        
                    # Set require_grad = False for Regressor parameters
                    for param in self.regressor.parameters():
                        param.requires_grad = False
                        
                elif self.train_type == "train_M1":
                
                    # Set require_grad = False for bottleneck parameters
                    for param in self.bottleneck.parameters():
                        param.requires_grad = False
                        
    # Forward
    def forward(self, X, mode, labels=None):
        if mode == "sample":
            assert self.variant == "VAE"
            z = self.bottleneck(X, mode)
            return self.decoder(z, None), self.classifier(z), self.regressor(z)
        elif mode == "decode":
            return self.decoder(X, None), self.classifier(X), self.regressor(X)
        else:
            z_prime = self.encoder(X)

            if self.variant == "AE":
                z = self.bottleneck(z_prime)
            elif self.variant == "VAE":
                z, mu, logvar = self.bottleneck(z_prime, None)
            elif self.variant == "M2" or self.variant == "M1_M2":
                z_y, z, mu, logvar, pi = self.bottleneck(z_prime, None, labels=labels)
            elif self.variant == "PURE_CL":
                return self.classifier(z_prime)
            
            if mode == "generate_latents":
                if self.variant == "AE":
                    return z
                elif self.variant == "VAE":
                    return z, mu, logvar
            elif mode == "M1":
                return self.classifier(z), self.regressor(z)
            elif mode == "ae_or_vae":
                if self.variant == "AE":
                    return self.decoder(z, None)
                elif self.variant == "VAE":
                    return self.decoder(z, None), z, mu, logvar, z_prime
            elif mode == "M2" or mode == "M1_M2":
                return self.decoder(z_y, None), z, mu, logvar, z_prime, pi
            elif mode == "all":
                if self.variant == "AE":
                    return self.decoder(z, None), self.classifier(z), self.regressor(z)
                elif self.variant == "VAE":
                    return self.decoder(z, None), z, mu, logvar, z_prime, self.classifier(z), self.regressor(z)