"""
loss_funcs.py

Module with implementations of different loss functions useful for autoencoder training

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import sum
from torch import mean

# Global variables
reconstruction_loss = nn.MSELoss(reduction="sum")
cl_loss = nn.CrossEntropyLoss()

# VAE generic loss function i.e. RECON Loss + KL Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss
def VAELoss(recon, data, mu, log_var):

    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    return recon_loss + kl_loss, recon_loss, kl_loss


# VAE+Classifier+Regressor loss function i.e. RECON Loss + KL Loss + CE Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss, CE (cross-entropy) loss, MSE (mean-squared) loss
def VAECLRGLoss(recon, data, mu, log_var, predicted_label, label, predicted_energy, energy):
    
    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = cl_loss(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / data.size(0)
    
    return recon_loss + kl_loss + ce_loss + mse_loss, recon_loss, kl_loss, ce_loss, mse_loss

# AE+Classifier loss funcito i.e. RECON Loss + CE Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss
def AECLRGLoss(recon, data, prediction, label):
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = cl_loss(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / data.size(0)
    
    return recon_loss + ce_loss + mse_loss, recon_loss, ce_loss, mse_loss

# Classifier+Regressor loss function i.e. CE Loss + MSE Loss
# Returns : Tuple of total loss, CE (cross-entropy) loss, MSE (mean-squared) loss
def CLRGLoss(predicted_label, label, predicted_energy, energy):
    
    # Cross entropy loss
    ce_loss = cl_loss(predicted_label, label)

    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / energy.size(0)
    
    return ce_loss + mse_loss, ce_loss, mse_loss