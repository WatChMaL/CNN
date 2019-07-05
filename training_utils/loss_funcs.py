"""
loss_funcs.py

Module with implementations of different loss functions useful for autoencoder training

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import sum

# Global variables
recon_loss = nn.MSELoss()

# VAE generic loss function i.e. MSE_Loss + KL_Loss
# Returns : Tuple of total loss, mse (reconstruction) loss, kl (divergence) loss
def VAELoss(recon, data, mean, log_var):

    # Reconstruction Loss
    mse_loss = recon_loss(recon, data)

    # Divergence Loss for Gaussian posterior
    kl_loss = -0.5 * sum(1 + log_var - mean.pow(2) - log_var.exp())

    return mse_loss + kl_loss, mse_loss, kl_loss