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
recon_loss = nn.MSELoss(reduction="sum")

# VAE generic loss function i.e. MSE_Loss + KL_Loss
# Returns : Tuple of total loss, mse (reconstruction) loss, kl (divergence) loss
def VAELoss(recon, data, mu, log_var, iteration, num_iterations):

    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    mse_loss = recon_loss(recon, data) / batch_kl_loss.size(0)
    
    return mse_loss + kl_loss, mse_loss, kl_loss