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
reconstruction_loss_val = nn.MSELoss(reduction="none")
cl_loss = nn.CrossEntropyLoss()

# AE generic loss function i.e. RECON Loss
# Returns : Tuple of total loss = RECON (reconstruction) loss
def AELoss(recon, data):

    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    return recon_loss

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
def AECLRGLoss(recon, data, predicted_label, label, predicted_energy, energy):
    
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

# NF generic loss function i.e. RECON Loss + KL Loss + LOGDET
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss, LOGDET
def NFLoss(recon, data, mu, log_var, log_det):
    """
        Compute the loss for the normalizing flows
        Input :
            recon   = reconstructed event tensors, (minibatch_size, *)
            data    = actual event tensors, (minibatch_size, *)
            mu      = mu tensor for z_0, (minibatch_size, latent_dims)
            log_var = log_var tensor for z_0, (minibatch_size, latent_dims)
            log_det = log_det tensor for the flow, (minibatch_size, flow_depth)
    """

    # KL|q_0(z_0)||p(z_k)| == KL|q_0(z_0)||N(O,I)|
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    a = recon_loss + kl_loss - log_det_flow
    return recon_loss + kl_loss - log_det_flow, recon_loss, kl_loss, log_det_flow

# NF+CL+RG loss i.e. RECON Loss + KL Loss + LOGDET + CE Loss + MSE Loss
def NFCLRGLoss(recon, data, mu, log_var, log_det, predicted_label, label, predicted_energy, energy):
    
    # KL|q_0(z_0)||p(z_k)| == KL|q_0(z_0)||N(O,I)|
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    # Cross entropy loss
    ce_loss = cl_loss(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / data.size(0)
    
    return recon_loss + kl_loss - log_det_flow + ce_loss + mse_loss, recon_loss, kl_loss, log_det_flow, ce_loss, mse_loss

# VAE validation loss
def VAEVALLoss(recon, data, mu, log_var):

    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    recon_loss_val = reconstruction_loss_val(recon, data)
    recon_loss_val = recon_loss_val.view(recon_loss_val.size(0), -1).sum(dim=1)
    
    return recon_loss + kl_loss, recon_loss, kl_loss, recon_loss_val, batch_kl_loss

# NF validation loss
def NFVALLoss(recon, data, mu, log_var, log_det):
    """
        Compute the validation loss for the normalizing flows
        Input :
            recon   = reconstructed event tensors, (minibatch_size, *)
            data    = actual event tensors, (minibatch_size, *)
            mu      = mu tensor for z_0, (minibatch_size, latent_dims)
            log_var = log_var tensor for z_0, (minibatch_size, latent_dims)
            log_det = log_det tensor for the flow, (minibatch_size, flow_depth)
    """
    
    # KL|q_0(z_0)||p(z_k)| == KL|N(mu, log_var.exp())||N(O,I)|
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    recon_loss_val = reconstruction_loss_val(recon, data)
    recon_loss_val = recon_loss_val.view(recon_loss_val.size(0), -1).sum(dim=1)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    a = recon_loss + kl_loss - log_det_flow
    return recon_loss + kl_loss - log_det_flow, recon_loss, kl_loss, log_det_flow, recon_loss_val, batch_kl_loss

