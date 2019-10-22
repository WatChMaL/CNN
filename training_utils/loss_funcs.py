"""
loss_funcs.py

Module with implementations of different loss functions useful for autoencoder training

Author : Abhishek .
"""

# PyTorch imports
from torch import nn
from torch import sum
from torch import mean
from torch import zeros
from torch import bmm

# Activation functions
softmax = nn.Softmax(dim=1)

# -------------------------
# Class definitions
# -------------------------

# Calculate the loss term corresponding to
# the entropy of a discrete variable
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = nn.functional.softmax(x, dim=1) * nn.functional.log_softmax(x, dim=1)
        b = 1.0 * sum(b, dim=1)
        return b

# Global variables
_RECON_LOSS     = nn.MSELoss(reduction="sum")
_RECON_LOSS_VAL = nn.MSELoss(reduction="none")
_CE_LOSS        = nn.CrossEntropyLoss()
_H_LOSS         = HLoss()

def AELoss(recon, data):
    """Return a reconstruction loss b/w the input and the output
    
    Args:
    recon -- n-dimensional tensor with dims = (mini_batch_size, *)
    data  -- n-dimensional tensor with same dimensionality as recon
    """
    return _RECON_LOSS(recon, data) / data.size(0)

def VAELoss(recon, data, mu, logvar, beta):
    """Return the ELBO loss for the VAE
    
    Args:
    recon  -- n-dimensional tensor with dims = (mini_batch_size, *)
    data   -- n-dimensional tensor with same dimensionality as recon
    mu     -- 2-dimensional tensor with dims = (mini_batch_size, latent_dims),
              corresponds to the mean of each component of the latent vector
              for each sample in the mini-batch
    logvar -- 2-dimensional tensor with dims = (mini_batch_size, latent_dims),
              corresponds to the log variance of each component of the latent
              vector for each sample in the mini-batch
    beta   -- scalar value to weight the KL term in the loss relative to the
              autoencoding term (reconstruction loss)
    """
    batch_kl_loss = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    recon_loss = _RECON_LOSS(recon, data) / recon.size(0)
    
    return recon_loss + (beta*kl_loss), recon_loss, kl_loss

def CELoss(predictions, labels):
    """Return the cross-entropy loss.
    
    Args:
    predictions -- raw, unnormalized output from the classifier
    labels      -- integer labels for each events in the mini-batch
    """
    return _CE_LOSS(prediction, labels)

def VAECLRGLoss(recon, data, mu, log_var, predicted_label, label, predicted_energy, energy):
    
    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = _CE_LOSS(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = _RECON_LOSS(predicted_energy, energy) / data.size(0)
    
    return recon_loss + kl_loss + ce_loss + mse_loss, recon_loss, kl_loss, ce_loss, mse_loss

# AE+Classifier loss funcito i.e. RECON Loss + CE Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss
def AECLRGLoss(recon, data, predicted_label, label, predicted_energy, energy):
    
    # Reconstruction Loss
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = _CE_LOSS(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = _RECON_LOSS(predicted_energy, energy) / data.size(0)
    
    return recon_loss + ce_loss + mse_loss, recon_loss, ce_loss, mse_loss

# Classifier+Regressor loss function i.e. CE Loss + MSE Loss
# Returns : Tuple of total loss, CE (cross-entropy) loss, MSE (mean-squared) loss
def CLRGLoss(predicted_label, label, predicted_energy, energy):
    
    # Cross entropy loss
    ce_loss = _CE_LOSS(predicted_label, label)

    # Mean squared error loss
    mse_loss = _RECON_LOSS(predicted_energy, energy) / energy.size(0)
    
    return ce_loss + mse_loss, ce_loss, mse_loss

# NF generic loss function i.e. RECON Loss + KL Loss + LOGDET
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss, LOGDET
def NFLoss(recon, data, mu, log_var, log_det, beta):
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
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    return recon_loss + (beta*kl_loss) - log_det_flow, recon_loss, kl_loss, log_det_flow

# NF+CL+RG loss i.e. RECON Loss + KL Loss + LOGDET + CE Loss + MSE Loss
def NFCLRGLoss(recon, data, mu, log_var, log_det, predicted_label, label, predicted_energy, energy):
    
    # KL|q_0(z_0)||p(z_k)| == KL|q_0(z_0)||N(O,I)|
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    # Cross entropy loss
    ce_loss = _CE_LOSS(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = _RECON_LOSS(predicted_energy, energy) / data.size(0)
    
    return recon_loss + kl_loss - log_det_flow + ce_loss + mse_loss, recon_loss, kl_loss, log_det_flow, ce_loss, mse_loss

# VAE validation loss
def VAEVALLoss(recon, data, mu, log_var):

    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    recon_loss_val = _RECON_LOSS_VAL(recon, data)
    recon_loss_val = sum(recon_loss_val.view(recon_loss_val.size(0), -1),dim=1)
    
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
    recon_loss = _RECON_LOSS(recon, data) / data.size(0)
    
    recon_loss_val = _RECON_LOSS(recon, data)
    recon_loss_val = sum(recon_loss_val.view(recon_loss_val.size(0), -1), dim=1)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    a = recon_loss + kl_loss - log_det_flow
    return recon_loss + kl_loss - log_det_flow, recon_loss, kl_loss, log_det_flow, recon_loss_val, batch_kl_loss

# M2 labelled loss
def M2LabelledLoss(recon, data, mu, log_var, pi, labels):
    # Compute the MSE Loss averaged over the batch
    mse_loss = mean(sum(_RECON_LOSS_VAL(recon, data).view(data.size(0), -1), dim=1))
    
    # Compute the KL Loss averaged over the batch
    kl_loss = mean(-0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    
    # Compute the CE Loss averaged over the batch
    ce_loss = _CE_LOSS(pi, labels)
    
    return mse_loss + kl_loss + ce_loss, mse_loss, kl_loss, ce_loss

# M2 Unlabelled loss
def M2UnlabelledLoss(recon, data, mu, log_var, pi, labels):
    # Make copies of the original data and reshape to the size of the recon data
    true_data = data.view(data.size(0), 1, data.size(1), data.size(2), data.size(3))
    true_data = true_data + zeros((data.size(0), pi.size(1), data.size(1), data.size(2), data.size(3)), device=true_data.device)
    
    true_data = true_data.view(data.size(0)*pi.size(1), data.size(1), data.size(2), data.size(3))
    
    # Compute the per sample and per class MSE loss
    batch_sample_mse = _RECON_LOSS_VAL(recon, true_data).view(data.size(0)*pi.size(1), -1)
    batch_mse = sum(batch_sample_mse, dim=1).view(data.size(0), pi.size(1), 1)
    mse = mean(mean(batch_mse, dim=1))
    
    # Compute the per sample and per class KL loss
    batch_kl = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).view(data.size(0), pi.size(1), 1)
    kl = mean(mean(batch_mse, dim=1))
    
    # Compute the per sample and per class log-likelihood loss
    batch_elbo = batch_mse + batch_kl
    
    # Apply softmax to the classifier predictions
    softmax_pi = softmax(pi).view(data.size(0), 1, pi.size(1))
    
    # Weighted loss
    weighted_loss = mean(bmm(softmax_pi, batch_elbo).view(-1))
    
    # Calculate the entropy of the variational distribution over the class variables, y
    entropy_loss = mean(h_loss(pi))
    
    # Compute the CE Loss averaged over the batch
    ce_loss = _CE_LOSS(pi, labels)
    
    return weighted_loss + entropy_loss, mse, kl, entropy_loss, ce_loss
