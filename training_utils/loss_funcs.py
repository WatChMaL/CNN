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
reconstruction_loss = nn.MSELoss(reduction="sum")
reconstruction_loss_val = nn.MSELoss(reduction="none")
cen_loss = nn.CrossEntropyLoss()
h_loss = HLoss()

# AE generic loss function i.e. RECON Loss
# Returns : RECON (reconstruction) loss
def AELoss(recon, data):

    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    return recon_loss

# VAE generic loss function i.e. RECON Loss + KL Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss
def VAELoss(recon, data, mu, log_var, beta):

    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    return recon_loss + (beta*kl_loss), recon_loss, kl_loss

# Classifier loss function i.e. CE loss
# Returns : CE (cross-entropy) loss
def CLLoss(predicted_label, label):
    
    # Cross entropy loss
    ce_loss = cen_loss(predicted_label, label)
    
    return ce_loss


# VAE+Classifier+Regressor loss function i.e. RECON Loss + KL Loss + CE Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss, CE (cross-entropy) loss, MSE (mean-squared) loss
def VAECLRGLoss(recon, data, mu, log_var, predicted_label, label, predicted_energy, energy):
    
    # Divergence Loss for Gaussian posterior
    batch_kl_loss = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = mean(batch_kl_loss, dim=0)
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = cen_loss(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / data.size(0)
    
    return recon_loss + kl_loss + ce_loss + mse_loss, recon_loss, kl_loss, ce_loss, mse_loss

# AE+Classifier loss funcito i.e. RECON Loss + CE Loss
# Returns : Tuple of total loss, RECON (reconstruction) loss, KL (divergence) loss
def AECLRGLoss(recon, data, predicted_label, label, predicted_energy, energy):
    
    # Reconstruction Loss
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Cross entropy loss
    ce_loss = cen_loss(predicted_label, label)
    
    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / data.size(0)
    
    return recon_loss + ce_loss + mse_loss, recon_loss, ce_loss, mse_loss

# Classifier+Regressor loss function i.e. CE Loss + MSE Loss
# Returns : Tuple of total loss, CE (cross-entropy) loss, MSE (mean-squared) loss
def CLRGLoss(predicted_label, label, predicted_energy, energy):
    
    # Cross entropy loss
    ce_loss = cen_loss(predicted_label, label)

    # Mean squared error loss
    mse_loss = reconstruction_loss(predicted_energy, energy) / energy.size(0)
    
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
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
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
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    # Cross entropy loss
    ce_loss = cen_loss(predicted_label, label)
    
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
    recon_loss = reconstruction_loss(recon, data) / data.size(0)
    
    recon_loss_val = reconstruction_loss_val(recon, data)
    recon_loss_val = sum(recon_loss_val.view(recon_loss_val.size(0), -1), dim=1)
    
    # Logdet
    batch_log_det_flow = sum(log_det, dim=0)
    log_det_flow = mean(batch_log_det_flow, dim=0)
    
    a = recon_loss + kl_loss - log_det_flow
    return recon_loss + kl_loss - log_det_flow, recon_loss, kl_loss, log_det_flow, recon_loss_val, batch_kl_loss

# M2 labelled loss
def M2LabelledLoss(recon, data, mu, log_var, pi, labels):
    # Compute the MSE Loss averaged over the batch
    mse_loss = mean(sum(reconstruction_loss_val(recon, data).view(data.size(0), -1), dim=1))
    
    # Compute the KL Loss averaged over the batch
    kl_loss = mean(-0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    
    # Compute the CE Loss average over the batch
    ce_loss = cen_loss(softmax(pi), labels)
    
    return mse_loss + kl_loss + ce_loss, mse_loss, kl_loss, ce_loss

# M2 Unlabelled loss
def M2UnlabelledLoss(recon, data, mu, log_var, pi):
    # Make copies of the original data and reshape to the size of the recon data
    true_data = data.view(data.size(0), 1, data.size(1), data.size(2), data.size(3))
    true_data = true_data + zeros((data.size(0), pi.size(1), data.size(1), data.size(2), data.size(3)), device=true_data.device)
    
    true_data = true_data.view(data.size(0)*pi.size(1), data.size(1), data.size(2), data.size(3))
    
    # Compute the per sample and per class MSE loss
    batch_sample_mse = reconstruction_loss_val(recon, true_data).view(data.size(0)*pi.size(1), -1)
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
    entropy_loss = mean(h_loss(softmax_pi.view(softmax_pi.size(0), softmax_pi.size(2))))
    
    return weighted_loss + entropy_loss, mse, kl, entropy_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
