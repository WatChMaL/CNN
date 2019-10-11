"""
nf.py

Implementation of different normalizing flows

Author : Abhishek .
"""

# Standard Python imports
import math

# PyTorch imports
from torch import nn
from torch import zeros
from torch import norm
from torch import log
from torch import abs
from torch import randn
from torch import sqrt
from torch import bmm

# Implementation of the planar flow
class Planar(nn.Module):
    """
        Implementation of the planar flow described in Variational Inference using normalizing flows by Renzende et al. (2015)
    """
    
    # Initializer
    def __init__(self, num_latent_dims, device):
        super(Planar, self).__init__()
        
        # Activation functions
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        
        # Define the parameters of the flows
        self.u_layer = nn.Linear(num_latent_dims, num_latent_dims)
        self.w_layer = nn.Linear(num_latent_dims, num_latent_dims)
        self.b_layer = nn.Linear(num_latent_dims, 1)
        
    # Derivative of the tanh activation function
    def tanh_prime(self, X):
        return 1 - self.tanh(X)**2
    
    # Forward pass
    def forward(self, z, z_prime):
        
        # Apply the amortization of the flow parameters as shown
        # in arxiv.org/pdf/1803.05649
        
        # Compute the flow parameters
        u = self.u_layer(z_prime)
        w = self.w_layer(z_prime)
        b = self.b_layer(z_prime)
        
        # Reshape the flow parameters for using torch matrix operations
        u = u.view(u.size(0), u.size(1), 1)
        w = w.view(w.size(0), w.size(1), 1)
        b = b.view(b.size(0), b.size(1), 1)
        
        # Enforce the constraint on u to make the flow transformation invertible
        wu = bmm(w.permute(0,2,1), u)
        
        m_wu = -1 + self.softplus(-wu)
        
        w_norm = norm(w, dim=1)**2
        w_norm = w_norm.view(w_norm.size(0), 1, 1)
        norm_w = w / w_norm
        
        u_diff = (m_wu - wu) * norm_w
        u_hat = u + u_diff
        
        # Calculate the transformed latent vector z_l
        wzb = bmm(w.permute(0,2,1), z) + b
        z_l = z + bmm(u_hat, wzb)
        
        # Compute the log det jacobian term for the current flow
        psi_z = bmm(w, self.tanh_prime(wzb))
        log_det_jacobian = log(abs(1 + bmm(u_hat.permute(0,2,1), psi_z)))
        
        return z_l, log_det_jacobian.view(log_det_jacobian.size(0))
    
# Implementation of radial flow
class Radial(nn.Module):
    """
        Implementation of the radial flow described in Variational Inference using normalizing flows by Renzende et al. (2015)
    """
    
    # Initializer
    def __init__(self, num_latent_dims, device):
        super(Radial, self).__init__()
        
        # Activation functions
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        
        # Flow attributes
        self.num_latent_dims = num_latent_dims
        
        # Define the parameters of the flow
        self.beta = nn.Parameter(randn((1), device=device))
        self.alpha = nn.Parameter(randn((1), device=device))
        
    # Implementation of the radial function
    def h(self, alpha, r):
        return 1 / (alpha + r)
        
    # Derivative of the radial function
    def h_prime(self, alpha, r):
        return -1 / (alpha + r)**2
        
    # Forward pass
    def forward(self, z, z_0):
        
        # Enforce the constraint that alpha has to be positive
        alpha = sqrt(self.alpha**2)
        
        # Enforce the constrain on beta to make the flow invertible
        beta_hat = alpha + self.softplus(-self.beta)
        
        # Apply the flow transformation
        r = norm(z - z_0, dim=1)
        
        # Compute the dyadic product of the flow
        h_a_r = self.h(alpha, r)
        z_diff = z - z_0
        
        # Transform the input latent vector
        z_l = z + beta_hat * h_a_r.view(h_a_r.size(0), 1, 1) * z_diff
        
        # Compute the log det jacobian term
        h_prime_r = self.h_prime(alpha, r)*r
        det_jac_1 = (1 + beta_hat*h_a_r.view(h_a_r.size(0), 1, 1))
        det_jac_2 = 1 + (beta_hat*h_a_r.view(h_a_r.size(0), 1, 1)) + (h_prime_r).view(h_prime_r.size(0), 1, 1)
        
        log_det_jacobian = log((self.num_latent_dims - 1)*det_jac_1) + log(det_jac_2)
        
        return z_l, log_det_jacobian.view(log_det_jacobian.size(0))