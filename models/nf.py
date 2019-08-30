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
        self.u = nn.Parameter(zeros((num_latent_dims, 1), device=device))
        self.w = nn.Parameter(zeros((num_latent_dims, 1), device=device))
        self.b = nn.Parameter(zeros((1), device=device))
        
        # Initialize the parameters of the flow using similar
        # initialization to the linear layer
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)
        
    # Derivative of the tanh activation function
    def tanh_prime(self, X):
        return 1 - self.tanh(X)**2
    
    # Forward pass
    def forward(self, z):
        # Enforce the constraint on u to make the flow invertible
        wu = self.w.t().matmul(self.u)
        m_wu = -1 + self.softplus(-wu)
        u_hat = self.u + (m_wu - wu) * (self.w / norm(self.w)**2)
        
        # Calculate the transformed latent vector z_l
        z_l = z + u_hat*self.tanh(self.w.t().matmul(z) + self.b)
        
        # Compute the log det jacobian term
        psi_z = self.w * self.tanh_prime(self.w.t().matmul(z) + self.b)
        log_det_jacobian = log(abs(1 + u_hat.t().matmul(psi_z)))
        
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