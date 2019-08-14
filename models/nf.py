"""
nf.py

Implementation of different normalizing flows

Author : Abhishek .
"""

# Standard Python imports
import math

# PyTorch imports
from torch import nn
from torch import Tensor
from torch import norm
from torch import log
from torch import abs

class Planar(nn.Module):
    """
        Implementation of the planar flow described in Variational Inference using normalizing flows by Renzende et al. (2015)
    """
    
    # Initializer
    def __init__(self, num_latent_dims):
        super(Planar, self).__init__()
        
        # Activation functions
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        
        # Define the parameters of the flow
        self.u = nn.Parameter(Tensor(num_latent_dims, 1))
        self.w = nn.Parameter(Tensor(num_latent_dims, 1))
        self.b = nn.Parameter(Tensor(1))
        
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
        m_wu = -1 + self.softplus(wu)
        u_hat = self.u + (m_wu - wu) * (self.w / norm(self.w)**2)
        
        # Calculate the transformed latent vector z_l
        z_l = z + u_hat*tanh(self.w.t().matmul(z) + self.b)
        
        # Compute the log det jacobian term
        psi_z = self.w * self.tanh_prime(self.w.t().matmul(z) + self.b)
        log_det_jacobian = log(abs(1 + u_hat.t().matmul(psi_z)))
        
        return z_l, log_det_jacobian.view(log_det_jacbian.size(0), -1)