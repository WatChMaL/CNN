from torch import nn
from torch import Tensor
from torch import norm
from torch import log
from torch import abs
import math

u = Tensor(64, 1)
w = Tensor(64, 1)
b = Tensor(1)

nn.init.kaiming_uniform_(u, a=math.sqrt(5))
nn.init.kaiming_uniform_(w, a=math.sqrt(5))

if b is not None:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(b, -bound, bound)
            
z = Tensor(128, 64, 1)

# Derivative of the tanh activation function
def tanh_prime(X):
    return 1 - tanh(X)**2

# Activation functions
softplus = nn.Softplus()
tanh = nn.Tanh()

# Enforce the constraint on u to make the flow invertible
wu = w.t().matmul(u)
m_wu = -1 + softplus(wu)
u_hat = u + (m_wu - wu) * (w / norm(w)**2)
        
# Calculate the transformed latent vector z_l
z_l = z + u_hat*tanh(w.t().matmul(z) + b)
        
# Compute the log det jacobian term
psi_z = w * tanh_prime(w.t().matmul(z) + b)
log_det_jacobian = log(abs(1 + u_hat.t().matmul(psi_z)))

print(z_l.size())
print(log_det_jacobian.size())

log_det_jacobians = Tensor(5, 128)
log_det_jacobians[0] = log_det_jacobian.view(log_det_jacobian.size(0))

print(log_det_jacobians.size())