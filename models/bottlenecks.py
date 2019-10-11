"""
bottlenecks.py

Module with implementations of different bottlenecks
"""

# PyTorch imports
from torch import cat, eye, mean, randn, zeros
from torch.nn import Module, Linear, ReLU 

# Global variables
__all__ = ["AEBottleneck", "VAEBottleneck", "M2Bottleneck", "LatentClassifier"]
_RELU = ReLU()

# Bottleneck for a simple AutoEncoder
class AEBottleneck(Module):
    
    def __init__(self, num_latent_dims):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        
    def forward(self, X):
        return X
    
# Reparameterization bottleneck for a simple Variational Autoencoder
class VAEBottleneck(Module):
    
    def __init__(self, num_latent_dims):
        super().__init__()
        
        self.num_latent_dims = num_latent_dims
        
        # Variational posterior parameter : mu and logvar
        self.en_mu     = Linear(num_latent_dims, num_latent_dims)
        self.en_logvar = Linear(num_latent_dims, num_latent_dims)

    def forward(self, X, mode, shots=1):
        if mode == None:
            mu, logvar = self.en_mu(X), self.en_logvar(X)
            
            # Reparameterization trick
            std = logvar.mul(0.5).exp()
            eps = std.new(std.size()).normal_()
            z   = eps.mul(std).add(mu)

            return z, mu, logvar
        elif mode == "sample":
            z_samples = randn((shots, X.size(0), self.num_latent_dims), device=X.device)
            return mean(z_samples, dim=0)
            
        
# Reparameterization bottleneck for a semi-supervised Variational Autoencoder
class M2Bottleneck(Module):
    
    def __init__(self, num_latent_dims, num_classes):
        super().__init__()
        
        self.num_latent_dims = num_latent_dims
        self.num_classes     = num_classes

        # Classifier layer for q_{phi}(y|x)
        self.classifier = Classifier(num_latent_dims, num_classes)
        
        # Diagonal gaussian parameter layers : mu_{phi}(z|x,y), Sigma_{phi}(z|x)
        self.en_mu_1  = Linear(num_latent_dims+num_classes, num_latent_dims)
        self.en_mu_2  = Linear(num_latent_dims, num_latent_dims)
        
        self.en_var_1 = Linear(num_latent_dims, num_latent_dims)
        self.en_var_2 = Linear(num_latent_dims, num_latent_dims)
        
    def forward(self, X, mode, shots=1, labels=None):
        if mode == "sample":
            raise NotImplementedError
        else:
            z_prime = X
            
            # Unlabelled dataset
            if labels is None:
                
                # Broadcast z_prime from (batch_size, num_latent_dims) to (batch_size, num_classes, num_latent_dims)
                z_prime = z_prime.view(-1, 1, z_prime.size(1))
                z_prime = z_prime + zeros((z_prime.size(0), self.num_classes, self.num_latent_dims), device=z_prime.device)
                
                # Create the one-hot tensors for each class and concatenate to the z_prime tensors and broadcast from
                # (num_classes, num_classes) to (batch_size, num_classes, num_classes)
                y_onehot = eye(self.num_classes, device=X.device).view(1, self.num_classes, self.num_classes)
                y_onehot = y_onehot + zeros((z_prime.size(0), self.num_classes, self.num_classes), device=y_onehot.device)
                
                # Concatenate the z_prime and identity tensor along the third dimension (dim=2)
                z_prime_y = cat((z_prime, y_onehot), dim=2)
                
                # Collapse the z_prime and concatenated tensor to 2D to pass to the linear layers
                z_prime   = z_prime.view(-1, self.num_latent_dims)
                z_prime_y = z_prime_y.view(-1, self.num_latent_dims + self.num_classes)
            
            # Labelled dataset
            else:
                
                # Create onehot vectors corresponding to the true labels
                y_onehot  = zeros((z_prime.size(0), self.num_classes), device=z_prime.device).scatter(1, labels.reshape(-1,1), 1)
                
                # Concatenate the one-hot vectors with z_prime
                z_prime_y = cat((z_prime, y_onehot), dim=1)
                
            # Pass the constructed vectors through the linear layers to obtain mu and logvar for reparameterization
            mu = _RELU(self.en_mu_1(z_prime_y))
            mu = self.en_mu_2(mu)
                
            logvar = _RELU(self.en_var_1(z_prime))
            logvar = self.en_var_2(logvar)
                
            # Perform the reparameterization trick
            std = logvar.mul(0.5).exp()
            eps = std.new(std.size()).normal_()
            z   = eps.mul(std).add(mu)
            
            # Concatenate the label one-hot vectors with z to pass to the decoder
            z_y = cat((z, y_onehot.view(-1, self.num_classes)), dim=1)
            
            return z_y, z, mu, logvar, self.classifier(X)

# Latent vector classifier     
class LatentClassifier(Module):
    
    def __init__(self, num_latent_dims, num_classes):
        super().__init__()
        
        # Classifier fully connected layers
        self.cl_fc1 = Linear(num_latent_dims, int(num_latent_dims/2))
        self.cl_fc2 = Linear(int(num_latent_dims/2), int(num_latent_dims/4))
        self.cl_fc3 = Linear(int(num_latent_dims/4), int(num_latent_dims/8))
        self.cl_fc4 = Linear(int(num_latent_dims/8), num_classes)
        
    def forward(self, X):
        
        # Fully-connected layers
        x = _RELU(self.cl_fc1(X))
        x = _RELU(self.cl_fc2(x))
        x = _RELU(self.cl_fc3(x))
        x = self.cl_fc4(x)
        
        return x