"""
kvaenet.py

PyTorch implementation of the KazuNet as a classifier for input of size Batch_Size x Input_Channels x 16 x 40

Author : Abhishek .
"""

# PyTorch imports
import torch.nn as nn
import torch.tensor as tensor
from torch import zeros
from torch import randn_like
from torch import randn
from torch import device

# KVAENet class
class KvaeNet(nn.Module):
    
    # Initializer for the ConvNetVAE model
    def __init__(self, num_input_channels=19, num_classes=3, num_latent_dimensions=64, train=True):
        
        # Initialize the superclass
        super(KvaeNet, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Parameters
        self.num_latent_dims = num_latent_dimensions
        
        # ------------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------------
        
        # Convolutions and max-pooling
        self.en_conv1  = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.en_maxconv1 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, groups=64)
        
        self.en_max1   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.en_max1_indices = None
        self.en_max1_input_size = None
        
        
        self.en_conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.en_conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.en_max2   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.en_max2_indices = None
        self.en_max2_input_size = None
        
        self.en_maxconv2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, groups=64)
        
        self.en_conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.en_conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        
        self.en_max3   = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        
        self.en_max3_indices = None
        self.en_max3_input_size = None
        
        
        self.en_maxconv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=1, groups=64)
        self.en_conv4  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Flattening
        self.en_maxflat_input_size = None
        self.en_maxflat_indices = None
        self.en_maxflat_output_size = None
        
        #----------------------------------
        # Testing flattening method
        #----------------------------------
        self.en_ffc0 = nn.Linear(2304,1024)
        self.en_ffc1 = nn.Linear(1024, 256)
        self.en_ffc2 = nn.Linear(256, 128)
        
        # Fully-connected layers
        self.en_fc1 = nn.Linear(128, 128)
        self.en_fc2 = nn.Linear(128, 128)
        
        # Classifier output layer
        self.en_fc3  = nn.Linear(128, num_classes)
            
        # Encoder parameters layer
        self.en_fc31 = nn.Linear(128, num_latent_dimensions)
        self.en_fc32 = nn.Linear(128, num_latent_dimensions)
        
        # ------------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------------
        
        # Fully-connected layers
        self.de_fc3 = nn.Linear(num_latent_dimensions, 128)
        self.de_fc2 = nn.Linear(128, 128)
        self.de_fc1 = nn.Linear(128, 128)
        
        #----------------------------------
        # Testing flattening method
        #----------------------------------
        self.de_ffc2 = nn.Linear(128, 256)
        self.de_ffc1 = nn.Linear(256, 1024)
        self.de_ffc0 = nn.Linear(1024, 2304)
        
        # Unflattening
        
        # De-convolutions and (un)max-pooling
        self.de_deconv4  = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        
        self.de_maxconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=1, groups=64)
        self.de_deconv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_deconv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        
        self.de_unmax3   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.de_deconv3b = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.de_deconv3a = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        
        
        
        self.de_unmax2   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.de_maxconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, groups=64)
        self.de_deconv2b = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.de_deconv2a = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        
        self.de_unmax1   = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        
        self.de_maxconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, groups=64)
        self.de_deconv1  = nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1)
        
        # Boolean to determine the reparametrization mode
        self.training = train 
            
    # Forward pass
    def forward(self, X, mode="train"):
        
        if mode == "train" or mode == "validate":
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar, shots=1)

            # Return the output image, mean and covariance matrix
            return z, self.decode(z), mu, logvar
        
        elif mode == "generate":
            
            # Encoder to get the parameters for the distribution
            mu, logvar = self.encode(X)

            # Reparameterization trick
            z = self.reparameterize(mu, logvar)

            return z, mu, logvar
        
        elif mode == "sample":
            return self.sample()
    
    # Encoder layers
    def encode(self, X):
        
        # Convolutions and max-pooling
        x = self.en_conv1(X)
        
        
        self.en_max1_input_size = x.size()
        #x, self.en_max1_indices = self.en_max1(x)
        
        x = self.en_maxconv1(x)
        
        x = self.en_conv2b(self.en_conv2a(x))
        
        
        #self.en_max2_input_size = x.size()
        #x, self.en_max2_indices = self.en_max2(x)
        x = self.en_maxconv2(x)
        
        x = self.en_conv3b(self.en_conv3a(x))
        
        #print("Size before applying MaxPooling : ", x.size())
        #self.en_max3_input_size = x.size()
        
        """
        #-----------------------------------------------------
        # Generating random indices
        #-----------------------------------------------------
        rand_x = randn_like(x)
        _, self.en_max3_indices = self.en_max3(rand_x)
        x, _ = self.en_max3(x)
        """
        x, self.en_max3_indices = self.en_max3(x)
        
        #print("Size after applying MaxPooling : ", x.size())
        #print("Size of the indices coming from MaxPooling : ", self.en_max3_indices.size())
        
        
        #x = self.en_maxconv3(x)
        x = self.en_conv4(x)
        
        # Flattening
        """
        self.en_maxflat_input_size = x.size()
        print("Size before the flattening MaxPooling : ", x.size())
        x, self.en_maxflat_indices = nn.MaxPool2d(x.size()[2:], return_indices=True)(x)
        print("Size after the flattening MaxPooling : ", x.size())
        x = x.view(-1, 128)
        print("Size after the reshaping after MaxPooling : ", x.size())
        """
        self.en_maxflat_input_size = x.size()
        x = x.view(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])
        x = self.relu(self.en_ffc0(x))
        x = self.relu(self.en_ffc1(x))
        x = self.relu(self.en_ffc2(x))
        
        # Fully-connected layers
        x = self.relu(self.en_fc1(x))
        x = self.relu(self.en_fc2(x))
        
        return self.en_fc31(x), self.en_fc32(x)
        
    # Reparameterization trick
    def reparameterize(self, mu, logvar, shots=1024):
        
        # Calculate the square-root covariance matrix
        std = logvar.mul(0.5).exp_()
            
        # Initialize a torch tensor holding the sum with the same size as std
        eps_sum = std.new(std.size())
        
        # Compute and reparameterize multiple samples
        for shot in range(shots):
            
            # Sample a vector eps from the normal distribution
            eps = std.new(std.size()).normal_()
            
            # Add in-place to the 
            eps_sum.add_(eps.mul_(std).add_(mu))
            
        # Return the average of the sampled and reparameterized latent vectors
        denom = tensor((1.0/shots))
        
        return eps_sum.mul_(denom)
    
    # Decoder layers
    def decode(self, X):
        # Fully-connected layers
        x = self.de_fc3(X)
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc1(x))
        
        # Unflattening
        """
        print("Size before the reshaping before MaxUnPooling : ", x.size())
        x = x.view(-1, 128, 1, 1)
        print("Size before the unflattening MaxUnpooling : ", x.size())
        x = nn.MaxUnpool2d(self.en_maxflat_input_size[2:])(x, self.en_maxflat_indices, self.en_maxflat_input_size) 
        print("Size after the unflattening MaxUnpooling : ", x.size())
        """
        x = self.relu(self.de_ffc2(x))
        x = self.relu(self.de_ffc1(x))
        x = self.relu(self.de_ffc0(x))
        
        x = x.view(self.en_maxflat_input_size)
        
        # De-convolutions and (un)max-pooling
        x = self.de_deconv4(x)
        
        
        #print("Size before applying MaxUnpooling : ", x.size())
        x = self.de_unmax3(x, self.en_max3_indices, self.en_max3_input_size)
        #print("Size after applying MaxUnpooling : ", x.size())
        
        
        #x = self.de_maxconv3(x)
        x = self.de_deconv3a(self.de_deconv3b(x))
        
        #x = self.de_unmax2(x, self.en_max2_indices, self.en_max2_input_size)
        
        x = self.de_maxconv2(x)
        x = self.de_deconv2a(self.de_deconv2b(x))
        
        
        #x = self.de_unmax1(x, self.en_max1_indices, self.en_max1_input_size)
        
        x = self.de_maxconv1(x)
        x = self.relu(self.de_deconv1(x))
        
        return x
    
    # Decoder for the sampler
    
    def decode_sample(self, X):
        # Fully-connected layers
        x = self.de_fc3(X)
        x = self.relu(self.de_fc2(x))
        x = self.relu(self.de_fc1(x))
        
        x = self.relu(self.de_ffc2(x))
        x = self.relu(self.de_ffc1(x))
        x = self.relu(self.de_ffc0(x))
        
        x = x.view(-1, 128, 3, 6)
        
        # De-convolutions and (un)max-pooling
        x = self.de_deconv4(x)
        
        #-------------------------------------
        # Generate random indices
        #-------------------------------------
        rand_input = randn(128, 4, 10, device=device('cuda'))
        rand_input = rand_input.view(1, rand_input.size()[0], rand_input.size()[1], rand_input.size()[2])
        _, rand_indices = self.en_max3(rand_input)
        
        x = self.de_unmax3(x, rand_indices, rand_input.size())

        x = self.de_deconv3a(self.de_deconv3b(x))
        
        x = self.de_maxconv2(x)
        x = self.de_deconv2a(self.de_deconv2b(x))
        
        x = self.de_maxconv1(x)
        x = self.relu(self.de_deconv1(x))
        
        return x
    
    # Sampler
    
    def sample(self):
        # Sample a vector from the normal distribution
        eps = randn(1, self.num_latent_dims, device=device('cuda'))
        
        # Decode the latent vector to generate an event
        return eps, self.decode_sample(eps)