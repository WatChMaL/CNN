"""
engine_vae.py

Engine to train, validate and test various VAE architectures

Author  : Abhishek .
"""

# PyTorch imports
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

# Standard and data processing imports
import os
import sys
import time
import math
import random
import collections
import numpy as np

# Scikit learn imports
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# WatChMaL imports
from io_utils import ioconfig
from io_utils.data_handling_2 import WCH5Dataset
from plot_utils.notebook_utils import CSVData
import training_utils.loss_funcs as loss_funcs

# Logging and dumping keys : values to save during logging or dummping
log_keys = ["loss", "recon_loss", "kl_loss", "ce_loss", "mse_loss", "accuracy", "logdet"]
event_dump_keys = ["recon", "z", "mu", "logvar", "z_prime", "softmax", "samples", "predicted_labels", "predicted_energies", "z_k", "logdet"]

# Dictionaries for the particle types and their keys in the dataset
label_dict = {0.:"gamma", 1.:"e", 2.:"mu"}
inverse_label_dict = {"gamma":0., "e":1., "mu":2.}

# Class for the training engine for the WatChMaLVAE
class EngineVAE:
    
    """
    Purpose : Training engine for the WatChMaLVAE. Performs training, validation,
              and testing of the models
    """
    def __init__(self, model, config, model_variant, model_train_type):
        
        # Initialize the engine attributes
        self.model = model
        self.model_variant = model_variant
        self.model_train_type = model_train_type
        
        self.softmax = nn.Softmax(dim=1)
        
        # Set the device to be used for the model
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]
            print("Main GPU: "+self.devids[0])
            
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)
                print("CUDA is available")
            else:
                self.device=torch.device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=torch.device("cpu")
            
        self.model.to(self.device)
        
        # Optimizer parameters to be used
        learning_rate = 0.0001
        
        # Initialize the optimizer with the correct parameters to optimize for different settings
        if self.model_train_type is "train_all":
            self.optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)
        elif self.model_train_type is "train_ae_or_vae_only" or self.model_train_type is "train_nf_only":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.model_train_type is "train_bottleneck_only":
            if type(self.model) is nn.DataParallel:
                self.optimizer = optim.Adam(self.model.module.bottleneck.parameters(),lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.bottleneck.parameters(),lr=learning_rate)
                
        elif self.model_train_type is "train_cl_or_rg_only":
            if type(self.model) is nn.DataParallel:
                self.optimizer = optim.Adam(list(self.model.module.classifier.parameters()) + list(self.model.module.regressor.parameters()),lr=learning_rate)
            else:
                self.optimizer = optim.Adam(list(self.model.classifier.parameters()) + list(self.model.regressor.parameters()),lr=learning_rate)
        
        # Declare the loss function
        if model_variant is "AE":
            if self.model_train_type is "train_all":
                self.criterion = loss_funcs.AECLRGLoss
            elif self.model_train_type is "train_cl_or_rg_only":
                self.criterion = loss_funcs.CLRGLoss
            elif self.model_train_type is "train_bottleneck_only" or self.model_train_type is "train_ae_or_vae_only":
                self.criterion = loss_funcs.AELoss
        elif model_variant is "VAE":
            if self.model_train_type is "train_all":
                self.criterion = loss_funcs.VAECLRGLoss
            elif self.model_train_type is "train_cl_or_rg_only":
                self.criterion = loss_funcs.CLRGLoss
            elif self.model_train_type is "train_bottleneck_only" or self.model_train_type is "train_ae_or_vae_only":
                self.criterion = loss_funcs.VAELoss
        elif model_variant is "NF":
            if self.model_train_type is "train_all":
                self.criterion = loss_funcs.NFCLRGLoss
            elif self.model_train_type is "train_cl_or_rg_only":
                self.criterion = loss_funcs.CLRGLoss
            elif self.model_train_type is "train_bottleneck_only" or self.model_train_type is "train_nf_only":
                self.criterion = loss_funcs.NFLoss
            
        # Placeholders for data and labels
        self.data=None
        self.labels=None
        self.energies=None
        self.iteration=None
        self.num_iterations=None

        # Initialize the dataset iterator
        # NOTE: The functionality of this block is coupled to the implementation of WCH5Dataset in the io_utils module

        # Create the dataset
        self.dset=WCH5Dataset(config.path, config.cl_train_split, config.cl_val_split,
                              config.vae_val_split, config.test_split, self.model_train_type,
                              shuffle=config.shuffle, reduced_dataset_size=config.subset)
        
        self.train_iter=DataLoader(self.dset,
                                   batch_size=config.batch_size_train,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices))
        
        self.val_iter=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        
        self.test_iter=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))

        self.dirpath=config.dump_path + time.strftime("%Y%m%d_%H%M%S") + "/"
        
        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        self.config=config
        
        # Save a copy of the config in the dump path
        ioconfig.saveConfig(self.config, self.dirpath + "config_file.ini")
        
    # Method to compute the loss using the forward pass
    def forward(self, mode="all", forward_type="train"):
        
        if self.data is not None and len(self.data.size()) is 4 and mode is not "decode":
            # Move the data to the user-specified device
            self.data = self.data.to(self.device)
            self.data = self.data.permute(0,3,1,2)
        
        if self.labels is not None:
            # Move the labels to the user-specified device
            self.labels = self.labels.to(self.device)
            
        if self.energies is not None:
            # Move the energies to the user-specified device
            self.energies = self.energies.to(self.device)
        
        # Set the grad calculation mode
        grad_mode = True if forward_type is "train" else False
        
        # Return dict
        return_dict = None
        
        with torch.set_grad_enabled(grad_mode):
            
            if mode is "all":
                
                # Forward for VAE
                if self.model_variant is "VAE":
                    
                    # Collect the output from the model
                    recon, z, mu, logvar, z_prime, predicted_labels, predicted_energies = self.model(self.data, mode, device=self.devids[0])
                    loss, recon_loss, kl_loss, ce_loss, mse_loss = self.criterion(recon, self.data, mu,
                                                                                  logvar, predicted_labels,
                                                                                  self.labels, predicted_energies,
                                                                                  self.energies)
                    self.loss = loss
                    
                    softmax           = self.softmax(predicted_labels).cpu().detach().numpy()
                    predicted_labels  = torch.argmax(predicted_labels, dim=-1)
                    accuracy          = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"               : loss.cpu().detach().item(),
                                   "recon_loss"         : recon_loss.cpu().detach().item(),
                                   "kl_loss"            : kl_loss.cpu().detach().item(),
                                   "ce_loss"            : ce_loss.cpu().detach().item(),
                                   "mse_loss"           : mse_loss.cpu().detach().item(),
                                   "accuracy"           : accuracy,
                                   "recon"              : recon.cpu().detach().numpy(),
                                   "z"                  : z.cpu().detach().numpy(),
                                   "mu"                 : mu.cpu().detach().numpy(),
                                   "logvar"             : logvar.cpu().detach().numpy(),
                                   "z_prime"            : z_prime.cpu().detach().numpy(),
                                   "predicted_labels"   : predicted_labels.cpu().detach().numpy(),
                                   "softmax"            : softmax,
                                   "predicted_energies" : predicted_energies.cpu().detach().numpy()}
                    
                # Forward for AE
                elif self.model_variant is "AE":
                
                    recon, predicted_labels, predicted_energies = self.model(self.data, mode, device=self.devids[0])
                    loss, recon_loss, ce_loss, mse_loss = self.criterion(recon, self.data,
                                                                         predicted_labels, self.labels,
                                                                         predicted_energies, self.energies)
                    self.loss = loss
                    
                    softmax          = self.softmax(predicted_labels).cpu().detach().numpy()
                    predicted_labels = torch.argmax(predicted_labels, dim=-1)
                    accuracy         = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())
                    
                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"               : loss.cpu().detach().item(),
                                   "recon_loss"         : recon_loss.cpu().detach().item(),
                                   "ce_loss"            : ce_loss.cpu().detach().item(),
                                   "mse_loss"           : mse_loss.cpu().detach().item(),
                                   "accuracy"           : accuracy,
                                   "recon"              : recon.cpu().detach().numpy(),
                                   "predicted_labels"   : predicted_labels.cpu().detach().numpy(),
                                   "softmax"            : softmax,
                                   "predicted_energies" : predicted_energies.cpu().detach().numpy()}
                    
                # Forward for NF
                elif self.model_variant is "NF":
                    
                    # Collect the output from the model
                    recon, z_k, z, mu, logvar, z_prime, logdet, predicted_labels, predicted_energies = self.model(self.data, mode, device=self.devids[0])
                    
                    # Compute the loss for the hybrid model
                    loss, recon_loss, kl_loss, logdet, ce_loss, mse_loss = self.criterion(recon, self.data, mu, logvar, logdet,
                                                                                           predicted_labels, self.labels, predicted_energies, self.energies)
                    
                    self.loss = loss
                    
                    # Define the classifier predictions 
                    softmax           = self.softmax(predicted_labels).cpu().detach().numpy()
                    predicted_labels  = torch.argmax(predicted_labels, dim=-1)
                    accuracy          = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)
                    
                    return_dict = {"loss"               : loss.cpu().detach().item(),
                                   "recon_loss"         : recon_loss.cpu().detach().item(),
                                   "kl_loss"            : kl_loss.cpu().detach().item(),
                                   "logdet"             : logdet.cpu().detach().item(),
                                   "ce_loss"            : ce_loss.cpu().detach().item(),
                                   "mse_loss"           : mse_loss.cpu().detach().item(),
                                   "accuracy"           : accuracy,
                                   "recon"              : recon.cpu().detach().numpy(),
                                   "z_k"                : z_k.cpu().detach().numpy(),
                                   "logdet"             : logdet.cpu().detach().numpy(),
                                   "z"                  : z.cpu().detach().numpy(),
                                   "mu"                 : mu.cpu().detach().numpy(),
                                   "logvar"             : logvar.cpu().detach().numpy(),
                                   "z_prime"            : z_prime.cpu().detach().numpy(),
                                   "predicted_labels"   : predicted_labels.cpu().detach().numpy(),
                                   "softmax"            : softmax,
                                   "predicted_energies" : predicted_energies.cpu().detach().numpy()}
                    

            elif mode is "ae_or_vae":
                
                # Forward for VAE
                if self.model_variant is "VAE":

                    # Collect the output from the model
                    recon, z, mu, logvar, z_prime = self.model(self.data, mode, device=self.devids[0])
                    
                    if forward_type is "train":
                        loss, recon_loss, kl_loss = self.criterion(recon, self.data, 
                                                                   mu, logvar, self.iteration,
                                                                  self.num_iterations)
                        
                        self.loss = loss

                        # Restore the shape of recon
                        recon = recon.permute(0,2,3,1)

                        return_dict = {"loss"            : loss.cpu().detach().item(),
                                       "recon_loss"      : recon_loss.cpu().detach().item(),
                                       "kl_loss"         : kl_loss.cpu().detach().item(),
                                       "recon"           : recon.cpu().detach().numpy(),
                                       "z"               : z.cpu().detach().numpy(),
                                       "mu"              : mu.cpu().detach().numpy(),
                                       "logvar"          : logvar.cpu().detach().numpy(),
                                       "z_prime"         : z_prime.cpu().detach().numpy()}
                    
                    elif forward_type is "validation":
                        loss, recon_loss, kl_loss, recon_loss_val, kl_loss_val = loss_funcs.VAEVALLoss(recon, self.data, mu, logvar)
                        
                        # Restore the shape of recon
                        recon = recon.permute(0,2,3,1)
                        
                        return_dict = {"loss"            : loss.cpu().detach().item(),
                                       "recon_loss"      : recon_loss.cpu().detach().item(),
                                       "kl_loss"         : kl_loss.cpu().detach().item(),
                                       "recon"           : recon.cpu().detach().numpy(),
                                       "z"               : z.cpu().detach().numpy(),
                                       "mu"              : mu.cpu().detach().numpy(),
                                       "logvar"          : logvar.cpu().detach().numpy(),
                                       "z_prime"         : z_prime.cpu().detach().numpy(),
                                       "recon_loss_val"  : recon_loss_val.cpu().detach().numpy(),
                                       "kl_loss_val"     : kl_loss_val.cpu().detach().numpy()}
                        
                # Forward for AE
                elif self.model_variant is "AE":

                    recon = self.model(self.data, mode, device=self.devids[0])
                    loss = self.criterion(recon, self.data)
                    self.loss = loss
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss" : loss.cpu().detach().item(),
                                   "recon"     : recon.cpu().detach().numpy()}
                    
            # Forward for the normalizing flow
            elif mode is "nf":
                
                # Collect the output from the model
                recon, z_k, logdet, z, mu, logvar, z_prime = self.model(self.data, mode, device=self.devids[0])
                
                # Forward for training passes
                if forward_type is "train":
                    loss, recon_loss, kl_loss, logdet = self.criterion(recon, self.data, mu, logvar, logdet, self.iteration, self.num_iterations)

                    self.loss = loss

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"            : loss.cpu().detach().item(),
                                   "recon_loss"      : recon_loss.cpu().detach().item(),
                                   "kl_loss"         : kl_loss.cpu().detach().item(),
                                   "logdet"         : logdet.cpu().detach().item(),
                                   "recon"           : recon.cpu().detach().numpy(),
                                   "z_k"             : z_k.cpu().detach().numpy(),
                                   "logdet"          : logdet.cpu().detach().numpy(),
                                   "z"               : z.cpu().detach().numpy(),
                                   "mu"              : mu.cpu().detach().numpy(),
                                   "logvar"          : logvar.cpu().detach().numpy(),
                                   "z_prime"         : z_prime.cpu().detach().numpy()}
                    
                # Forward for validation passes
                elif forward_type is "validation":
                    loss, recon_loss, kl_loss, logdet, recon_loss_val, kl_loss_val = loss_funcs.NFVALLoss(recon, self.data, mu, logvar, logdet)

                    self.loss = loss

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"            : loss.cpu().detach().item(),
                                   "recon_loss"      : recon_loss.cpu().detach().item(),
                                   "kl_loss"         : kl_loss.cpu().detach().item(),
                                   "logdet"         : logdet.cpu().detach().item(),
                                   "recon"           : recon.cpu().detach().numpy(),
                                   "z_k"             : z_k.cpu().detach().numpy(),
                                   "logdet"          : logdet.cpu().detach().numpy(),
                                   "z"               : z.cpu().detach().numpy(),
                                   "mu"              : mu.cpu().detach().numpy(),
                                   "logvar"          : logvar.cpu().detach().numpy(),
                                   "z_prime"         : z_prime.cpu().detach().numpy(),
                                   "recon_loss_val"  : recon_loss_val.cpu().detach().numpy(),
                                   "kl_loss_val"     : kl_loss_val.cpu().detach().numpy()}
                
            elif mode is "generate_latents":
                
                if self.model_variant is "VAE":
                    
                    # Generate only the latent vectors
                    z_gen, mu, logvar = self.model(self.data, mode, device=self.devids[0])

                    return_dict = {"z_gen"   : z_gen.cpu().detach().numpy(),
                                   "mu"      : mu.cpu().detach().numpy(),
                                   "logvar"  : logvar.cpu().detach().numpy()}
                    
                elif self.model_variant is "AE":
                    
                    # Generate only the latent vectors
                    z_gen = self.model(self.data, mode, device=self.devids[0])

                    return_dict = {"z_gen"   : z_gen.cpu().detach().numpy()}

            elif mode is "cl_or_rg":
                    
                predicted_labels, predicted_energies = self.model(self.data, mode, device=self.devids[0])
                loss, ce_loss, mse_loss = self.criterion(predicted_labels, self.labels, predicted_energies, self.energies)
                self.loss = loss
            
                softmax          = self.softmax(predicted_labels).cpu().detach().numpy()
                predicted_labels = torch.argmax(predicted_labels,dim=-1)
                accuracy         = (predicted_labels == self.labels).sum().item() / float(predicted_labels.nelement())
                    
                return_dict = {"loss"               : loss.cpu().detach().item(),
                               "ce_loss"            : ce_loss.cpu().detach().item(),
                               "mse_loss"           : mse_loss.cpu().detach().item(),
                               "accuracy"           : accuracy,
                               "predicted_labels"   : predicted_labels.cpu().detach().numpy(),
                               "softmax"            : softmax,
                               "predicted_energies" : predicted_energies.cpu().detach().numpy()}
            
            elif mode is "sample":
            
                samples, predicted_labels, energies = self.model(self.data, mode, device=self.devids[0])
                
                labels = torch.argmax(predicted_labels,dim=-1)
                
                return_dict = {"samples"            : samples.permute(0,2,3,1).cpu().detach().numpy(),
                               "predicted_labels"   : labels.cpu().detach().numpy(),
                               "predicted_energies" : energies.cpu().detach().numpy()}
                
            elif mode is "decode":
                
                samples, predicted_labels, energies = self.model(self.data, mode, device=self.devids[0])
                labels = torch.argmax(predicted_labels,dim=-1)
                
                return_dict = {"samples"            : samples.permute(0,2,3,1).cpu().detach().numpy(),
                               "predicted_labels"   : labels.cpu().detach().numpy(),
                               "predicted_energies" : energies.cpu().detach().numpy()}
  
        if self.data is not None and len(self.data.size()) is 4 and mode is not "decode":
            # Restore the shape of the data
            self.data = self.data.permute(0,2,3,1)
        
        return return_dict
    
    def backward(self):
        
        self.optimizer.zero_grad()  # Reset gradient accumulation
        self.loss.backward()        # Propagate the loss backwards
        self.optimizer.step()       # Update the optimizer parameters
        
    def train(self, epochs=10.0, report_interval=10, num_validations=1000):

        # Prepare attributes for data logging
        self.train_log = CSVData(self.dirpath+'log_train.csv')
        self.val_log = CSVData(self.dirpath+'log_val.csv')
        
        # Variables to save the actual and reconstructed events
        np_event_path = self.dirpath+"/iteration_"
        
        # Calculate the total number of iterations in this training session
        num_iterations = math.ceil(epochs*len(self.train_iter))
        self.num_iterations = num_iterations
        
        # Determine the validation interval to use depending on the total number of iterations
        # Add max to avoid modulo by zero error
        valid_interval = max(1, math.floor(num_iterations/num_validations))
        
        # Save the dump at the earliest validation, middle of the training, last validation
        # near the end of training
        
        dump_iterations = [valid_interval]
        dump_iterations.append(valid_interval*math.floor(num_validations/2))
        dump_iterations.append(valid_interval*num_validations)
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to save the best model
        best_loss = 1000000.
        
        # Set neural net to training mode
        self.model.train()
        
        # Training loop
        while (math.floor(epoch) < epochs):
            
            print('Epoch',math.floor(epoch),
                  'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                
                # Get only the charge data
                self.data     = data[0][:,:,:,:19].float()
                self.labels   = data[1].long()
                self.energies = data[2].float()
                
                # Update the global iteration counter
                self.iteration = iteration
                
                # Setup the mode to call the forward method
                if self.model_train_type is "train_all":
                    mode = "all"
                elif self.model_train_type in ["train_ae_or_vae_only", "train_bottleneck_only"]:
                    mode = "ae_or_vae"
                elif self.model_train_type is "train_cl_or_rg_only":
                    mode = "cl_or_rg"
                elif self.model_train_type is "train_nf_only":
                    mode = "nf"

                # Call forward: pass the data to the model and compute predictions and loss
                res = self.forward(mode=mode, forward_type="train")
                
                # Call backward: backpropagate error and update weights
                self.backward()
                
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                keys = ['iteration','epoch']
                values = [iteration, epoch]
                for key in log_keys:
                    if key in res.keys():
                        keys.append(key)
                        values.append(res[key])

                # Log/Report
                self.train_log.record(keys, values)
                self.train_log.write()
                
                # once in a while, report
                if i is 0 or (i+1)%report_interval is 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' %
                          (iteration, epoch, res['loss']))
                    
                # Run validation on user-defined intervals
                if iteration%valid_interval is 0:
                        
                    self.model.eval()
                    val_data = next(iter(self.val_iter))

                    # Extract the event data from the input data tuple
                    self.data     = val_data[0][:,:,:,:19].float()
                    self.labels   = val_data[1].long()
                    self.energies = val_data[2].float()

                    res = self.forward(mode=mode)
                    
                    if iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies"]
                        save_arr_values = [self.data.cpu().numpy(), val_data[1], val_data[2]]
                        for key in event_dump_keys:
                            if key in res.keys():
                                save_arr_keys.append(key)
                                save_arr_values.append(res[key])

                        # Save the actual and reconstructed event to the disk
                        np.savez(np_event_path + str(iteration) + ".npz",
                                 **{key:value for key,value in zip(save_arr_keys,save_arr_values)})

                    keys = ['iteration','epoch']
                    values = [iteration, epoch]
                    for key in log_keys:
                        if key in res.keys():
                            keys.append(key)
                            values.append(res[key])
                            
                    # Record the validation stats to the csv
                    self.val_log.record(keys,values)
                    
                    # Save the best model
                    if res["loss"] < best_loss:
                        self.save_state(model_type="best")
                    
                    # Save the latest model
                    self.save_state(model_type="latest")

                    self.val_log.write()
                    self.model.train()
                    
                if epoch >= epochs:
                    break
                    
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
            
        self.val_log.close()
        self.train_log.close()
        
    # Method to validate the model training on the validation set
    def validate(self, subset="validation", num_dump_events=1024):
        
        # Print start message
        if subset is "train":
            message = "Validating model on the validation set"
        elif subset is "validation":
            message = "Validating model on the train set"
        elif subset is "test":
            message = "Validating model on the test set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None
        
        print(message)
        
        # Variables to output at the end
        curr_iteration = 1
        
        # Setup the CSV file for logging the output, path to save the actual and reconstructed events, dataloader iterator
        if subset is "train":
            self.log = CSVData(self.dirpath+"train_validation_log.csv")
            np_event_path = self.dirpath + "/train_valid_iteration_"
            data_iter = self.train_iter
            dump_iterations = max(1, math.ceil(num_dump_events/self.config.batch_size_train))
        elif subset is "validation":
            self.log = CSVData(self.dirpath+"valid_validation_log.csv")
            np_event_path = self.dirpath + "/val_valid_iteration_"
            data_iter = self.val_iter
            dump_iterations = max(1, math.ceil(num_dump_events/self.config.batch_size_val))
        else:
            self.log = CSVData(self.dirpath+"test_validation_log.csv")
            np_event_path = self.dirpath + "/test_validation_iteration_"
            data_iter = self.test_iter
            dump_iterations = max(1, math.ceil(num_dump_events/self.config.batch_size_test))
            
        print("Dump iterations = {0}".format(dump_iterations))
        
        # List holding the keys for values to dump at the end
        metric_dump_keys = ["indices", "recon_loss", "kl_loss"]
        
        # Array holding all the dataset indices
        global_indices = []
        recon_loss_values = []
        kl_loss_values = []
        
        save_arr_dict = {"events":[], "labels":[], "energies":[]}
        
        for data in iter(data_iter):
            
            sys.stdout.write("Iteration : " + str(curr_iteration) + "\n")

            # Extract the event data from the input data tuple
            self.data     = data[0][:,:,:,:19].float()
            self.labels   = data[1].long()
            self.energies = data[2].float()
            indices       = data[4].cpu().numpy()
            
            # Add the indices from the dataset to the array
            global_indices.extend(indices)

            # Setup the mode to call the forward method
            if self.model_train_type is "train_all":
                mode = "all"
            elif self.model_train_type in ["train_ae_or_vae_only", "train_bottleneck_only"]:
                mode = "ae_or_vae"
            elif self.model_train_type is "train_nf_only":
                mode = "nf"
            elif self.model_train_type is "train_cl_or_rg_only":
                mode = "cl_or_rg"
                    
            res = self.forward(mode=mode, forward_type="validation")
                 
            keys = ["epoch"]
            values = [curr_iteration]
            for key in log_keys:
                if key in res.keys():
                    keys.append(key)
                    values.append(res[key])
            
            # Log/Report
            self.log.record(keys, values)
            self.log.write()
            
            # Add the loss values to the global arrays
            recon_loss_values.extend(res["recon_loss_val"])
            kl_loss_values.extend(res["kl_loss_val"])
            
            # Add the result keys to the dump dict in the first iterations
            if curr_iteration is 1:
                for key in event_dump_keys:
                    if key in res.keys():
                        save_arr_dict[key] = []
                        
                print(save_arr_dict)
            
            # Save the actual and reconstructed event to the disk
            if curr_iteration <= dump_iterations:
                save_arr_dict["events"].append(self.data.cpu().numpy())
                save_arr_dict["labels"].append(self.labels.cpu().numpy())
                save_arr_dict["energies"].append(self.energies.cpu().numpy())
                
                for key in event_dump_keys:
                    if key in res.keys():
                        save_arr_dict[key].append(res[key])
                        
                if curr_iteration == dump_iterations:
                    print("Saving the npz array :")
                    np.savez(np_event_path + "dump.npz", **save_arr_dict)
            
            curr_iteration += 1
            
        # Save and dump all the computed metrics for this training dataset
        np.savez(np_event_path + "metrics.npz",
                 **{key:value for key,value in zip(metric_dump_keys,[global_indices, recon_loss_values, kl_loss_values])})
        
    # Sample vectors from the normal distribution and decode to reconstruct events
    def sample(self, num_samples=10, trained=False):
        
        # Setup the path
        sample_save_path = self.dirpath + 'samples/'
            
        # Create the directory if it does not already exist
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)
            
        # Setup model status to differentiate samples from trained and untrained model
        model_status = "trained" if trained else "untrained"
        
        sample_save_path = sample_save_path + str(self.config.model[1])
        if num_samples <= self.config.batch_size_val:
            num_iterations = 1
            sample_batch_size = num_samples
        else:
            num_iterations = int(num_samples/self.config.batch_size_val)
            sample_batch_size = self.config.batch_size_val
        
        save_arr_keys = ["samples", "predicted_labels", "predicted_energies"]
        save_arr_values = [[],[],[]]
        
        for i in range(num_iterations):
            self.data = torch.zeros((sample_batch_size, 1), device=self.devids[0])
            res = self.forward(mode="sample", forward_type="sample")
            
            index = 0
            for key in event_dump_keys:
                if key in res.keys():
                    save_arr_values[index].append(res[key])
                    index = index + 1
                    
        # Save the actual and reconstructed event to the disk
        np.savez(sample_save_path + "_" + str(model_status) + ".npz", 
                 **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
            
        
    # Generate and save the latent vectors for training and validation sets
    
    def generate_latent_vectors(self, mode="pre", report_interval=10):
        
        # Setup the save path for the vectors
        train_save_path = self.dirpath + mode + "_train_latent.npz"
        valid_save_path = self.dirpath + mode + "_valid_latent.npz"
        
        # Switch the model
        self.model.eval()
        
        # List to hold the values
        train_latent_list = []
        train_labels_list = []
        train_energies_list = []
        
        valid_latent_list = []
        valid_labels_list = []
        valid_energies_list = []
        
        # Print message
        print("Generating latent vectors over the training data")
        
        with torch.set_grad_enabled(False):
        
            # Iterate over the training samples
            for i, data in enumerate(self.train_iter):

                # once in a while, report
                if i is 0 or (i+1)%report_interval is 0:
                    print("... Training data iteration %d ..." %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19], data[1], data[2]
                
                res = self.forward(mode="generate")

                # Add the values to the lists
                train_latent_list.extend(res["z_gen"])
                train_labels_list.extend(labels)
                train_energies_list.extend(energies)

            # Print message
            print("Generating latent vectors over the validation data")

            # Iterate over the validation samples
            for i, data in enumerate(self.val_iter):

                # once in a while, report
                if i is 0 or (i+1)%report_interval is 0:
                    print("... Validation data iteration %d ..." %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19], data[1], data[2]

                res = self.forward(mode="generate_latents")

                # Add the values to the lists
                valid_latent_list.extend(res["z_gen"])
                valid_labels_list.extend(labels)
                valid_energies_list.extend(energies)
            
        # Save the lists as numpy arrays
        np.savez(train_save_path,
                 latents=train_latent_list,
                 labels=train_labels_list,
                 energies=train_energies_list)
        
        np.savez(valid_save_path,
                 latents=valid_latent_list,
                 labels=valid_labels_list,
                 energies=valid_energies_list)
        
        # Switch the model back to training
        self.model.train()
        
    # Interpolate b/w two regions in the latent space
    
    def interpolate(self, subset="validation", event_type="e", angle_1=0, energy_1=200, angle_2=0, energy_2=800, intervals=10, num_neighbors=1024, trained=False):
        
        assert subset in ["train", "validation", "test"]
        assert energy_1 > 0 and energy_1 < 1000
        assert energy_2 > 0 and energy_2 < 1000
        assert angle_1 > -3.14 and angle_1 < 3.14
        assert angle_2 > -3.14 and angle_2 < 3.14
        
        model_status = "trained" if trained else "untrained"
        
        # Initialize the dataloader based on the user-defined subset of the entire dataset
        if subset is "train":
            data_iter=DataLoader(self.dset,
                                 batch_size=self.config.batch_size_train,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.train_indices))
        elif subset is "validation":
            data_iter=DataLoader(self.dset,
                                 batch_size=self.config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        else:
            data_iter = DataLoader(self.dset,
                                  batch_size=self.config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))
            
        print("engine_vae.interpolate() : Initialized the dataloader object")
        
        # Iterate over the data_iter object and collect all the labels, energies, angles and indices
        labels = []
        energies = []
        indices = []
        angles = []

        # Loop over the dataloader object to collect values
        for data in data_iter:
            labels.append(data[1])
            energies.append(data[2])
            angles.append(data[3])
            indices.append(data[4])
            
        print("engine_vae.interpolate() : Collected the labels, energies, angles and indices from the dataset, list length =", len(labels))
            
        num_batches = len(labels)
        num_samples_per_batch = labels[0].size(0)
        num_samples = (num_batches-1)*num_samples_per_batch

        labels_np, energies_np, indices_np, angles_np = np.ndarray((num_samples)), np.ndarray((num_samples)), np.ndarray((num_samples)), np.ndarray((num_samples))

        i = 0
        j = 0

        # Read the values from the tensor and insert them into the 1-d numpy arrays
        while i < num_batches-1:
            labels_np[j:j+num_samples_per_batch]   = labels[i].numpy()
            energies_np[j:j+num_samples_per_batch] = energies[i].numpy().reshape(-1)
            indices_np[j:j+num_samples_per_batch]  = indices[i].numpy()
            angles_np[j:j+num_samples_per_batch]   = angles[i].numpy()[:,1]

            i = i + 1
            j = j + num_samples_per_batch
            
        label_counter = collections.Counter(labels_np)
        print("engine_vae.interpolate() : Label counter dictionary =", label_counter)
        
        print("engine_vae.interpolate() : Constructed the numpy arrays for labels, energies, angles and indices, Shape =", labels_np.shape)
            
        # Select the values based on the particle type i.e. use the energy and index
        # values to build the knn tree only for the user-defined particle type
        energies_np = energies_np[labels_np == inverse_label_dict[event_type]]
        indices_np  = indices_np[labels_np == inverse_label_dict[event_type]]
        angles_np   = angles_np[labels_np == inverse_label_dict[event_type]]
        
        # Normalize the energy numpy array to the same scale as angle np array to prevent bias
        # in the NearestNeighbors method
        energy_scaler = MinMaxScaler(feature_range=(-math.pi, math.pi), copy=True)
        energies_np = energy_scaler.fit_transform(energies_np.reshape(-1, 1)).reshape(-1)
        
        # Transform the query energies to allow for correct querying of the ball tree
        energy_1 = energy_scaler.transform(np.array([energy_1]).reshape(-1,1)).item()
        energy_2 = energy_scaler.transform(np.array([energy_2]).reshape(-1,1)).item()
        
        # Construct the tree for the k-nearest neighbors
        nn_features = np.array([angles_np, energies_np]).T
            
        nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree',
                                metric='euclidean').fit(nn_features)
        
        print("engine_vae.interpolate() : Constructed the query tree using the nearest neighbors method")
        print("engine_vae.interpolate() : NearestNeighbors object = ", nbrs)
            
        local_indices_1 = nbrs.kneighbors(np.array([angle_1, energy_1]).reshape(1,-1), return_distance=False)
        local_indices_2 = nbrs.kneighbors(np.array([angle_2, energy_2]).reshape(1,-1), return_distance=False)
            
        indices_1 = indices_np[local_indices_1]
        indices_2 = indices_np[local_indices_2]
        
        # Iterate over the dataset and collect the event data for the nearest neighbors
        events_1 = torch.zeros(num_neighbors, 16, 40, 19)
        events_2 = torch.zeros(num_neighbors, 16, 40, 19)
        
        energies_1 = torch.zeros(num_neighbors, 1)
        energies_2 = torch.zeros(num_neighbors, 1)
        
        angles_1 = torch.zeros(num_neighbors, 1)
        angles_2 = torch.zeros(num_neighbors, 1)
        
        events_1_index = 0
        events_2_index = 0
        
        for data in data_iter:
            data_indices = data[4]
            for i, index in enumerate(data_indices):
                if index.numpy().item() in indices_1:
                    events_1[events_1_index] = data[0][i][:,:,:19]
                    energies_1[events_1_index] = data[2][i][0]
                    angles_1[events_1_index] = data[3][i][1]
                    events_1_index = events_1_index + 1
                if index.numpy().item() in indices_2:
                    events_2[events_2_index] = data[0][i][:,:,:19]
                    energies_2[events_2_index] = data[2][i][0]
                    angles_2[events_2_index] = data[3][i][1]
                    events_2_index = events_2_index + 1
        
        # Generate the latent vectors corresponding to the first cluster
        self.data = events_1
        z_gen_1 = self.forward(mode="generate_latents")["z_gen"]
        
        self.data = events_2
        z_gen_2 = self.forward(mode="generate_latents")["z_gen"]
        
        # Initialize the dump arrays
        save_arr_keys = ["energies_1", "energies_2", "angles_1", "angles_2"]
        save_arr_values = [energies_1.numpy(), energies_2.numpy(), angles_1.numpy(), angles_2.numpy()]
        
        # Compute the mean of the two latent cluster
        z_gen_1 = np.mean(z_gen_1, axis=0).reshape(-1)
        z_gen_2 = np.mean(z_gen_2, axis=0).reshape(-1)
        
        # Compute the alpha values to use given the number of intervals
        alpha = 0.0
        interval = 1.0/intervals
        
        # Initialize a placeholder for the z latent vectors
        z_tensor_shape = [intervals+1]
        z_tensor_shape.extend(z_gen_1.shape)
        self.data = torch.zeros(z_tensor_shape, dtype=torch.float32, device=self.devids[0])
        
        i = 0
        
        # Iterate over the different alpha values and add the inter-
        # polated vectors to the tensor
        while alpha <= 1.0:
            
            self.data[i] = torch.from_numpy(alpha*z_gen_1 + (1-alpha)*z_gen_2)
            i = i + 1
            alpha = alpha + interval
            
        # Call the foward method with decoder mode
        res = self.forward(mode="decode", forward_type="decode")
        
        # Dump the decoded events into a .npz file
        for key in event_dump_keys:
                if key in res.keys():
                    save_arr_keys.append(key)
                    save_arr_values.append(res[key])
                
        # Save the interpolated events to the disk
        np.savez(self.dirpath + "/" + model_status + "_interpolations.npz",
                 **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
        
    def save_state(self, model_type="latest"):
            
        filename = self.dirpath+"/"+str(self.config.model[1])+"_"+model_type+".pth"
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        return filename
    
    
    def restore_state(self, weight_file):
        
        weight_file = self.config.restore_state
        
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f, map_location=self.devids[0])
            
            print("Loading weights from file : {0}".format(weight_file))
            
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                
            # load iteration count
            self.iteration = checkpoint['global_step']