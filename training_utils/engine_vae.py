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
from torch.utils.data.sampler import SubsetRandomSampler

# Standard and data processing imports
import os
import sys
import time
import math
import random
import numpy as np

# WatChMaL imports
from io_utils import ioconfig
from io_utils.data_handling import WCH5Dataset
from plot_utils.notebook_utils import CSVData
import training_utils.loss_funcs as loss_funcs

# Logging and dumping keys : values to save during logging or dummping
log_keys = ["loss", "recon_loss", "kl_loss", "ce_loss", "mse_loss", "accuracy", "logdet"]
event_dump_keys = ["recon", "z", "mu", "logvar", "z_prime", "softmax", "samples", "predicted_labels", "predicted_energies", "z_k", "logdet"]

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
                    
                    if forward_type is "training":
                        loss, recon_loss, kl_loss = self.criterion(recon, self.data, 
                                                                   mu, logvar)
                        
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
                
                recon, z_k, logdet, z, mu, logvar, z_prime = self.model(self.data, mode, device=self.devids[0])
                loss, recon_loss, kl_loss, logdet = self.criterion(recon, self.data, mu, logvar, logdet)
                
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
    def validate(self, subset="validation"):
        
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
        iteration = 0
        
        # Setup the CSV file for logging the output, path to save the actual and reconstructed events, dataloader iterator
        if subset is "train":
            self.log = CSVData(self.dirpath+"train_validation_log.csv")
            np_event_path = self.dirpath + "/train_valid_iteration_"
            data_iter = self.train_iter
        elif subset is "validation":
            self.log = CSVData(self.dirpath+"valid_validation_log.csv")
            np_event_path = self.dirpath + "/val_valid_iteration_"
            data_iter = self.val_iter
        else:
            self.log = CSVData(self.dirpath+"test_validation_log.csv")
            np_event_path = self.dirpath + "/test_valid_iteration_"
            data_iter = self.test_iter
        
        # List holding the keys for values to dump at the end
        metric_dump_keys = ["indices", "recon_loss", "kl_loss"]
        
        # Array holding all the dataset indices
        global_indices = []
        recon_loss_values = []
        kl_loss_values = []
        
        for data in iter(data_iter):
            
            sys.stdout.write("Iteration : " + str(iteration) + "\n")

            # Extract the event data from the input data tuple
            self.data     = data[0][:,:,:,:19].float()
            self.labels   = data[1].long()
            self.energies = data[2].float()
            indices       = data[3].cpu().numpy()
            
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
            values = [iteration]
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
            
            # Save the actual and reconstructed event to the disk
            if iteration is 0:
                save_arr_keys = ["events", "labels", "energies"]
                save_arr_values = [self.data.cpu().numpy(), data[1], data[2]]

                for key in event_dump_keys:
                    if key in res.keys():
                        save_arr_keys.append(key)
                        save_arr_values.append(res[key])
                
                np.savez(np_event_path + str(iteration) + ".npz",
                         **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
            
            iteration += 1
            
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
        
    # Interpolate b/w two samples from the dataset
    
    def interpolate(self, subset="validation", intervals=10, trained=False):
        
        assert subset in ["train", "validation", "test"]
        
        model_status = "trained" if trained else "untrained"
        
        if subset is "train":
            data_iter = self.train_iter
        elif subset is "validation":
            data_iter = self.val_iter
        else:
            data_iter = self.test_iter
            
        # Read the data, labels and energies from data loader
        data, labels, energies = next(iter(data_iter))
        
        # Use only the charge data for the events
        i, j = random.randint(0, data.size(0)-1), random.randint(0, data.size(0)-1)
        
        # Initialize the tensor to be used for the forward pass
        tensor_shape = [2]
        tensor_shape.extend(data[i].size()[:2])
        tensor_shape.extend([19])
        
        self.data = torch.zeros(tensor_shape, dtype=torch.float32)
        
        # Insert the event values in the placeholder zero tensors
        self.data[0], self.data[1] = data[i][:,:,:19], data[j][:,:,:19]
        
        # Initialize the dump arrays
        save_arr_keys = ["events", "labels", "energies"]
        save_arr_values = [self.data.cpu().numpy(), [labels[i], labels[j]], [energies[i], energies[j]]]

        # Forward pass
        z_gen = self.forward(mode="generate_latents")["z_gen"]
        
        # Compute the alpha values to use given the number of intervals
        alpha = 0.0
        interval = 1.0/intervals
        
        # Initialize a placeholder for the z latent vectors
        z_tensor_shape = [intervals+1]
        z_tensor_shape.extend(z_gen.shape[1:])
        self.data = torch.zeros(z_tensor_shape, dtype=torch.float32, device=self.devids[0])
        
        i = 0
        
        # Iterate over the different alpha values and add the inter-
        # polated vectors to the tensor
        while alpha <= 1.0:
            
            self.data[i] = torch.from_numpy(alpha*z_gen[0] + (1-alpha)*z_gen[1])
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