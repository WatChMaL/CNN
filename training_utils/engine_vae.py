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
import numpy as np

# WatChMaL imports
from io_utils import ioconfig
from io_utils.data_handling import WCH5Dataset
from plot_utils.notebook_utils import CSVData
import training_utils.loss_funcs as loss_funcs

# Logging and dumping keys : values to save during logging or dummping
log_keys = ["loss", "mse_loss", "kl_loss", "ce_loss", "accuracy"]
event_dump_keys = ["recon", "z", "mu", "logvar", "z_prime", "predicted_label", "softmax"]

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
        elif self.model_train_type is "train_ae_or_vae_only":
            self.optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)
        elif self.model_train_type is "train_bottleneck_only":
            if type(self.model) is nn.DataParallel:
                self.optimizer = optim.Adam(self.model.module.bottleneck.parameters(),lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.bottleneck.parameters(),lr=learning_rate)
                
        elif self.model_train_type is "train_classifier_only":
            if type(self.model) is nn.DataParallel:
                self.optimizer = optim.Adam(self.model.module.classifier.parameters(),lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.classifier.parameters(),lr=learning_rate)
        
        # Declare the loss function
        if model_variant is "AE":
            if self.model_train_type is "train_all"
                self.criterion = loss_funcs.AECLLoss
            elif self.model_train_type is "train_classifier_only":
                self.criterion = nn.CrossEntropyLoss()
            elif self.model_train_type is "train_bottleneck_only":
                self.criterion = nn.MSELoss()
        elif model_variant is "VAE":
            if self.model_train_type is "train_all"
                self.criterion = loss_funcs.VAECLLoss
            elif self.model_train_type is "train_classifier_only":
                self.criterion = nn.CrossEntropyLoss()
            elif self.model_train_type is "train_bottleneck_only":
                self.criterion = loss_funcs.VAELoss
        
        # Placeholders for data and labels
        self.data=None
        self.label=None
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
        
        # Move the data to the user-specified device
        self.data = self.data.to(self.device)
        self.data = self.data.permute(0,3,1,2)
        
        # Move the labels to the user-specified device
        self.label = self.label.to(self.device)
        
        # Set the grad calculation mode
        grad_mode = True if forward_type is "train" else False
        
        # Return dict
        return_dict = None
        
        with torch.set_grad_enabled(grad_mode):
            
            if mode == "all":
                
                # Forward for VAE
                if self.model_variant is "VAE":
                    
                    # Collect the output from the model
                    recon, z, mu, logvar, z_prime, predicted_label = self.model(self.data, mode, device=self.devids[0])
                    loss, mse_loss, kl_loss, ce_loss = self.criterion(recon, self.data, mu,
                                                                      logvar, predicted_label, self.label)
                    self.loss = loss
                    
                    softmax          = self.softmax(predicted_label).cpu().detach().numpy()
                    predicted_label  = torch.argmax(predicted_label, dim=-1)
                    accuracy         = (predicted_label == self.labels).sum().item() / float(predicted_label.nelement())

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"            : loss.cpu().detach().item(),
                                   "mse_loss"        : mse_loss.cpu().detach().item(),
                                   "kl_loss"         : kl_loss.cpu().detach().item(),
                                   "ce_loss"         : ce_loss.cpu().detach().item(),
                                   "accuracy"        : accuracy,
                                   "recon"           : recon.cpu().detach().numpy(),
                                   "z"               : z.cpu().detach().numpy(),
                                   "mu"              : mu.cpu().detach().numpy(),
                                   "logvar"          : logvar.cpu().detach().numpy(),
                                   "z_prime"         : z_prime.cpu().detach().numpy(),
                                   "predicted_label" : predicted_label.cpu().detach().numpy(),
                                   "softmax"         : softmax.cpu().detach.numpy()}
                    
                # Forward for AE
                elif self.model_variant is "AE";
                
                    recon, predicted_label = self.model(self.data, mode, device=self.devids[0])
                    loss = self.criterion(recon, self.data, predicted_label, self.label)
                    self.loss = loss
                    
                    softmax          = self.softmax(predicted_label).cpu().detach().numpy()
                    predicted_label  = torch.argmax(predicted_label, dim=-1)
                    accuracy         = (predicted_label == self.labels).sum().item() / float(predicted_label.nelement())
                    
                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"            : loss.cpu().detach().item(),
                                   "recon"           : recon.cpu().detach().numpy(),
                                   "predicted_label" : predicted_label.cpu().detach().numpy()}

            elif mode == "ae_or_vae":
                
                # Forward for VAE
                if self.model_variant is "VAE":

                    # Collect the output from the model
                    recon, z, mu, logvar, z_prime = self.model(self.data, mode, device=self.devids[0])
                    loss, mse_loss, kl_loss = self.criterion(recon, self.data, mu, 
                                                             logvar, self.iteration, self.num_iterations)
                    self.loss = loss

                    # Restore the shape of recon
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"            : loss.cpu().detach().item(),
                                   "mse_loss"        : mse_loss.cpu().detach().item(),
                                   "kl_loss"         : kl_loss.cpu().detach().item(),
                                   "recon"           : recon.cpu().detach().numpy(),
                                   "z"               : z.cpu().detach().numpy(),
                                   "mu"              : mu.cpu().detach().numpy(),
                                   "logvar"          : logvar.cpu().detach().numpy(),
                                   "z_prime"         : z_prime.cpu().detach().numpy()}
                        
                # Forward for AE
                elif self.model_variant is "AE":

                    recon = self.model(self.data, mode, device=self.devids[0])
                    loss = self.criterion(recon, self.data)
                    self.loss = loss
                    recon = recon.permute(0,2,3,1)

                    return_dict = {"loss"       : loss.cpu().detach().item(),
                                    "recon"     : recon.cpu().detach().numpy()}
                
        elif mode == "generate_latents":
                    
            # Generate only the latent vectors
            z_gen = self.model(self.data, mode, device=self.devids[0])

            return_dict = {"z_gen" : z_gen.cpu().detach().numpy()}
                    
        elif mode == "classifier":
                    
            predicted_label = self.model(self.data, mode, device=self.devids[0])
            loss = self.criterion(predicted_label, self.label)
            self.loss = loss
            
            softmax          = self.softmax(predicted_label).cpu().detach().numpy()
            predicted_label  = torch.argmax(predicted_label,dim=-1)
            accuracy         = (predicted_label == self.labels).sum().item() / float(predicted_label.nelement())
                    
            return_dict = {"loss"            : loss.cpu().detach().item(),
                           "accuracy"        : accuracy
                           "predicted_label" : predicted_label.cpu().detach().numpy(),
                           "softmax"         : softmax.cpu().detach().numpy()}
            
        elif mode == "sample":
            
            samples = self.model(None, mode, device=self.devids[0])
            
            return_dict = {"samples" : samples.permute(0,2,3,1).cpu().detach().numpy()}
  
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
        self.val_log = CSVData(self.dirpath+'val_test.csv')
        
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
                self.data = data[0][:,:,:,:19].float()
                self.labels = data[1].long()
                
                # Update the global iteration counter
                self.iteration = iteration
                
                # Setup the mode to call the forward method
                if self.model_train_type is "train_all":
                    mode = "all"
                elif self.model_train_type is "train_ae_or_vae_only" or self.model_train_type is "train_bottleneck_only":
                    mode = "ae_or_vae"
                elif self.model_train_type is "train_classifier_only":
                    mode = "classifier"

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
                if i==0 or (i+1)%report_interval == 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' %
                          (iteration, epoch, res['loss']))
                    
                # Run validation on user-defined intervals
                if iteration%valid_interval == 0:
                        
                    self.model.eval()
                    val_data = next(iter(self.val_iter))

                    # Extract the event data from the input data tuple
                    self.data = val_data[0][:,:,:,:19].float()

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
                    
                    # Save the best model
                    self.save_state(model_type="latest")

                    self.val_log.write()
                    self.model.train()
                    
                if epoch >= epochs:
                    break
                    
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
            
        self.val_log.close()
        self.train_log.close()
        
    # Method to validate the model training on the validation set
    def validate(self):
        
        # Variables to output at the end
        val_loss = 0.0
        val_iteration = 0
        
        # CSV file for logging the output
        self.validation_log = CSVData(self.dirpath+"validation_log.csv")
            
        # Variables to save the actual and reconstructed events
        np_event_path = self.dirpath + "/val_iteration_"
        
        # Extract the event data and label from the DataLoader iterator
        for val_data in iter(self.val_iter):
            
            sys.stdout.write("val_iterations : " + str(val_iteration) + "\n")

            # Extract the event data from the input data tuple
            self.data = val_data[0][:,:,:,:19].float()

            # Setup the mode to call the forward method
            if self.model_train_type is "train_all":
                mode = "all"
            elif self.model_train_type is "train_ae_or_vae_only" or self.model_train_type is "train_bottleneck_only":
                mode = "ae_or_vae"
            elif self.model_train_type is "train_classifier_only":
                mode = "classifier"
                    
            res = self.forward(mode=mode, forward_type="validation")
                 
            keys = ["epoch"]
            values = [val_iteration]
            for key in log_keys:
                if key in res.keys():
                    keys.append(key)
                    values.append(res[key])
            
            # Log/Report
            self.validation_log.record(keys, values)
            self.validation_log.write()
            
            # Save the actual and reconstructed event to the disk
            if val_iteration == 0:
                save_arr_keys = ["events", "labels", "energies"]
                save_arr_values = [self.data.cpu().numpy(), val_data[1], val_data[3]]

                for key in event_dump_keys:
                    if key in res.keys():
                        save_arr_keys.append(key)
                        save_arr_values.append(res[key])
                
                np.savez(np_event_path + str(val_iteration) + ".npz",
                         **{key:value for key,value in zip(save_arr_keys,save_arr_values)})
            
            val_iteration += 1
        
    # Sample vectors from the normal distribution and decode to reconstruct events
    def sample(self, num_samples=10, trained=False):
        
        # Setup the path
        sample_save_path = self.dirpath + 'samples/'
        
        # Check if the iteration has not been specified
        if self.iteration is None:
            self.iteration = 0
            
        # Create the directory if it does not already exist
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)
            
        model_status = "trained" if trained else "untrained"
        
        sample_save_path = sample_save_path + str(self.config.model[1])
        
        # Create the directory if it does not already exist
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)
            
        # Samples list
        sample_list = []
        
        # Iterate over the counter
        for i in range(num_samples):
            
            with torch.no_grad():

                res = self.forward(mode="sample", forward_type="sample")
                sample_list.extend(res["samples"])
        
        # Convert the list to an numpy array and save to the given path
        np.save(sample_save_path + '/' + model_status + "_samples".format(str(num_samples)) + ".npy", np.array(sample_list))
        
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
                if i==0 or (i+1)%report_interval == 0:
                    print("... Training data iteration %d ..." %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19],data[1], data[3]
                
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
                if i==0 or (i+1)%report_interval == 0:
                    print("... Validation data iteration %d ..." %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19],data[1], data[2]

                res = self.forward(mode="generate")

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