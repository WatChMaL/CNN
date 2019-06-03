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
import numpy as np

# WatChMaL imports
from io_utils.data_handling import WCH5Dataset
from visualization_utils.notebook_utils import CSVData

# Class for the training engine for the WatChMaLVAE
class EngineVAE:
    
    """
    Purpose : Training engine for the WatChMaLVAE. Performs training, validation,
              and testing of the models
    """
    # Initializer for the training engine object
    def __init__(self, model, config):
        
        # Initialize the model
        self.model = model
        
        # Determine the device(s) to be used
        if (config.device == 'gpu') and config.gpu_list:
            
            print("Requesting GPU :\nConfig GPU list = {0}".format(config.gpu_list))
            
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]
            print("Main GPU = "+self.devids[0])
            
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on {0}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)
                else:
                    print("Unable to use GPUs")
                    self.device = torch.device("cpu")
            else:
                print("Unable to use GPUs")
                self.device = torch.device("cpu")
                
        # Move the model to the selected device
        self.model.to(self.device)
        
        # Initialize the optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(),eps=1e-3)
        self.criterion = self.VAELoss
        self.recon_loss = nn.MSELoss()
        
        # Engine variables for the input data, labels and number of iterations
        self.data=None
        self.labels=None
        self.iteration=None
        
        # Create a dataset object to iterate over the dataset stored on the disk
        self.dset=WCH5Dataset(config.path,
                              config.val_split,
                              config.test_split,
                              shuffle=config.shuffle,
                              reduced_dataset_size=config.subset)

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
        
        # Directory to save the model and data information
        
        self.dirpath=config.save_path
        
        self.data_description=config.data_description
        
        try:
            os.stat(self.dirpath)
        except:
            print("Making a directory for model data : {}".format(self.dirpath))
            os.mkdir(self.dirpath)
        
        # Add the path for the data type to the dirpath
        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.dirpath=self.dirpath + self.data_description + "/" + self.start_time_str

        try:
            os.stat(self.dirpath)
        except:
            print("Making a directory for model data for data prepared at: {}".format(self.dirpath))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config
        
    # Loss function for the VAE combining MSELoss and KL-divergence
    def VAELoss(self, reconstruction, mean, log_var, data):
        
        # MSE Reconstruction Loss
        mse_loss = self.recon_loss(reconstruction, data)
        
        # KL-divergence Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return mse_loss + kl_loss
    
        
    # Method to compute the loss using the forward pass
    def forward(self,train=True):
        
        with torch.set_grad_enabled(train):
            
            # Move the data to the user-specified device
            self.data = self.data.to(self.device)
                        
            # Prediction
            self.data = self.data.permute(0,3,1,2)
            prediction, mu, covar = self.model(self.data)
            
            # Training
            loss = -1
            loss = self.criterion(prediction, mu, covar, self.data)
            self.loss = loss
            
            # Restore the shape of the data and the prediction
            self.data = self.data.permute(0,2,3,1)
            prediction = prediction.permute(0,2,3,1)
        
        return {"loss"       : loss.cpu().detach().item(),
                "prediction" : prediction.cpu().detach().numpy(),
                "mu"         : mu.cpu().detach().numpy(),
                "covar"      : covar.cpu().detach().numpy()}
        
    def backward(self):
        
        self.optimizer.zero_grad()  # Reset gradient accumulation
        self.loss.backward()        # Propagate the loss backwards
        self.optimizer.step()       # Update the optimizer parameters
        
    def train(self, epochs=3.0, report_interval=10, valid_interval=1000, save_interval=1000):

        # Prepare attributes for data logging
        self.train_log, self.val_log = CSVData(self.dirpath+'/log_train.csv'), CSVData(self.dirpath+'/val_test.csv')
        
        # Variables to save the actual and reconstructed events
        np_event_path = self.dirpath+"/event_vs_recon_iteration_"
        
        # Set neural net to training mode
        self.model.train()
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Training loop
        while (int(epoch+0.5) < epochs):
            
            print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                
                # Move the data to the device specified by the user
                self.data = data[0][:,:,:,:19]
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                
                # Call backward: backpropagate error and update weights
                self.backward()
                
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                # Log/Report
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','loss'],[iteration, epoch, res['loss']])
                self.train_log.write()
                
                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
                    
                # Run validation on user-defined intervals
                if (iteration+1)%valid_interval == 0:
                    
                    with torch.no_grad():
                        
                        self.model.eval()
                        val_data = next(iter(self.val_iter))
                        
                        # Extract the event data from the input data tuple
                        self.data = val_data[0][:,:,:,:19]
                        
                        res = self.forward(False)
                        
                        # Save the actual and reconstructed event to the disk
                        np.savez(np_event_path + str(iteration) + ".npz",
                                 event=self.data.cpu().numpy(), recon=res['prediction'], mu=res["mu"], covar=res["covar"])
                        
                        self.val_log.record(['iteration','epoch','loss'], [iteration, epoch, res['loss']])
                        
                        self.val_log.write()
                        
                    self.model.train()
                    
                if epoch >= epochs:
                    
                    break
                    
                # Save the model and optimizer state on user-defined intervals
                if(iteration+1)%save_interval == 0:
                    self.save_state(curr_iter=1)
                    
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
            
        self.val_log.close()
        self.train_log.close()
        
    def save_state(self, curr_iter=0):
        
        state_dir = self.config.save_path+"/saved_states"
        
        if not os.path.exists(state_dir):
            os.mkdir(state_dir)
            
        filename = state_dir+"/"+str(self.config.model[1])+ "_iter_" + str(curr_iter)
        
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
        
        weight_file = self.config.save_path+'/saved_states/'+weight_file
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f,map_location="cuda:0")
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']