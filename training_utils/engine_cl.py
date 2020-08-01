"""
engine_cl.py

Derived engine class for training a fully supervised classifier
"""

# +
import pdb
import os.path
from os import path

# Python standard imports
from sys import stdout
from math import floor, ceil
from time import strftime, localtime, time
import numpy as np
import random
import csv
import gc
import resource

# +
# Numerical imports
from numpy import savez

from numba import cuda
# -

# PyTorch imports
from torch import cat
from torch import argmax
from torch import tensor
from torch.nn import Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from io_utils.custom_samplers import SubsetSequenceSampler
from torchviz import make_dot

# WatChMaL imports
from training_utils.engine import Engine
from training_utils.loss_funcs import CELoss,weighted_CELoss_factory
from plot_utils.notebook_utils import CSVData

# Global variables
_SOFTMAX   = Softmax(dim=1)
_LOG_KEYS  = ["loss", "accuracy"]
_DUMP_KEYS = ["predicted_labels", "softmax"]

class EngineCL(Engine):
    
    def __init__(self, model, config):
        super().__init__(model, config)
        
        if config.loss_weights is not None:
            loss_weights = tensor(config.loss_weights).to(self.device) if self.device != "cpu" else tensor(config.loss_weights)
            self.criterion = weighted_CELoss_factory(loss_weights)
        else:
            self.criterion = CELoss
        
        if config.train_all:
            self.optimizer=Adam(self.model_accs.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            print("Entire model parameters passed to the optimizer")
        else:
            self.optimizer=Adam(self.model_accs.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            print("Only model.classifier parameters passed to the optimizer")
        
        # Assert that we have samples to train the classifier
        assert config.cl_ratio > 0, "Set config.cl_ratio > 0 since samples needed to train the classifier"
        
        
        for i in np.arange(config.num_datasets):
            # Split the dataset into labelled and unlabelled subsets
            # Note : Only the labelled subset will be used for classifier training
            n_cl_train = int(len(self.train_dset.train_indices[i]) * config.cl_ratio)
            n_cl_val = int(len(self.val_dset.val_indices[i]) * config.cl_ratio)

            if i == 0:
                self.train_indices = np.array(self.train_dset.train_indices[i][:n_cl_train])
                self.val_indices = np.array(self.val_dset.val_indices[i][:n_cl_val])
                self.test_indices = np.array(self.test_dset.test_indices[i])
            else:
                self.train_indices = np.concatenate((self.train_indices,self.train_dset.train_indices[i]),axis=0)
                self.train_indices = np.concatenate((self.val_indices,self.val_dset.val_indices[i]),axis=0)
                self.test_indices = np.concatenate((self.test_indices, self.test_dset.test_indices[i]),axis=0)
            
        
        
        
        # Initialize the torch dataloaders
  
        self.train_loader = DataLoader(self.train_dset, batch_size=self.config.batch_size_train, shuffle=False,
                                           pin_memory=False, sampler=SubsetRandomSampler(self.train_indices), num_workers=8)
        self.val_loader = DataLoader(self.val_dset, batch_size=self.config.batch_size_val, shuffle=False,
                                           pin_memory=False, sampler=SubsetRandomSampler(self.val_indices), num_workers=8)
        self.test_loader = DataLoader(self.test_dset, batch_size=self.config.batch_size_test, shuffle=False,
                                           pin_memory=False, sampler=SubsetSequenceSampler(self.test_indices), num_workers=8)

        # Define the placeholder attributes
        self.data     = None
        self.labels   = None
        self.energies = None
        self.eventids = None
        self.rootfiles = None
        self.angles = None
        self.index = None
        
    def forward(self, mode):
        """Overrides the forward abstract method in Engine.py.
        
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """

        if self.data is not None and len(self.data.size()) == 4:
            self.data = self.data.to(self.device)
            #self.data = self.data.permute(0,3,1,2)
            
        if self.labels is not None:
            self.labels = self.labels.to(self.device)
        
        # Set the correct grad_mode given the mode
        if mode == "train":
            grad_mode = True
            self.model.train()
        elif mode == "validation":
            grad_mode= False
            self.model.eval()
            
        predicted_labels = self.model(self.data)
        #print(predicted_labels.size(), self.labels.size())
        loss             = self.criterion(predicted_labels, self.labels)
        self.loss        = loss
   
        softmax          = _SOFTMAX(predicted_labels)
        pred_labels      = argmax(predicted_labels, dim=-1)
        accuracy         = (pred_labels == self.labels).sum().item() / float(pred_labels.nelement())
        
        #if self.data is not None and len(self.data.size()) == 4:
            #self.data = self.data.permute(0,2,3,1)
                    
        return {"loss"               : loss.cpu().detach().item(),
                "predicted_labels"   : pred_labels.cpu().detach().numpy(),
                "softmax"            : softmax.cpu().detach().numpy(),
                "accuracy"           : accuracy,
                "raw_pred_labels"    : predicted_labels}
    
    def train(self):
        """Overrides the train method in Engine.py.
        
        Args: None
        """
        epochs          = self.config.epochs
        report_interval = self.config.report_interval
        num_vals        = self.config.num_vals
        num_val_batches = self.config.num_val_batches
 
        # Set the iterations at which to dump the events and their metrics
        dump_iterations = self.set_dump_iterations(self.train_loader)
        print(f"Validation Interval: {dump_iterations[0]}")
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to upadte when saving the best model
        best_loss = 1000000.
        
        # Initialize the iterator over the validation subset
        val_iter = iter(self.val_loader)

        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            times = []

            start_time = time()

            # Local training loop for a single epoch
            #print('entering loop')
            for data in self.train_loader:
                #print('in loop')    

                # Using only the charge data
                self.data     = data[0][:,:,:,:].float()
                self.labels   = data[1].long()
                self.energies = data[2]
                self.angles = data[3]
                self.index = data[4]


                # Do a forward pass using data = self.data
                res = self.forward(mode="train")

                # Do a backward pass using loss = self.loss
                self.backward()

                # Update the epoch and iteration
                epoch     += 1./len(self.train_loader)
                iteration += 1

                # Iterate over the _LOG_KEYS and add the vakues to a list
                keys   = ["iteration", "epoch"]
                values = [iteration, epoch]

                for key in _LOG_KEYS:
                    if key in res.keys():
                        keys.append(key)
                        values.append(res[key])

                # Record the metrics for the mini-batch in the log
                self.train_log.record(keys, values)
                self.train_log.write()

                # Print the metrics at given intervals
                if iteration == 0 or iteration%report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f" %
                          (iteration, epoch, res["loss"], res["accuracy"]))

                # Save the model computation graph to a file
                """if iteration == 1:
                    graph = make_dot(res["raw_pred_labels"], params=dict(list(self.model_accs.named_parameters())))
                    graph.render(self.dirpath + "/model", view=False)
                    break"""

                # Run validation on given intervals
                if iteration%dump_iterations[0] == 0:
                    curr_loss = 0.
                    val_batch = 0

                    keys = ['iteration','epoch']
                    values = [iteration, epoch]

                    local_values = []

                    for val_batch in range(num_val_batches):

                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(self.val_loader)

                        # Extract the event data from the input data tuple
                        self.data     = val_data[0][:,:,:,:].float()
                        self.labels   = val_data[1].long()
                        self.energies = val_data[2].float()
                        self.angles = val_data[3].float()
                    
                        res = self.forward(mode="validation")

                        if val_batch == 0:
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    keys.append(key)
                                    local_values.append(res[key])
                        else:
                            log_index = 0
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    local_values[log_index] += res[key]
                                    log_index += 1

                        curr_loss += res["loss"]

                    for local_value in local_values:
                        values.append(local_value/num_val_batches)

                    # Record the validation stats to the csv
                    self.val_log.record(keys, values)

                    # Average the loss over the validation batch
                    curr_loss = curr_loss / num_val_batches

                    # Save the best model
                    if curr_loss < best_loss:
                        self.save_state(mode="best")
                        curr_loss = best_loss

                    if iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies", "angles"]
                        save_arr_values = [self.data.cpu().numpy(), self.labels.cpu().numpy(), self.energies.cpu().numpy(), self.angles.cpu().numpy()]
                        for key in _DUMP_KEYS:
                            if key in res.keys():
                                save_arr_keys.append(key)
                                save_arr_values.append(res[key])

                        # Save the actual and reconstructed event to the disk
                        savez(self.dirpath + "/iteration_" + str(iteration) + ".npz",
                              **{key:value for key,value in zip(save_arr_keys,save_arr_values)})

                    self.val_log.write()

                    # Save the latest model
                    self.save_state(mode="latest")

                if epoch >= epochs:
                    break

            print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f" %
                  (iteration, epoch, res['loss'], res['accuracy']))

            
        self.val_log.close()
        self.train_log.close()
        
    def validate(self, subset):
        """Overrides the validate method in Engine.py.
        
        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        """
        # Print start message
        if subset == "train":
            message = "Validating model on the train set"
        elif subset == "validation":
            message = "Validating model on the validation set"
        elif subset == "test":
            message = "Validating model on the test set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None
        
        print(message)
        
        num_dump_events = self.config.num_dump_events
        
        # Setup the CSV file for logging the output, path to save the actual and reconstructed events, dataloader iterator
        if subset == "train":
            self.log        = CSVData(self.dirpath+"train_validation_log.csv")
            np_event_path   = self.dirpath + "/train_valid_iteration_"
            data_iter       = self.train_loader
            dump_iterations = max(1, ceil(num_dump_events/self.config.batch_size_train))
        elif subset == "validation":
            self.log        = CSVData(self.dirpath+"valid_validation_log.csv")
            np_event_path   = self.dirpath + "/val_valid_iteration_"
            data_iter       = self.val_loader
            dump_iterations = max(1, ceil(num_dump_events/self.config.batch_size_val))
        else:
            self.log        = CSVData(self.dirpath+"test_validation_log.csv")
            np_event_path   = self.dirpath + "/test_validation_iteration_"
            data_iter       = self.test_loader
            dump_iterations = max(1, ceil(num_dump_events/self.config.batch_size_test))
        
        print("Dump iterations = {0}".format(dump_iterations))
        save_arr_dict = {"events":[], "labels":[], "energies":[], "angles":[], "eventids":[], "rootfiles":[]}

        avg_loss = 0
        avg_acc = 0
        count = 0
        for iteration, data in enumerate(data_iter):
            
            stdout.write("Iteration : " + str(iteration) + "\n")

            # Extract the event data from the input data tuple
            self.data     = data[0][:,:,:,:].float()
            self.labels   = data[1].long()
            self.energies = data[2].float()
            self.eventids = data[5].float()
            self.rootfiles = data[6]
            self.angles = data[3].float()
            
                    
            res = self.forward(mode="validation")
                 
            keys   = ["iteration"]
            values = [iteration]
            for key in _LOG_KEYS:
                if key in res.keys():
                    keys.append(key)
                    values.append(res[key])
            
            # Log/Report
            self.log.record(keys, values)
            self.log.write()
            
            # Add the result keys to the dump dict in the first iterations
            if iteration == 0:
                for key in _DUMP_KEYS:
                    if key in res.keys():
                        save_arr_dict[key] = []
            
            avg_acc += res['accuracy']
            avg_loss += res['loss']
            count += 1

            if iteration < dump_iterations:
                save_arr_dict["labels"].append(self.labels.cpu().numpy())
                save_arr_dict["energies"].append(self.energies.cpu().numpy())
                save_arr_dict["eventids"].append(self.eventids.cpu().numpy())
                save_arr_dict["rootfiles"].append(self.rootfiles)
                save_arr_dict["angles"].append(self.angles.cpu().numpy())
                
                for key in _DUMP_KEYS:
                    if key in res.keys():
                        save_arr_dict[key].append(res[key])
            elif iteration == dump_iterations:
                break
        
        print("Saving the npz dump array :")
        savez(np_event_path + "dump.npz", **save_arr_dict)
        avg_acc /= count
        avg_loss /= count
        stdout.write("Overall acc : {}, Overall loss : {}\n".format(avg_acc, avg_loss))

