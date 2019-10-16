"""
engine_ssl.py

Derived engine class for training a deep generative model
for semi-supervised learning
"""

# Python standard imports
from sys import stdout
from math import floor, ceil
from time import strftime, localtime

# Numerical imports
from numpy import savez

# PyTorch imports
from torch import argmax, cat, mean
from torch.nn import Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# WatChMaL imports
from training_utils.engine import Engine
from training_utils.loss_funcs import M2UnlabelledLoss, M2LabelledLoss
from plot_utils.notebook_utils import CSVData

# Global variables
_LOG_KEYS  = ["loss", "recon_loss", "kl_loss", "ce_loss", "accuracy"]
_DUMP_KEYS = ["recon", "z", "mu", "logvar", "softmax"]
_SOFTMAX   = Softmax(dim=1)

class EngineSSL(Engine):
    
    def __init__(self, model, config):
        super().__init__(model, config)
        
        self.u_criterion = M2UnlabelledLoss
        self.l_criterion = M2LabelledLoss
        
        # Setup the optimizer with the correct parameters
        if config.train_all:
            self.optimizer=Adam(self.model_accs.parameters(), lr=config.lr)
        else:
            raise NotImplementedError
            
        # Split the dataset into labelled and unlabelled subsets
        # Note : Both the labelled and unlabelled datasets are used for SSL training
        n_cl_train = int(len(self.dset.train_indices) * config.cl_ratio)
        n_cl_val   = int(len(self.dset.val_indices) * config.cl_ratio)
        
        self.u_train_indices = self.dset.train_indices[n_cl_train:]
        self.l_train_indices = self.dset.train_indices[:n_cl_train]
            
        self.u_val_indices = self.dset.train_indices[n_cl_val:]
        self.l_val_indices = self.dset.train_indices[:n_cl_val]
        
        # Determine the mini-batch sizes for the labelled and unlabelled datasets
        l_train_batch_size = round(max(2, (config.cl_ratio/(1. - config.cl_ratio))*config.batch_size_train))
        l_val_batch_size   = round(max(2, (config.cl_ratio/(1. - config.cl_ratio))*config.batch_size_val))
        
        # Initialize the torch dataloaders
        self.u_train_loader = DataLoader(self.dset, batch_size=config.batch_size_train, shuffle=False,
                                         pin_memory=True, sampler=SubsetRandomSampler(self.u_train_indices))
        self.l_train_loader = DataLoader(self.dset, batch_size=l_train_batch_size, shuffle=False,
                                         pin_memory=True, sampler=SubsetRandomSampler(self.l_train_indices))
        
        self.u_val_loader = DataLoader(self.dset, batch_size=config.batch_size_val, shuffle=False,
                                       pin_memory=True, sampler=SubsetRandomSampler(self.u_val_indices))
        self.l_val_loader = DataLoader(self.dset, batch_size=l_val_batch_size, shuffle=False,
                                       pin_memory=True, sampler=SubsetRandomSampler(self.l_val_indices))
            
        # Define the placeholder attributes
        self.u_data     = None
        self.u_labels   = None
        self.u_energies = None
            
        self.l_data     = None
        self.l_labels   = None
        self.l_energies = None
        
    def forward(self, mode):
        """Overrides the forward abstract method in Engine.py.
        
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """
        
        ret_dict = None
        
        if self.u_data is not None and len(self.u_data.size()) == 4 and mode in ["train", "validation"]:
            self.u_data = self.u_data.to(self.device)
            self.u_data = self.u_data.permute(0,3,1,2)
            
        if self.l_data is not None and len(self.l_data.size()) == 4 and mode in ["train", "validation"]:
            self.l_data = self.l_data.to(self.device)
            self.l_data = self.l_data.permute(0,3,1,2)
        
        if self.u_labels is not None and self.l_labels is not None:
            self.u_labels = self.u_labels.to(self.device)
            self.l_labels  = self.l_labels.to(self.device)
            
        # Set the correct grad_mode given the mode
        if mode == "train":
            grad_mode=True
            self.model.train()
        elif mode in ["validation", "sample", "decode"]:
            grad_mode=False
            self.model.eval()
            
        if mode in ["train", "validation"]:
            u_recon, u_z, u_mu, u_logvar, u_pi = self.model(self.u_data, mode, None)
            l_recon, l_z, l_mu, l_logvar, l_pi = self.model(self.l_data, mode, self.l_labels)
                
            u_loss, u_recon_loss, u_kl_loss, u_h_loss, u_ce_loss = self.u_criterion(u_recon, self.u_data, u_mu, u_logvar, u_pi, self.u_labels)
            l_loss, l_recon_loss, l_kl_loss, l_ce_loss = self.l_criterion(l_recon, self.l_data, l_mu, l_logvar, l_pi, self.l_labels)
            
            loss = ((u_loss*self.u_data.size(0))+(l_loss*self.l_data.size(0)))/(self.u_data.size(0)+self.l_data.size(0))
            self.loss = loss
            
            recon_loss = ((u_recon_loss*self.u_data.size(0))+(l_recon_loss*self.l_data.size(0)))/(self.u_data.size(0)+self.l_data.size(0))
            kl_loss    = ((u_kl_loss*self.u_data.size(0))+(l_kl_loss*self.l_data.size(0)))/(self.u_data.size(0)+self.l_data.size(0))
            ce_loss    = ((u_ce_loss*self.u_data.size(0))+(l_ce_loss*self.l_data.size(0)))/(self.u_data.size(0)+self.l_data.size(0))
            
            pi      = cat((u_pi, l_pi), dim=0)
            softmax = _SOFTMAX(pi)
            
            true_labels      = cat((self.u_labels, self.l_labels), dim=0)
            predicted_labels = argmax(pi, dim=-1)
            accuracy         = (predicted_labels == true_labels).sum().item() / float(predicted_labels.nelement())
            
            recon = cat((u_recon, l_recon), dim=0)
            recon = recon.permute(0,2,3,1)
                
            z      = cat((u_z, l_z), dim=0)
            mu     = cat((u_mu, l_mu), dim=0)
            logvar = cat((u_logvar, u_logvar), dim=0)
            

            ret_dict = {"loss"       : loss.cpu().detach().item(),
                        "recon_loss" : recon_loss.cpu().detach().item(),
                        "kl_loss"    : kl_loss.cpu().detach().item(),
                        "ce_loss"    : ce_loss.cpu().detach().item(),
                        "h_loss"     : u_h_loss.cpu().detach().item(),
                        "accuracy"   : accuracy,
                        "recon"      : recon.cpu().detach().numpy(),
                        "z"          : z.cpu().detach().numpy(),
                        "mu"         : mu.cpu().detach().numpy(),
                        "logvar"     : logvar.cpu().detach().numpy(),
                        "softmax"    : softmax.cpu().detach().numpy()}
            
        elif mode == "sample":
            raise NotImplementedError
            
        elif mode == "decode":
            raise NotImplementedError
            
        if self.u_data is not None and len(self.u_data.size()) == 4 and mode in ["train", "validation"]:
            self.u_data=self.u_data.permute(0,2,3,1)
            
        if self.l_data is not None and len(self.l_data.size()) == 4 and mode in ["train", "validation"]:
            self.l_data=self.l_data.permute(0,2,3,1)

        return ret_dict
    

    def train(self, epochs, report_interval, num_vals, num_val_batches):
        """Overrides the train method in Engine.py.
        
        Args:
        epcohs          -- Number of epochs to train the model for
        report_interval -- Interval at which to report the training metrics to the user
        num_vals        -- Number of validations to perform throughout training
        num_val_batches -- Number of batches to use during each validation
        """
        # Calculate the total number of iterations in this training session
        self.num_iterations=ceil(epochs * len(self.u_train_loader))

        # Set the iterations at which to dump the events and their metrics
        dump_iterations=self.set_dump_iterations(epochs, num_vals, self.u_train_loader)
        
        # Initialize the training labelled data iterator
        l_train_iter = iter(self.l_train_loader)
        
        # Initialize the validation data iterators
        u_val_iter = iter(self.u_val_loader)
        l_val_iter = iter(self.l_val_loader)
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to upadte when saving the best model
        best_loss = 1000000.
            
        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):
            
            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
        
            # Local training loop for a single epoch
            for data in self.u_train_loader:
                
                # Using only the charge data [:19]
                self.u_data     = data[0][:,:,:,:19].float()
                self.u_labels   = data[1].long()
                self.u_energies = data[2]
                
                # Read the data and labels from the labelled dataset
                try: l_data = next(l_train_iter)
                except StopIteration:
                    l_train_iter = iter(self.l_train_loader)
                    l_data = next(l_train_iter)
                    
                self.l_data     = l_data[0][:,:,:,:19].float()
                self.l_labels   = l_data[1].long()
                self.l_energies = l_data[2]
                
                # Update the iteration counter
                self.iteration = iteration
                
                # Call forward: pass the data to the model and compute predictions and loss
                res = self.forward(mode="train")
                
                # Call backward: backpropagate error and update weights
                self.backward()
                
                # Epoch update
                epoch += 1./len(self.u_train_loader)
                iteration += 1
                
                # Iterate over the _LOG_KEYS and add the vakues to a list
                keys = ['iteration','epoch']
                values = [iteration, epoch]
                
                for key in _LOG_KEYS:
                    if key in res.keys():
                        keys.append(key)
                        values.append(res[key])
                        
                # Record the metrics for the mini-batch in the log
                self.train_log.record(keys, values)
                self.train_log.write()
                
                # Print the metrics at given intervals
                if iteration == 0 or (iteration)%report_interval == 0:
                    print("""... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Recon Loss %1.3f ... KL Loss %1.3f ... CE Loss %1.3f ...  Accuracy %1.3f """ % (iteration, epoch, res["loss"], res["recon_loss"], res["kl_loss"], res["ce_loss"], res["accuracy"]))
                    
                # Run validation on user-defined intervals
                if iteration % dump_iterations[0] == 0:
                    
                    curr_loss=0.
                    val_batch=0
                    
                    keys=['iteration', 'epoch']
                    values=[iteration, epoch]
                    
                    local_values=[]
                    
                    for val_batch in range(num_val_batches):
                        
                        # Iterate over the unlabelled validation 
                        # dataset and collect the next mini-batch
                        try:
                            u_val_data = next(u_val_iter)
                        except StopIteration:
                            u_val_iter = iter(self.u_val_loader)
                            u_val_data = next(u_val_iter)
                         
                        # Iterate over the labelled validation
                        # dataset and collect the next sample
                        try:
                            l_val_data = next(l_val_iter)
                        except StopIteration:
                            l_val_iter = iter(self.l_val_loader)
                            l_val_data = next(l_val_iter)
                            
                        self.u_data     = u_val_data[0][:,:,:,:19].float()
                        self.u_labels   = u_val_data[1].long()
                        self.u_energies = u_val_data[2]
                        
                        self.l_data     = l_val_data[0][:,:,:,:19].float()
                        self.l_labels   = l_val_data[1].long()
                        self.l_energies = l_val_data[2]
                        
                        res = self.forward(mode="validation")
                        
                        if val_batch == 0:
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    keys.append(key)
                                    local_values.append(res[key])
                        else:
                            log_index=0
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    local_values[log_index]+=res[key]
                                    log_index+=1

                        curr_loss+=res["loss"]
                        
                    for local_value in local_values:
                        values.append(local_value/num_val_batches)
                            
                    # Record the validation stats to the csv
                    self.val_log.record(keys, values)
                        
                    # Average the loss over the validation batch
                    curr_loss=curr_loss/num_val_batches

                    if iteration in dump_iterations:
                        save_arr_keys=["events", "labels", "energies"]
                        
                        data     = cat((self.u_data, self.l_data), dim=0)
                        labels   = cat((self.u_labels, self.l_labels), dim=0)
                        energies = cat((self.u_energies, self.l_energies), dim=0)
                        
                        save_arr_values=[data.cpu().numpy(), labels.cpu().numpy(), energies.cpu().numpy()]
                        for key in _DUMP_KEYS:
                            if key in res.keys():
                                save_arr_keys.append(key)
                                save_arr_values.append(res[key])

                        # Save the actual and reconstructed event to the disk
                        savez(self.dirpath + "/iteration_" + str(iteration) + ".npz",
                              **{key: value for key, value in zip(save_arr_keys, save_arr_values)})

                    # Save the best model
                    if curr_loss < best_loss:
                        self.save_state(mode="best")

                    # Save the latest model
                    self.save_state(mode="latest")

                    self.val_log.write()

                if epoch >= epochs:
                    break

            print("""... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Recon Loss %1.3f ... KL Loss %1.3f ... 
                  CE Loss %1.3f ...  Accuracy %1.3f """ % (iteration, epoch, res["loss"], res["recon_loss"],
                                                           res["kl_loss"], res["ce_loss"], res["accuracy"]))

        self.val_log.close()
        self.train_log.close()
        
    def validate(self, subset, num_dump_events):
        """Overrides the validate method in Engine.py.
        
        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        num_dump_events -- Number of (events, true labels, and predicted labels) to dump as a .npz file
        """
        # Print start message
        if subset == "train":
            message="Validating model on the train set"
        elif subset == "validation":
            message="Validating model on the validation set"
        elif subset == "test":
            message="Validating model on the test set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None

        print(message)
        
        # Setup the CSV file for logging the output, path to save the actual and reconstructed events, dataloader iterator
        if subset == "train":
            self.log=CSVData(self.dirpath + "train_validation_log.csv")
            np_event_path=self.dirpath + "/train_valid_iteration_"
            data_iter=self.train_loader
            dump_iterations=max(1, ceil(num_dump_events / self.config.batch_size_train))
        elif subset == "validation":
            self.log=CSVData(self.dirpath + "valid_validation_log.csv")
            np_event_path=self.dirpath + "/val_valid_iteration_"
            data_iter=self.val_loader
            dump_iterations=max(1, ceil(num_dump_events / self.config.batch_size_val))
        else:
            self.log=CSVData(self.dirpath + "test_validation_log.csv")
            np_event_path=self.dirpath + "/test_validation_iteration_"
            data_iter=self.test_loader
            dump_iterations=max(1, ceil(num_dump_events / self.config.batch_size_test))
        
        print("Dump iterations = {0}".format(dump_iterations))
        
        save_arr_dict={"events": [], "labels": [], "energies": []}
        
        return None
    
    def sample(self):
        raise NotImplementedError

    def interpolate(self):
        raise NotImpelementedError
        
        
        
    
                        
                        
                
                
                