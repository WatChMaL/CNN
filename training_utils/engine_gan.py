"""
engine_gan.py

Derived engine class for training a GAN
"""

# +
import pdb
import os.path
from os import path

# Python standard imports
from sys import stdout
from math import floor, ceil
from time import strftime, localtime
import numpy as np
import random
import pdb
# -

# Numerical imports
from numpy import savez

# PyTorch imports
from torch import cat, Tensor, from_numpy
from torch import argmax
from torch.nn import Softmax, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchviz import make_dot

# WatChMaL imports
from training_utils.engine import Engine
from training_utils.loss_funcs import CELoss
from plot_utils.notebook_utils import CSVData

# Global variables
_SOFTMAX   = Softmax(dim=1)
_LOG_KEYS  = ["g_loss", "d_loss"]
_DUMP_KEYS = ["g_loss", "d_loss", "gen_imgs"]

class EngineGAN(Engine):
    
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Factor used to scale real images into range of generator output
        self.scale = 3146.8333
        
        # Loss function
        self.criterion = BCEWithLogitsLoss()
            
        # Optimizers
        self.optimizer_G = Adam(self.model_accs.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizer_D = Adam(self.model_accs.parameters(), lr=config.lr, betas=(0.5, 0.999))


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
                                           pin_memory=False, sampler=SubsetRandomSampler(self.train_indices), num_workers=10)
        self.val_loader = DataLoader(self.val_dset, batch_size=self.config.batch_size_val, shuffle=False,
                                           pin_memory=False, sampler=SubsetRandomSampler(self.val_indices), num_workers=10)
        self.test_loader = DataLoader(self.test_dset, batch_size=self.config.batch_size_test, shuffle=False,
                                           pin_memory=False, sampler=SequentialSampler(self.test_indices), num_workers=10)

        # Define the placeholder attributes
        self.data     = None
        self.labels   = None
        self.energies = None
        
        self.g_loss = None
        self.d_loss = None
        self.dreal_loss = None
        self.dfake_loss = None
        
        # Adversarial ground truths
        self.valid = None
        self.fake = None
        
        
    def forward(self, mode):
        """Overrides the forward abstract method in Engine.py.
        
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """
       
        if self.data is not None and len(self.data.size()) == 4:
            self.data = self.data.to(self.device)
            # Put data into same range as output
            self.data = (self.data/self.scale) - 0.5
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
        
        
        valid = Tensor(self.data.shape[0], 1).fill_(1.0).to(self.device)
        fake = Tensor(self.data.shape[0], 1).fill_(0.0).to(self.device)
        
        # Discriminator gets updated multiple times for every time the generator does
        for i in np.arange(9):
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (self.data.shape[0], 128, 1, 1))).to(self.device)
            #self.model.zero_grad()
            model_results = self.model(self.data, z)
            gen_results = model_results['genresults'].detach()
            real_results = model_results['realresults'].detach()

            # ---------------------
            #  Discriminator Loss 1 
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            dreal_loss = self.criterion(real_results, valid)
            self.dreal_loss = dreal_loss

            dfake_loss = self.criterion(gen_results, fake)
            self.dfake_loss = dfake_loss

            d_loss = dreal_loss + dfake_loss
            #self.d_loss = d_loss

            if self.dreal_loss.requires_grad is False:
                self.dreal_loss.requires_grad = True

            if self.dfake_loss.requires_grad is False:
                self.dfake_loss.requires_grad = True

            # Discriminator gets updated
            self.optimizer_D.zero_grad()  # Reset gradient accumulation   
            self.dreal_loss.contiguous()
            self.dreal_loss.backward()    # Propagate the loss backwards
            self.dfake_loss.contiguous()
            self.dfake_loss.backward()    # Propagate the loss backwards
            self.optimizer_D.step()       # Update the optimizer parameters

            del z, gen_results, real_results, model_results
        
        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (self.data.shape[0], 128, 1, 1))).to(self.device)
        
        #self.model.zero_grad()
        
        model_results = self.model(self.data, z)
        
        # -----------------
        #  Generator Loss
        # -----------------
        
        gen_results = model_results['genresults'].detach()
        #gen_numpy = gen_results.cpu().detach().numpy()
        #gen_numpy = gen_numpy + abs(min(gen_numpy))
        #gen_range = max(gen_numpy) - min(gen_numpy)
        #gen_numpy = (gen_numpy - min(gen_numpy)) / gen_range
        #gen_results = from_numpy(gen_numpy).to(self.device)
        
        real_results = model_results['realresults'].detach()
        #real_numpy = real_results.cpu().detach().numpy()
        #real_numpy = real_numpy + abs(min(real_numpy))
        #real_range = max(real_numpy) - min(real_numpy)
        #real_numpy = (real_numpy - min(real_numpy)) / real_range
        #real_results = from_numpy(real_numpy).to(self.device)

        g_loss = self.criterion(gen_results, valid)
        self.g_loss = g_loss

        # ---------------------
        #  Discriminator Loss 2
        # ---------------------
        
        # Measure discriminator's ability to classify real from generated samples
        dreal_loss = self.criterion(real_results, valid)
        self.dreal_loss = dreal_loss
        
        dfake_loss = self.criterion(gen_results, fake)
        self.dfake_loss = dfake_loss
        
        d_loss = dreal_loss + dfake_loss
        #self.d_loss = d_loss
        
        if self.g_loss.requires_grad is False:
            self.g_loss.requires_grad = True
        
        if self.dreal_loss.requires_grad is False:
            self.dreal_loss.requires_grad = True
            
        if self.dfake_loss.requires_grad is False:
            self.dfake_loss.requires_grad = True
        
        D_x = real_results.mean().item()
        D_G_z = gen_results.mean().item()
        
        if mode == "validation":
            genimgs = (model_results['genimgs'][:50].cpu().detach().numpy() + 0.5)*self.scale
        else:
            genimgs = None
        
        del z, fake, valid, gen_results, real_results, model_results
        
        return {"g_loss"               : g_loss.cpu().detach().item(),
                "d_loss"               : d_loss.cpu().detach().item(),
                "gen_imgs"   : genimgs,
                "D_x"        : D_x,
                "D_G_z"      : D_G_z
               }
    
    def backward(self):
        """Overrides the backward method in Engine.py."""
        
        """Backward pass using the loss computed for a mini-batch."""
        
        # Generator
        self.optimizer_G.zero_grad()  # Reset gradient accumulation
        self.g_loss.contiguous()
        self.g_loss.backward()        # Propagate the loss backwards
        self.optimizer_G.step()       # Update the optimizer parameters

        # Discriminator
        self.optimizer_D.zero_grad()  # Reset gradient accumulation   
        self.dreal_loss.contiguous()
        self.dreal_loss.backward()    # Propagate the loss backwards
        self.dfake_loss.contiguous()
        self.dfake_loss.backward()    # Propagate the loss backwards
        self.optimizer_D.step()       # Update the optimizer parameters

    
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
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to upadte when saving the best model
        best_g_loss = 1000000.
        best_d_loss = 1000000.
        
        # Initialize the iterator over the validation subset
        val_iter = iter(self.val_loader)

        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            # Local training loop for a single epoch
            for data in self.train_loader:

                # Using only the charge data
                self.data     = data[0][:,:,:,:].float()
                self.labels   = data[1].long()
                self.energies = data[2]
                
                # Do a forward pass using data = self.data
                res = self.forward(mode="train")

                # Do a backward pass using loss = self.loss
                self.backward()
                
                # Update the epoch and iteration
                epoch     += 1./len(self.train_loader)
                iteration += 1

                # Iterate over the _LOG_KEYS and add the values to a list
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
                    print("... Iteration %d ... Epoch %1.2f ... G Loss %1.3f ... D Loss %1.3f ... D_x %1.3f ... D_G_z %1.3f" %
                          (iteration, epoch, res["g_loss"], res["d_loss"], res["D_x"], res["D_G_z"]))

                # Save the model computation graph to a file
                """if iteration == 1:
                    graph = make_dot(res["raw_pred_labels"], params=dict(list(self.model_accs.named_parameters())))
                    graph.render(self.dirpath + "/model", view=False)
                    break"""

                # Run validation on given intervals
                if iteration%dump_iterations[0] == 0:

                    curr_g_loss = 0.
                    curr_d_loss = 0.
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

                        curr_g_loss += res["g_loss"]
                        curr_d_loss += res["d_loss"]

                    for local_value in local_values:
                        values.append(local_value/num_val_batches)

                    # Record the validation stats to the csv
                    self.val_log.record(keys, values)

                    # Average the loss over the validation batch
                    curr_g_loss = curr_g_loss / num_val_batches
                    curr_d_loss = curr_d_loss / num_val_batches

                    # Save the best model
                    if curr_g_loss < best_g_loss:
                        self.save_state(mode="best")
                        curr_g_loss = best_g_loss

                    if iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies"]
                        save_arr_values = [self.data.cpu().numpy(), self.labels.cpu().numpy(), self.energies.cpu().numpy()]
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

            print("... Iteration %d ... Epoch %1.2f ... G Loss %1.3f ... D Loss %1.3f" %
                  (iteration, epoch, res['g_loss'], res['d_loss']))

            
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
        save_arr_dict = {"events":[], "labels":[], "energies":[]}
        
        for iteration, data in enumerate(data_iter):
            
            stdout.write("Iteration : " + str(iteration) + "\n")

            # Extract the event data from the input data tuple
            self.data     = data[0][:,:,:,:].float()
            self.labels   = data[1].long()
            self.energies = data[2].float()
                    
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
                        
            if iteration < dump_iterations:
                save_arr_dict["labels"].append(self.labels.cpu().numpy())
                save_arr_dict["energies"].append(self.energies.cpu().numpy())
                
                for key in _DUMP_KEYS:
                    if key in res.keys():
                        save_arr_dict[key].append(res[key])
            elif iteration == dump_iterations:
                save_arr_dict["gen_imgs"].append(res["gen_imgs"])
                print("Saving the npz dump array :")
                savez(np_event_path + "dump.npz", **save_arr_dict)
                break
        
        #if not path.exists(np_event_path + "dump.npz"):
        #    print("Saving the npz dump array :")
        #    savez(np_event_path + "dump.npz", **save_arr_dict)
