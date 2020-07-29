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
import torchvision.transforms as transforms
# -

# Numerical imports
from numpy import savez

# +
# PyTorch imports
from torch import cat, Tensor, from_numpy, randn, manual_seed, full, FloatTensor
from torch import argmax
from torch.nn import Softmax, BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
#from torchviz import make_dot
from random import seed

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
seed(manualSeed)
manual_seed(manualSeed)
# -

# WatChMaL imports
from training_utils.engine import Engine
from training_utils.loss_funcs import CELoss
from plot_utils.notebook_utils import CSVData

# Global variables
_SOFTMAX   = Softmax(dim=1)
_LOG_KEYS  = ["g_loss", "d_loss"]
_DUMP_KEYS = ["g_loss", "d_loss"]

class EngineGAN(Engine):
    
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Loss function
        self.criterion = BCELoss()

        # Optimizers
        self.optimizerG = Adam(self.model.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.optimizerD = Adam(self.model.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))        
        
        # Initialize the torch dataloaders

        self.train_loader = DataLoader(self.train_dset, batch_size=self.config.batch_size_train,
                                         shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dset, batch_size=self.config.batch_size_val,
                                         shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.train_dset, batch_size=self.config.batch_size_test,
                                         shuffle=True, num_workers=2)
        

        # Define the placeholder attributes
        self.data     = None
        self.labels   = None
        self.energies = None
        
        self.g_loss = None
        self.d_loss = None
        self.dreal_loss = None
        self.dfake_loss = None
        
        
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.nz = 128
        self.fixed_noise = randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0
        
        
    def forward(self, mode):
        """Overrides the forward abstract method in Engine.py.
        
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """

        # if self.data is not None and len(self.data.size()) == 4:
            #self.data = self.data.permute(0,3,1,2)
        self.data = self.data.to(self.device)
        # Set the correct grad_mode given the mode
        if mode == "train":
            grad_mode = True
            self.model.train()
        elif mode == "validation":
            grad_mode= False
            self.model.eval()
        
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.model.discriminator.zero_grad()
        # Format batch
        b_size = self.data.size(0)
        label = full((b_size,), self.real_label, device=self.device)
        # Forward pass real batch through D
        output = self.model.discriminator(self.data).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = randn(b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.model.generator(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.model.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        self.model.generator.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.model.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()
        

        if mode == "validation":
            genimgs = self.model.generator(self.fixed_noise).cpu().detach().numpy()
        else:
            genimgs = None
        
        #del fake, output, label, noise, errD_real, errD_fake
        
        return {"g_loss"               : errG.cpu().detach().item(),
                "d_loss"               : errD.cpu().detach().item(),
                "gen_imgs"   : genimgs,
                "D_x"        : D_x,
                "D_G_z1"     : D_G_z1,
                "D_G_z2"     : D_G_z2
               }
    
    def backward(self, iteration, epoch):
        """Overrides the backward method in Engine.py."""
        
        """Backward pass using the loss computed for a mini-batch."""
        
        # For the GAN, the backward pass is taken care of in the forward function
    
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

        os.mkdir(os.path.join(self.dirpath, 'imgs'))

        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            # Local training loop for a single epoch
            for data in self.train_loader:

                # Using only the charge data
                self.data     = data[0]
                
                # Do a forward pass using data = self.data
                res = self.forward(mode="train")
                
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
                    print("... Iteration %d ... Epoch %1.2f ... G Loss %1.3f ... D Loss %1.3f ... D_x %1.3f ... D_G_z1 %1.3f ... D_G_z2 %1.3f" %
                          (iteration, epoch, res["g_loss"], res["d_loss"], res["D_x"], res["D_G_z1"], res["D_G_z2"]))

                #Save example images
                if iteration % 500 ==0:
                    res = self.forward(mode='validation')
                    save_arr_keys=["gen_imgs"]
                    save_arr_values=[res["gen_imgs"]]
                    # Save the actual and reconstructed event to the disk
                    savez(os.path.join(self.dirpath, 'imgs') + "/iteration_" + str(iteration) + ".npz",
                        **{key:value for key,value in zip(save_arr_keys,save_arr_values)})

                # # Save the latest model   
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
            self.data     = data[0]
            #self.labels   = data[1].long()
            #self.energies = data[2].float()
                    
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
                save_arr_dict["gen_imgs"] = []
                for key in _DUMP_KEYS:
                    if key in res.keys():
                        save_arr_dict[key] = []
                        
            if iteration < dump_iterations:
                #save_arr_dict["labels"].append(self.labels.cpu().numpy())
                #save_arr_dict["energies"].append(self.energies.cpu().numpy())
                
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


