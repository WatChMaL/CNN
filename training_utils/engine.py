"""
engine.py

Abstract base class for supporting engines for different types of models with
varying architectures and forward passes
"""

# +
# For debugging
import pdb

# Python standard imports
from abc import ABC, abstractmethod
from time import strftime
from os import stat, mkdir
from math import floor, ceil
import numpy as np
# -

# WatChMaL imports
from io_utils.data_handling_train import WCH5DatasetT
from io_utils.data_handling_val import WCH5DatasetV
from io_utils.data_handling_test import WCH5DatasetTest
from io_utils.ioconfig import save_config
from plot_utils.notebook_utils import CSVData

# PyTorch imports
from torch import device, load, save
from torch.nn import DataParallel
from torch.cuda import is_available

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class Engine(ABC):

    def __init__(self, model, config):
        super().__init__()

        # Engine attributes
        self.model=model
        self.config=config

        # Determine the device to be used for model training and inference
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids=["cuda:{0}".format(x) for x in config.gpu_list]
            print("Main GPU : " + self.devids[0])

            if is_available():
                self.device=device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model=DataParallel(self.model, device_ids=config.gpu_list, dim=0)
                print("CUDA is available")
            else:
                self.device=device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=device("cpu")

        # Send the model to the selected device
        self.model.to(self.device)

        # Setup the parameters tp save given the model type
        if type(self.model) == DataParallel:
            self.model_accs=self.model.module
        else:
            self.model_accs=self.model

        # Create the dataset object for the trainval and test samples
        self.train_dset = WCH5DatasetT(config.trainval_path, config.trainval_idxs, config.norm_params_path, config.chrg_norm, config.time_norm,
                                         shuffle=config.shuffle, num_datasets=config.num_datasets, trainval_subset=config.trainval_subset)
        self.val_dset = WCH5DatasetV(config.trainval_path, config.trainval_idxs, config.norm_params_path, config.chrg_norm, config.time_norm,
                                         shuffle=config.shuffle, num_datasets=config.num_datasets, trainval_subset=config.trainval_subset)
        
        self.test_dset = WCH5DatasetTest(config.test_path, config.test_idxs, config.norm_params_path, config.chrg_norm, config.time_norm,
                                       shuffle=config.shuffle, num_datasets=config.num_datasets, test_subset=config.test_subset)
        
        # Define the variant dependent attributes
        self.criterion=None

        # Define the variant independent attributes
        self.loss=None
        self.latest_savepath=None
        self.best_savepath=None

        # Create the directory for saving the log and dump files
        self.dirpath=config.dump_path + strftime("%Y%m%d_%H%M%S") + "/"
        try:
            stat(self.dirpath)
        except:
            print("Creating a directory for run dump at : {}".format(self.dirpath))
            mkdir(self.dirpath)

        # Logging attributes
        self.train_log=CSVData(self.dirpath + "log_train.csv")
        self.val_log=CSVData(self.dirpath + "log_val.csv")

        # Save a copy of the config in the dump path
        save_config(self.config, self.dirpath + "config_file.ini")
        
    @abstractmethod
    def forward(self, mode):
        """Forward pass using self.data as input."""
        raise NotImplementedError

    def backward(self):
        """Backward pass using the loss computed for a mini-batch."""
        self.optimizer.zero_grad()  # Reset gradient accumulation
        self.loss.contiguous()
        self.loss.backward()        # Propagate the loss backwards
        self.optimizer.step()       # Update the optimizer parameters

    @abstractmethod
    def train(self):
        """Training loop over the entire dataset for a given number of epochs."""
        raise NotImplementedError

    def save_state(self, mode="latest"):
        """Save the model parameters in a file.
        
        Args :
        mode -- one of "latest", "best" to differentiate
                the latest model from the model with the
                lowest loss on the validation subset (default "latest")
        """
        path=self.dirpath + str(self.config.model[1]) + "_" + mode + ".pth"

        # Update the corresponding path attribute for restoring state during runtime 
        if mode == "latest":
            self.latest_savepath=path
        elif mode == "best":
            self.best_savepath=path

        # Extract modules from the model dict and add to start_dict 
        modules=list(self.model_accs._modules.keys())
        state_dict={module: getattr(self.model_accs, module).state_dict() for module in modules}

        # Save the model parameter dict
        save(state_dict, path)

    def load_state(self, path):
        """Load the model parameters from a file.
        
        Args :
        path -- absolute path to the .pth file containing the dictionary
        with the model parameters to load from
        """
        # Open a file in read-binary mode
        with open(path, 'rb') as f:

            # Interpret the file using torch.load()
            checkpoint=load(f, map_location=self.device)

            print("Loading weights from file : {0}".format(path))

            local_module_keys=list(self.model_accs._modules.keys())
            for module in checkpoint.keys():
                if module in local_module_keys:
                    print("Loading weights for module = ", module)
                    getattr(self.model_accs, module).load_state_dict(checkpoint[module])

    def set_dump_iterations(self, train_loader):
        """Determine the intervals during training at which to dump the events and metrics.
        
        Args:
        train_loader       -- Total number of validations performed throughout training
        """

        # Determine the validation interval to use depending on the 
        # total number of iterations in the current session
        valid_interval=max(1, floor(ceil(self.config.epochs * len(train_loader)) / self.config.num_vals))

        # Save the dump at the earliest validation, middle of the training
        # and last validation near the end of training
        dump_iterations=[valid_interval, valid_interval*floor(self.config.num_vals/2),
                         valid_interval*self.config.num_vals]

        return dump_iterations
