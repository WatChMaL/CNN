"""
WCH5Dataset update to apply normalization on the fly and use
the pre-computed indices for the training and validation datasets
"""

# PyTorch imports
from torch.utils.data import Dataset
import h5py

import numpy as np

# WatChMaL imports
import preprocessing.normalize_funcs as norm_funcs

class WCH5Dataset(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from the hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    def __init__(self, trainval_dset_path, trainval_idx_path, norm_params_path,
                 chrg_norm="identity", time_norm="identity", shuffle=1, reduced_dataset_size=None, seed=42):
        
        f = h5py.File(trainval_dset_path, "r")
        
        hdf5_event_data = f["event_data"]
        hdf5_labels = f["labels"]
        hdf5_energies = f["energies"]
        hdf5_positions = f["positions"]
        hdf5_angles = f["angles"]
        
        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]
        
        assert hasattr(norm_funcs, chrg_norm) and hasattr(norm_funcs, time_norm), "Functions "+ chrg_norm + " and/or " + time_norm + " are not implemented in normalize_funcs.py, aborting."
        
        # Create a memory map for event_data - loads event data into memory only on __getitem__()
        self.event_data = np.memmap(trainval_dset_path, mode="r", shape=hdf5_event_data.shape,
                                    offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype)
        
        # Load the contents which could fit easily into memory
        self.labels = np.array(hdf5_labels)
        self.energies = np.array(hdf5_energies)
        self.positions = np.array(hdf5_positions)
        self.angles = np.array(hdf5_angles)
        
        # Load the normalization parameters used by normalize_hdf5 methods
        norm_params = np.load(norm_params_path, allow_pickle=True)
        self.chrg_acc = norm_params["c_acc"]
        self.time_acc = norm_params["t_acc"]
        
        self.chrg_func = getattr(norm_funcs, chrg_norm)
        self.time_func = getattr(norm_funcs, time_norm)
        
        # Set the total size of the trainval dataset to use
        self.reduced_size = reduced_dataset_size
        
        # Load the train and validation indices from the disk
        trainval_indices = np.load(trainval_idx_path, allow_pickle=True)
        self.train_indices = trainval_indices["train_idxs"]
        self.val_indices = trainval_indices["val_idxs"]
        
        # Seed the pseudo random number generator
        if seed is not None:
            np.random.seed(seed)
            
        # Shuffle the indices
        if shuffle:
            np.random.shuffle(self.train_indices)
            np.random.shuffle(self.val_indices)
            
        # If using a subset of the entire dataset, preserve the ratio b/w
        # the training and validation subsets
        if reduced_dataset_size is not None:
            total_indices = len(self.train_indices) + len(self.val_indices)
            n_train = int((len(self.train_indices)/total_indices) * reduced_dataset_size)
            n_val = int((len(self.val_indices)/total_indices) * reduced_dataset_size)
            
            self.train_indices = self.train_indices[:n_train]
            self.val_indices = self.val_indices[:n_val]
            
    def __getitem__(self, index):
        return np.array(self.event_data[index,:]), np.concatenate([np.squeeze(self.chrg_func(np.expand_dims(self.event_data[index, :, :, :19], axis=0), self.chrg_acc, apply=True)), np.squeeze(self.time_func(np.expand_dims(self.event_data[index, :, :, 19:], axis=0), self.time_acc, apply=True))], axis=2), self.labels[index], self.energies[index], self.angles[index], index, self.positions[index]