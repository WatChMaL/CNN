"""
WCH5Dataset update to apply normalization on the fly to the test dataset
"""

# PyTorch imports
from torch.utils.data import Dataset
import h5py

import numpy as np

# WatChMaL imports
import preprocessing.normalize_funcs as norm_funcs

class WCH5DatasetTest(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from the hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    def __init__(self, test_dset_path, norm_params_path, chrg_norm="identity", time_norm="identity", shuffle=1, test_subset=None, seed=42):
        
        f = h5py.File(test_dset_path, "r")
        
        hdf5_event_data = f["event_data"]
        hdf5_labels = f["labels"]
        hdf5_energies = f["energies"]
        hdf5_positions = f["positions"]
        hdf5_angles = f["angles"]
        
        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]
        
        assert hasattr(norm_funcs, chrg_norm) and hasattr(norm_funcs, time_norm), "Functions "+ chrg_norm + " and/or " + time_norm + " are not implemented in normalize_funcs.py, aborting."
        
        # Create a memory map for event_data - loads event data into memory only on __getitem__()
        self.event_data = np.memmap(test_dset_path, mode="r", shape=hdf5_event_data.shape,
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
        self.reduced_size = test_subset
        
        self.test_indices = np.arange(len(self))
        if self.reduced_size is not None:
            assert len(self.test_indices)>=self.reduced_size
            self.test_indices = np.random.choice(self.labels.shape[0], self.reduced_size)
        
        # Seed the pseudo random number generator
        if seed is not None:
            np.random.seed(seed)
            
        # Shuffle the indices
        if shuffle:
            np.random.shuffle(self.test_indices)
            
    def __getitem__(self, index):
        return np.array(self.event_data[index,:]), np.concatenate([np.squeeze(self.chrg_func(np.expand_dims(self.event_data[index, :, :, :19], axis=0), self.chrg_acc, apply=True)), np.squeeze(self.time_func(np.expand_dims(self.event_data[index, :, :, 19:], axis=0), self.time_acc, apply=True))], axis=2), self.labels[index], self.energies[index], self.angles[index], index, self.positions[index]
    
    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size