"""
WCH5Dataset update to apply normalization on the fly to the test dataset
"""

# PyTorch imports
from torch.utils.data import Dataset
import h5py

import numpy as np
import numpy.ma as ma
import math
import random
import pdb

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

    def __init__(self, test_dset_path, test_idx_path, norm_params_path, chrg_norm="identity", time_norm="identity", shuffle=1, test_subset=None, num_datasets=1,seed=42, collapse_arrays=False):
        
        assert hasattr(norm_funcs, chrg_norm) and hasattr(norm_funcs, time_norm), "Functions "+ chrg_norm + " and/or " + time_norm + " are not implemented in normalize_funcs.py, aborting."

        self.collapse_arrays=collapse_arrays

        # Load the normalization parameters used by normalize_hdf5 methods
        norm_params = np.load(norm_params_path, allow_pickle=True)
        self.chrg_acc = norm_params["c_acc"]
        self.time_acc = norm_params["t_acc"]

        self.chrg_func = getattr(norm_funcs, chrg_norm)
        self.time_func = getattr(norm_funcs, time_norm)
        
        self.labels = []
        self.energies = []
        self.positions = []
        self.angles = []
        self.hit_pmt = []
        self.time = []
        self.charge = []
        self.event_hits_index = []
        
        self.train_indices = []
        self.val_indices = []
        
        
        self.event_data = []
        self.labels = []
        self.energies = []
        self.positions = []
        self.angles = []
        self.eventids = []
        self.rootfiles = []
        
        self.test_indices = []
        
        for i in np.arange(num_datasets):
            f = h5py.File(test_dset_path[i], "r")

            hdf5_labels = f["labels"]
            hdf5_energies = f["energies"]
            hdf5_positions = f["positions"]
            hdf5_angles = f["angles"]
            hdf5_eventids = f["event_ids"]
            hdf5_rootfiles = f["root_files"]

            hdf5_hit_pmt = f["hit_pmt"]
            hdf5_hit_charge = f["hit_charge"]
            hdf5_hit_time = f["hit_time"]
            hdf5_event_hits_index = f["event_hits_index"]

            # Create a memory map for event_data - loads event data into memory only on __getitem__()
            self.hit_pmt.append(np.memmap(test_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                            offset=hdf5_hit_pmt.id.get_offset(), dtype=hdf5_hit_pmt.dtype))
            self.time.append(np.memmap(test_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                            offset=hdf5_hit_time.id.get_offset(), dtype=hdf5_hit_pmt.dtype))
            self.charge.append(np.memmap(test_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                            offset=hdf5_hit_charge.id.get_offset(), dtype=hdf5_hit_pmt.dtype))

            # Load the contents which could fit easily into memory
            self.labels.append(np.array(hdf5_labels))
            self.energies.append(np.array(hdf5_energies))
            self.positions.append(np.array(hdf5_positions))
            self.angles.append(np.array(hdf5_angles))
            self.eventids.append(np.array(hdf5_eventids))
            self.rootfiles.append(np.array(hdf5_rootfiles))
            self.event_hits_index.append(np.append(hdf5_event_hits_index, self.hit_pmt[i].shape[0]).astype(np.int64))

            # Running only on events that went through fiTQun
            
            # Set the total size of the trainval dataset to use
            self.reduced_size = test_subset                
            
            if test_idx_path[i] is not None:
                test_indices = np.load(test_idx_path[i], allow_pickle=True)
                self.test_indices.append(test_indices["test_idxs"])
                self.test_indices[i] = self.test_indices[i][:]
                print("Loading test indices from: ", test_idx_path[i])
            
            else:
                test_indices = np.arange(self.labels[i].shape[0])
                np.random.shuffle(test_indices)
                #n_test = int(0.9 * test_indices.shape[0])
                #self.test_indices[i] = test_indices[n_test:]
                self.test_indices.append(test_indices)
                    
                
            #np.random.shuffle(self.test_indices[i])

            ## Seed the pseudo random number generator
            #if seed is not None:
                #np.random.seed(seed)

            # Shuffle the indices
            #if shuffle:
                #np.random.shuffle(self.test_indices[i])

            # If using a subset of the entire dataset
            if self.reduced_size is not None:
                assert len(self.test_indices[i])>=self.reduced_size
                self.test_indices[i] = np.random.choice(self.labels[i].shape[0], self.reduced_size)
            
        self.datasets = np.array(np.arange(num_datasets))

        self.mpmt_positions = np.load("/data/WatChMaL/data/IWCD_mPMT_image_positions.npz")['mpmt_image_positions']

    def __getitem__(self, index):
        np.random.shuffle(self.datasets)
        for i in np.arange(len(self.datasets)):
            
            if index < (self.labels[self.datasets[i]].shape[0]):
                label = self.labels[self.datasets[i]][index] 

                start = self.event_hits_index[i][index]
                stop = self.event_hits_index[i][index+1]
                hit_pmts = self.hit_pmt[i][start:stop].astype(np.int16)
                hit_mpmts = hit_pmts // 19
                hit_pmt_in_modules = hit_pmts % 19
                hit_rows = self.mpmt_positions[hit_mpmts, 0]
                hit_cols = self.mpmt_positions[hit_mpmts, 1]
                hit_charges = self.charge[i][start:stop]
                data = np.zeros((19,40,40))
                data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_charges

                if self.collapse_arrays:
                    data = np.expand_dims(np.sum(data, 0),0)
                return np.squeeze(self.chrg_func(np.expand_dims(data, axis=0), self.chrg_acc, apply=True)), label, self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.eventids[self.datasets[i]][index], self.rootfiles[self.datasets[i]][index]
                
        assert False, "empty batch"
        raise RuntimeError("empty batch")
        
        
    def __len__(self):
        if self.reduced_size is None:
            return self.labels[0].shape[0]
        else:
            return self.reduced_size


