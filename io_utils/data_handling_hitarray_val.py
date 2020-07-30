"""
WCH5Dataset update to apply normalization on the fly and use
the pre-computed indices for the training and validation datasets
"""
import pdb

# PyTorch imports
from torch.utils.data import Dataset
import h5py

import numpy as np
import numpy.ma as ma
import math
import random

# WatChMaL imports
import preprocessing.normalize_funcs as norm_funcs


class WCH5DatasetV(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from the hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    
    Use on the traning and validation datasets
    """

    def __init__(self, trainval_dset_path, trainval_idx_path, norm_params_path, chrg_norm="identity", time_norm="identity", shuffle=1, trainval_subset=None, num_datasets = 1, seed=42, collapse_arrays=False):
        
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
        self.train_indices_original = []
        self.val_indices = []
        self.val_indices_original = []
        
        for i in np.arange(num_datasets):

            f = h5py.File(trainval_dset_path[i], "r")

            hdf5_labels = f["labels"]
            hdf5_energies = f["energies"]
            hdf5_positions = f["positions"]
            hdf5_angles = f["angles"]
            hdf5_hit_pmt = f["hit_pmt"]
            hdf5_hit_charge = f["hit_charge"]
            hdf5_hit_time = f["hit_time"]
            hdf5_event_hits_index = f["event_hits_index"]

            # Create a memory map for event_data - loads event data into memory only on __getitem__()
            self.hit_pmt.append(np.memmap(trainval_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                        offset=hdf5_hit_pmt.id.get_offset(), dtype=hdf5_hit_pmt.dtype))
            self.time.append(np.memmap(trainval_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                        offset=hdf5_hit_time.id.get_offset(), dtype=hdf5_hit_pmt.dtype))
            self.charge.append(np.memmap(trainval_dset_path[i], mode="r", shape=hdf5_hit_pmt.shape,
                                        offset=hdf5_hit_charge.id.get_offset(), dtype=hdf5_hit_pmt.dtype))

            # Load the contents which could fit easily into memory
            self.labels.append(np.array(hdf5_labels))
            self.energies.append(np.array(hdf5_energies))
            self.positions.append(np.array(hdf5_positions))
            self.angles.append(np.array(hdf5_angles))
            self.event_hits_index.append(np.append(hdf5_event_hits_index, self.hit_pmt[i].shape[0]).astype(np.int64))

            # Set the total size of the trainval dataset to use
            self.reduced_size = trainval_subset

             # Load the train and validation indices from the disk
            if trainval_idx_path[i] is not None:
                trainval_indices = np.load(trainval_idx_path[i], allow_pickle=True)
                train_indices = trainval_indices["train_idxs"]
                self.train_indices.append(train_indices[:])
                val_indices = trainval_indices["val_idxs"]
                self.val_indices.append(val_indices[:])
                print("Loading validation indices from: ", trainval_idx_path[i])
            else:
                trainval_indices = np.arange(self.labels.shape[0])
                np.random.shuffle(trainval_indices)
                n_train = int(0.8 * trainval_indices.shape[0])
                n_valid = int(0.1 * trainval_indices.shape[0])
                self.train_indices = trainval_indices[:n_train]
                self.val_indices = trainval_indices[n_train:n_train+n_valid]
                print("Validation Indices Randomized")

            np.random.shuffle(self.train_indices[i])
            np.random.shuffle(self.val_indices[i])

            # Seed the pseudo random number generator
            if seed is not None:
                np.random.seed(seed)

            # Shuffle the indices
            if shuffle:
                np.random.shuffle(self.train_indices[i])
                np.random.shuffle(self.val_indices[i])

            # If using a subset of the entire dataset, preserve the ratio b/w
            # the training and validation subsets
            if self.reduced_size is not None:
                assert (len(self.train_indices[i]) + len(self.val_indices[i]))>=self.reduced_size
                total_indices = len(self.train_indices[i]) + len(self.val_indices[i])
                n_train = int((len(self.train_indices[i])/total_indices) * self.reduced_size)
                n_val = int((len(self.val_indices[i])/total_indices) * self.reduced_size)

                self.train_indices[i] = self.train_indices[i][:n_train]

            self.train_indices_original.append(self.train_indices[i])
            self.val_indices_original.append(self.val_indices[i])
            
        self.datasets = np.array(np.arange(num_datasets))

        self.mpmt_positions = np.load("/data/WatChMaL/data/IWCD_mPMT_image_positions.npz")['mpmt_image_positions']

    def __getitem__(self, index):
        np.random.shuffle(self.datasets)
        for i in np.arange(len(self.datasets)):
            if index < self.labels[self.datasets[i]].shape[0]:
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
                label = self.labels[self.datasets[i]][index]

                if self.collapse_arrays:
                    data = np.expand_dims(np.sum(data, 0),0)
                return np.expand_dims(np.squeeze(self.chrg_func(np.expand_dims(data, axis=0), self.chrg_acc, apply=True)),0), label, self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.positions[self.datasets[i]][index]

        assert False, "empty batch"
        raise RuntimeError("empty batch")
        
    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size
