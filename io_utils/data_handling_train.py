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
import os

# WatChMaL imports
import preprocessing.normalize_funcs as norm_funcs

# Returns the maximum height at which cherenkov radiation will hit the tank
def find_bounds(pos, ang, label, energy):
    # Arguments:
    # pos - position of particles
    # ang - polar and azimuth angles of particle
    # label - type of particle
    # energy - particle energy
    
    '''
    #label = np.where(label==0, mass_dict[0], label)
    #label = np.where(label==1, mass_dict[1], label)
    #label = np.where(label==2, mass_dict[2], label)
    #beta = ((energy**2 - label**2)**0.5)/energy
    #max_ang = abs(np.arccos(1/(1.33*beta)))*1.5
    '''
    max_ang = abs(np.arccos(1/(1.33)))*1.05
    theta = ang[:,1]
    phi = ang[:,0]
    
    # Determine shortest distance emission will travel before it hits the tank
    # It checks the middle and edges of the emitted ring
    
    # radius of barrel
    r = 400
    
    # position of particle in barrel
    end = np.array([pos[:,0], pos[:,2]]).transpose()
    
    # Checks one edge of the ring
    # a point along the particle direction (plus max Cherenkov angle) located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta + max_ang), np.sin(theta + max_ang)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length1 = (length[:,0]**2 + length[:,1]**2)**0.5
    
    # Checks the middle of the ring
    # a point along the particle direction located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta - max_ang), np.sin(theta - max_ang)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length2 = (length[:,0]**2 + length[:,1]**2)**0.5 
    
    # Checks the other edge of the ring
    # a point along the particle direction (minus max Cherenkov angle) located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta), np.sin(theta)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length3 = (length[:,0]**2 + length[:,1]**2)**0.5 
    
    length = np.maximum(np.maximum(length1,length2), length3)

    top_ang = math.pi/2 - np.arctan((520 - pos[:,2])/ length)
    bot_ang = math.pi/2 + np.arctan(abs(-520 - pos[:,2])/length)
    lb = top_ang + max_ang
    ub = bot_ang - max_ang
    return np.array([lb, ub, np.minimum(np.minimum(length1,length2), length3)]).transpose()


class WCH5DatasetT(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from the hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    
    Use on the traning and validation datasets
    """

    def __init__(self, trainval_dset_path, trainval_idx_path, norm_params_path, chrg_norm="identity", time_norm="identity", shuffle=1, trainval_subset=None, num_datasets = 1, seed=42):
        
        assert hasattr(norm_funcs, chrg_norm) and hasattr(norm_funcs, time_norm), "Functions "+ chrg_norm + " and/or " + time_norm + " are not implemented in normalize_funcs.py, aborting."
        
        # Load the normalization parameters used by normalize_hdf5 methods
        norm_params = np.load(norm_params_path, allow_pickle=True)
        self.chrg_acc = norm_params["c_acc"]
        self.time_acc = norm_params["t_acc"]

        self.chrg_func = getattr(norm_funcs, chrg_norm)
        self.time_func = getattr(norm_funcs, time_norm)
        
        self.fds = []
        self.filesizes = []
        self.offsets = []
        self.dataset_sizes = []

        self.event_data = []
        self.labels = []
        self.energies = []
        self.positions = []
        self.angles = []
        
        self.train_indices = []
        self.val_indices = []
        
        for i in np.arange(num_datasets):

            fd = open(trainval_dset_path[i], 'rb')
            self.fds.append(fd)
            f = h5py.File(fd, "r")

            self.filesizes.append(f.id.get_filesize())

            hdf5_event_data = f["event_data"]
            hdf5_labels = f["labels"]
            hdf5_energies = f["energies"]
            hdf5_positions = f["positions"]
            hdf5_angles = f["angles"]

            assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

            # Create a memory map for event_data - loads event data into memory only on __getitem__()
            self.event_data.append(np.memmap(trainval_dset_path[i], mode="r", shape=hdf5_event_data.shape,
                                        offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype))

            # self.event_data.append(hdf5_event_data)
            self.offsets.append(hdf5_event_data.id.get_offset())
            self.dataset_sizes.append(hdf5_event_data.id.get_storage_size())

            # Load the contents which could fit easily into memory
            self.labels.append(np.array(hdf5_labels))
            self.energies.append(np.array(hdf5_energies))
            self.positions.append(np.array(hdf5_positions))
            self.angles.append(np.array(hdf5_angles))

            # Set the total size of the trainval dataset to use
            self.reduced_size = trainval_subset

             # Load the train and validation indices from the disk
            if trainval_idx_path[i] is not None:
                trainval_indices = np.load(trainval_idx_path[i], allow_pickle=True)
                train_indices = trainval_indices["train_idxs"]
                self.train_indices.append(train_indices[:])
                val_indices = trainval_indices["val_idxs"]
                self.val_indices.append(val_indices[:])
                print("Loading training indices from: ", trainval_idx_path[i])
            else:
                trainval_indices = np.arange(self.labels.shape[0])
                np.random.shuffle(trainval_indices)
                n_train = int(0.8 * trainval_indices.shape[0])
                n_valid = int(0.1 * trainval_indices.shape[0])
                self.train_indices = trainval_indices[:n_train]
                self.val_indices = trainval_indices[n_train:n_train+n_valid]
                print("Training Indices Randomized")

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

            # DATA SLICING
            # Use only data within barrel that does not touch the edges of the tank
            '''
            # For center dataset:
            # train indices
            c = ma.masked_where((find_height_center(self.angles[self.train_indices], self.labels[self.train_indices], self.energies[self.train_indices,0]) > 400) | (find_height_center(self.angles[self.train_indices], self.labels[self.train_indices], self.energies[self.train_indices,0]) < -400), self.train_indices)
            self.train_indices = c.compressed()
            # validation indices
            c = ma.masked_where((find_height_center(self.angles[self.val_indices], self.labels[self.val_indices], self.energies[self.val_indices,0]) > 400) | (find_height_center(self.angles[self.val_indices], self.labels[self.val_indices], self.energies[self.val_indices,0]) < -400), self.val_indices) 
            self.val_indices = c.compressed()
            # For dataset with varying position:
            bound = find_bounds(self.positions[:,0,:], self.angles[:,:], self.labels[:], self.energies[:,0])
            c = ma.masked_where(bound[self.train_indices,2] < 200, self.train_indices)
            c = ma.masked_where(abs(self.positions[self.train_indices,0,1]) > 250, c)
            c = ma.masked_where((self.angles[self.train_indices,0] > bound[self.train_indices,1]) | (self.angles[self.train_indices,0] < bound[self.train_indices,0]), self.train_indices)
            self.train_indices = c.compressed()
            c = ma.masked_where(bound[self.val_indices,2] < 200, self.val_indices)
            c = ma.masked_where(abs(self.positions[self.val_indices,0,1]) > 250, c)
            c = ma.masked_where((self.angles[self.val_indices,0] > bound[self.val_indices,1]) | (self.angles[self.val_indices,0] < bound[self.val_indices,0]), self.val_indices)
            self.val_indices = c.compressed()
            '''

            if self.event_data[i][0,:,:,:].shape[0] == 16:         
                self.a = None
                self.b = np.zeros((12, 40, 19), dtype=self.event_data[i].dtype)
                self.c = None
                d = np.array([[0,19],[0,20],
                                [1,17],[1,18],[1,19],[1,20],[1,21],[1,22],
                                [2,16],[2,17],[2,18],[2,19],[2,20],[2,21],[2,22], [2,23],
                                [3,15],[3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],[3,23],[3,24],
                               [4,15],[4,16],[4,17],[4,18],[4,19],[4,20],[4,21],[4,22],[4,23],[4,24],
                               [5,14],[5,15],[5,16],[5,17],[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],
                               [6,14],[6,15],[6,16],[6,17],[6,18],[6,19],[6,20],[6,21],[6,22],[6,23],[6,24],[6,25],
                               [7,15],[7,16],[7,17],[7,18],[7,19],[7,20],[7,21],[7,22],[7,23],[7,24],
                                [8,15],[8,16],[8,17],[8,18],[8,19],[8,20],[8,21],[8,22],[8,23],[8,24],
                               [9,16],[9,17],[9,18],[9,19],[9,20],[9,21],[9,22],[9,23],
                               [10,17],[10,18],[10,19],[10,20],[10,21],[10,22],
                               [11,19],[11,20]])
                self.d = np.concatenate((d,d), axis=0)
                self.d[96:,0] = self.d[96:,0] + 28
                self.e = None
                self.f = None
                self.g = None
            else:
                self.a = None
                self.c = None
            
        
        cap_ind = np.array([[0,19],[0,20],
                    [1,17],[1,18],[1,19],[1,20],[1,21],[1,22],
                    [2,16],[2,17],[2,18],[2,19],[2,20],[2,21],[2,22], [2,23],
                    [3,15],[3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],[3,23],[3,24],
                   [4,15],[4,16],[4,17],[4,18],[4,19],[4,20],[4,21],[4,22],[4,23],[4,24],
                   [5,14],[5,15],[5,16],[5,17],[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],
                   [6,14],[6,15],[6,16],[6,17],[6,18],[6,19],[6,20],[6,21],[6,22],[6,23],[6,24],[6,25],
                   [7,15],[7,16],[7,17],[7,18],[7,19],[7,20],[7,21],[7,22],[7,23],[7,24],
                    [8,15],[8,16],[8,17],[8,18],[8,19],[8,20],[8,21],[8,22],[8,23],[8,24],
                   [9,16],[9,17],[9,18],[9,19],[9,20],[9,21],[9,22],[9,23],
                   [10,17],[10,18],[10,19],[10,20],[10,21],[10,22],
                   [11,19],[11,20]]);
        self.cap_ind = np.concatenate((cap_ind,cap_ind), axis=0);
        self.cap_ind[96:,0] = self.cap_ind[96:,0] + 28;
        
        new_cap_ind_top = np.array([[11,0],[11,39],
                    [11,3],[10,4],[10,3],[10,36],[10,35],[11,36],
                    [11,4],[10,6],[9,8],[9,7],[9,33],[9,32],[10,33], [11,35],
                    [11,6],[10,7],[9,10],[8,10],[8,9],[8,30],[8,29],[9,30],[10,32],[11,33],
                   [10,9],[9,11],[8,12],[7,13],[7,12],[7,27],[7,26],[8,27],[9,29],[10,30],
                   [11,9],[10,10],[9,13],[8,13],[7,15],[6,18],[6,21],[7,24],[8,26],[9,27],[10,29],[11,30],
                   [11,10],[10,12],[9,14],[8,15],[7,16],[6,19],[6,20],[7,23],[8,24],[9,26],[10,27],[11,29],
                   [10,13],[9,16],[8,16],[7,18],[7,19],[7,20],[7,21],[8,23],[9,24],[10,26],
                    [11,13],[10,15],[9,17],[8,18],[8,19],[8,20],[8,21],[9,23],[10,24],[11,26],
                   [11,15],[10,16],[9,18],[9,19],[9,20],[9,21],[10,23],[11,24],
                   [11,16],[10,18],[10,19],[10,20],[10,21],[11,23],
                   [11,19],[11,20]]);
        new_cap_ind_bottom = np.array([[11,19],[11,20],
                    [11,16],[10,18],[10,19],[10,20],[10,21],[11,23],
                   [11,15],[10,16],[9,18],[9,19],[9,20],[9,21],[10,23],[11,24],
                    [11,13],[10,15],[9,17],[8,18],[8,19],[8,20],[8,21],[9,23],[10,24],[11,26],
                   [10,13],[9,16],[8,16],[7,18],[7,19],[7,20],[7,21],[8,23],[9,24],[10,26],
                   [11,10],[10,12],[9,14],[8,15],[7,16],[6,19],[6,20],[7,23],[8,24],[9,26],[10,27],[11,29],
                  [11,9],[10,10],[9,13],[8,13],[7,15],[6,18],[6,21],[7,24],[8,26],[9,27],[10,29],[11,30],
                   [10,9],[9,11],[8,12],[7,13],[7,12],[7,27],[7,26],[8,27],[9,29],[10,30],
                     [11,6],[10,7],[9,10],[8,10],[8,9],[8,30],[8,29],[9,30],[10,32],[11,33],
                   [11,4],[10,6],[9,8],[9,7],[9,33],[9,32],[10,33], [11,35],
                    [11,3],[10,4],[10,3],[10,36],[10,35],[11,36],
                   [11,0],[11,39]])
        self.new_cap_ind = np.concatenate((new_cap_ind_top,new_cap_ind_bottom), axis=0);
        self.new_cap_ind[96:,0] = 39 - self.new_cap_ind[96:,0];
        self.b = np.zeros((40, 40, 19), dtype=self.event_data[0].dtype);
        
        self.endcap_mPMT_order = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11],[6,0],[7,1],[8,2],[9,3],[10,4],[11,5],[12,15],[13,16],[14,17],[15,12],[16,13],[17,14],[18,18]])
            
        self.datasets = np.array(np.arange(num_datasets))

    # @profile
    def __getitem__(self, index):
        '''
        self.a = self.event_data[self.datasets[0]][index,:,:,:19]
        #self.c = self.a[:,:,self.endcap_mPMT_order[:,1]]
        #self.c[12:28,:,:] = self.a[12:28,:,:19]
        self.c = self.a
        return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(self.c,[2,0,1])), axis=0), self.chrg_acc, apply=True)), self.labels[self.datasets[0]][index], self.energies[self.datasets[0]][index], self.angles[self.datasets[0]][index], index, self.positions[self.datasets[0]][index]
        '''
        np.random.shuffle(self.datasets)
        for i in np.arange(len(self.datasets)):

            if index < self.labels[self.datasets[i]].shape[0]:
                self.a = self.event_data[self.datasets[i]][index,:,:,:19]
                if self.a.shape[0] == 16:
                    self.c = np.concatenate((self.b,self.a,self.b), axis=0)
                    self.e = np.random.rand(192,19,2)
                    prob = random.randrange(1, 7, 1)/100
                    self.f = self.e[:,:,0] > prob
                    self.g = np.where(self.f, 0, self.e[:,:,1])
                    self.c[self.d[:,0], self.d[:,1]] = self.g
                    # os.posix_fadvise(self.fds[i].fileno(), 0, self.filesizes[i], os.POSIX_FADV_DONTNEED)
                    # os.posix_fadvise(self.fds[i].fileno(), self.offsets[i], self.dataset_sizes[i], 
                    #                                      os.POSIX_FADV_DONTNEED)
                    return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(self.c,[2,0,1])),axis=0), self.chrg_acc, apply=True)), self.labels[self.datasets[i]][index], self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.positions[self.datasets[i]][index]

                else:
                    self.b[12:28,:,:] = self.a[12:28, :, :]
                    self.b[self.new_cap_ind[:,0], self.new_cap_ind[:,1],:] = self.a[self.cap_ind[:,0], self.cap_ind[:,1]]
                    self.c = self.b
                    #self.c = self.a[:,:,self.endcap_mPMT_order[:,1]]
                    #self.c[12:28,:,:] = self.a[12:28,:,:19]
                    # os.posix_fadvise(self.fds[i].fileno(), 0, self.filesizes[i], os.POSIX_FADV_DONTNEED)
                    # os.posix_fadvise(self.fds[i].fileno(), self.offsets[i], self.dataset_sizes[i], 
                    #                                      os.POSIX_FADV_DONTNEED)
                    return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(self.c,[2,0,1])), axis=0), self.chrg_acc, apply=True)), self.labels[self.datasets[i]][index], self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.positions[self.datasets[i]][index]
        assert False, "empty batch"
        raise RuntimeError("empty batch")
        
    def __len__(self):
        if self.reduced_size is None:
            return self.labels[0].shape[0]
        else:
            return self.reduced_size

if __name__ == "__main__":
    @profile
    def run_test():
        train_dset = WCH5DatasetT(trainval_path, trainval_idxs, norm_params_path, chrg_norm, time_norm, shuffle=shuffle, num_datasets=num_datasets, trainval_subset=trainval_subset)
        train_indices = [i for i in range(len(train_dset))]
        
        for epoch in range(2):
            indices_left = train_indices
            i = 0
            while len(indices_left) > 0:
                batch_idxs = indices_left[0:512 if len(indices_left) >= 512 else len(indices_left)]
                assert len(batch_idxs) == 512
                data = fetch_batch(train_dset,batch_idxs)
                indices_left = np.delete(indices_left, range(512 if len(indices_left) >= 512 else len(indices_left)))
                print("Epoch: {} Batch: {} ".format(epoch+1,i+1))
                i+=1
    @profile
    def fetch_batch(dset, batch_idxs):
        data = []
        for idx in batch_idxs:
            data.append(dset[idx])
        return data

    batch_size_train = 512
    cfg              = None
    chrg_norm        = 'identity'
    norm_params_path = '/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_norm_params/IWCDmPMT_4pi_fulltank_9M_trainval_norm_params.npz'
    shuffle          = 1
    time_norm        = 'identity'
    trainval_idxs    = ['/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval_idxs.npz']
    trainval_path    = ['/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval.h5']
    trainval_subset  = None
    num_datasets     = 1
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    run_test()    