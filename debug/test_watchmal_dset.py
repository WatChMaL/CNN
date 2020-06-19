import numpy as np 
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from test_dset import Test_Dset

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from io_utils.data_handling_train import WCH5DatasetT

import argparse
import os,sys

import time

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

@profile
def run_test(args):
    train_dset = WCH5DatasetT(trainval_path, trainval_idxs, norm_params_path, chrg_norm, time_norm,
                                            shuffle=shuffle, num_datasets=num_datasets, trainval_subset=trainval_subset)
    train_indices = [i for i in range(len(train_dset))]
    train_loader = DataLoader(train_dset, batch_size=512, shuffle=False,
                                        pin_memory=False, sampler=SubsetRandomSampler(train_indices), num_workers=args.num_workers)
    start = time.time()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            print(time.time() - start)
            print("Epoch: {} Batch: {} ".format(epoch+1,i))
            start = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_name', type=str, dest='h5_name',default='dummy_dataset.h5')
    parser.add_argument('--triumf_data', action='store_true',default=False,dest='use_triumf_path')
    parser.add_argument('--epochs', type=int,dest='epochs',default=1)
    parser.add_argument('--num_workers', type=int,dest='num_workers',default=1)
    parser.add_argument('--use_memmap', action='store_true',default=False,dest='use_mem_map')
    parser.add_argument('--fadvise', type=str, dest='fadvise',default='file', help='Choose which part of the h5 to advise kernel to dump. Choose from file, dataset')
    args = parser.parse_args()
    
    run_test(args)