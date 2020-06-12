import numpy as np 
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_handling_train import WCH5DatasetT
from test_dset import Test_Dset

from generate_h5 import generate_h5_file, generate_indices 
import argparse
from math import floor
import os,sys

norm_params_path =  '/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_norm_params/IWCDmPMT_4pi_fulltank_9M_trainval_norm_params.npz'
chrg_norm =         'identity'
time_norm =         'identity'
shuffle =           1
num_datasets =      1
trainval_subset =   None
label_map =         None
batch_size_train =  512
cl_ratio =          1

def run_test(args):

    if args.generate:
        if args.dir==None:
            print('Please specify the desired data directory using --data_dir')
            return
        generate_h5_file(args.n_samples, args.dir)
        generate_indices(floor(args.n_samples*0.9),
                         floor(args.n_samples*0.1),
                         args.n_samples, args.dir)

    if args.use_triumf_path:
        trainval_idxs    = ['/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval_idxs.npz']
        trainval_path    = ['/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval.h5']
    else:
        trainval_path   =     [os.path.join(args.dir,args.h5_name)]
        trainval_idxs   =     [os.path.join(args.dir,args.idxs_name)]

    train_dset = Test_Dset(trainval_path[0], use_mem_map=args.use_mem_map)

    # train_dset = WCH5DatasetT(trainval_path, trainval_idxs, norm_params_path, chrg_norm, time_norm,
                                        #  shuffle=shuffle, num_datasets=num_datasets, trainval_subset=trainval_subset)

    # for i in np.arange(num_datasets):
    #     # Split the dataset into labelled and unlabelled subsets
    #     # Note : Only the labelled subset will be used for classifier training
    #     n_cl_train = int(len(train_dset.train_indices[i]) * cl_ratio)
    #     if i == 0:
    #         train_indices = np.array(train_dset.train_indices[i][:n_cl_train])
    #     else:
    #         train_indices = np.concatenate((train_indices,train_dset.train_indices[i]),axis=0)
    train_indices = [i for i in range(len(train_dset))]
    train_loader = DataLoader(train_dset, batch_size=batch_size_train, shuffle=False,
                                        pin_memory=False, sampler=SubsetRandomSampler(train_indices), num_workers=args.num_workers)
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            print("Epoch: {} Batch: {} Object Size: {}".format(epoch+1,i,sys.getsizeof(data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true',dest='generate',default=False)
    parser.add_argument('--data_dir', type=str, dest='dir',default='/fast_scratch/WatChMaL/debug')
    parser.add_argument('--h5_name', type=str, dest='h5_name',default='dummy_dataset.h5')
    parser.add_argument('--idxs_name', type=str, dest='idxs_name',default='dummy_trainval_idxs.npz')
    parser.add_argument('--n_samples', type=int, dest='n_samples',default=10000)
    parser.add_argument('--triumf_data', action='store_true',default=False,dest='use_triumf_path')
    parser.add_argument('--epochs', type=int,dest='epochs',default=1)
    parser.add_argument('--num_workers', type=int,dest='num_workers',default=1)
    parser.add_argument('--use_memmap', action='store_true',default=False,dest='use_mem_map')

    args = parser.parse_args()
    
    run_test(args)
