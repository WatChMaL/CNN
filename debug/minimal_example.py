import numpy as np 
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from test_dset import Test_Dset

from generate_h5 import generate_h5_file, generate_indices 
import argparse
from math import floor, ceil
import os,sys

import gc, psutil

def pprint_ntuple(nt):
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = psutil._common.bytes2human(value)
        print('%-10s : %7s' % (name.capitalize(), value))

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
        trainval_path    = ['/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval.h5']
    else:
        trainval_path   =     [os.path.join(args.dir,args.h5_name)]
    train_dset = Test_Dset(trainval_path[0], use_mem_map=args.use_mem_map, use_tables=args.use_tables)

    if args.no_torch:
        for epoch in range(args.epochs):
            training_idxs = np.arange(len(train_dset))
            np.random.shuffle(training_idxs)
            start_idx = 0
            end_idx = 512

            for i in range(1,ceil(len(train_dset)/512)):
                batch_start = 512*(i-1)
                batch_end = 512 * i if 512 * i < len(train_dset) else len(train_dset)
                data = train_dset[training_idxs[batch_start:batch_end]]
                print("Epoch: {} Batch: {} Object Size: {} Event_data Refs: {} Data: {} File Size: {}".format(epoch+1, 
                                                i,sys.getsizeof(data),len(gc.get_referrers(train_dset.event_data)), sys.getsizeof(data), sys.getsizeof(train_dset.f)))
                # pprint_ntuple(psutil.swap_memory())
    else:
        train_indices = [i for i in range(len(train_dset))]
        train_loader = DataLoader(train_dset, batch_size=512, shuffle=False,
                                            pin_memory=False, sampler=SubsetRandomSampler(train_indices), num_workers=args.num_workers)
        for epoch in range(args.epochs):
            for i, data in enumerate(train_loader):
                print("Epoch: {} Batch: {} Object Size: {} Event_data Refs: {} Data: {} File Size: {}".format(epoch+1, 
                                                i,sys.getsizeof(data),len(gc.get_referrers(train_dset.event_data)), sys.getsizeof(data), sys.getsizeof(train_dset.f)))
                # pprint_ntuple(psutil.swap_memory())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true',dest='generate',default=False)
    parser.add_argument('--data_dir', type=str, dest='dir',default='/fast_scratch/WatChMaL/debug')
    parser.add_argument('--h5_name', type=str, dest='h5_name',default='dummy_dataset.h5')
    parser.add_argument('--n_samples', type=int, dest='n_samples',default=10000)
    parser.add_argument('--triumf_data', action='store_true',default=False,dest='use_triumf_path')
    parser.add_argument('--epochs', type=int,dest='epochs',default=1)
    parser.add_argument('--num_workers', type=int,dest='num_workers',default=1)
    parser.add_argument('--use_memmap', action='store_true',default=False,dest='use_mem_map')
    parser.add_argument('--no_torch', action='store_true',default=False,dest='no_torch')
    parser.add_argument('--use_tables', action='store_true',default=False,dest='use_tables')

    args = parser.parse_args()
    
    run_test(args)
