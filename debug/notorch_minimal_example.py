import numpy as np 

from generate_h5 import generate_h5_file, generate_indices 
import argparse
from math import floor, ceil
import os,sys
from test_dset import Test_Dset
import gc
import resource
import objgraph
import memory_profiler
import psutil

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

    pprint_ntuple(psutil.swap_memory())

    test_dset = Test_Dset(trainval_path[0], use_mem_map=args.use_mem_map)
    data = np.zeros((512,40,40,19))
    for epoch in range(args.epochs):
        training_idxs = np.arange(len(test_dset))
        np.random.shuffle(training_idxs)
        start_idx = 0
        end_idx = 512

        for i in range(ceil(len(test_dset)/512)):
            batch_start = 512*(i-1)
            batch_end = 512 * i if 512 * i < len(test_dset) else len(test_dset)
            data = test_dset[training_idxs[batch_start:batch_end]]
            print("Epoch: {} Batch: {} Object Size: {} Event_data Refs: {} Data: {} File Size: {}".format(epoch+1, 
                                            i,sys.getsizeof(data),len(gc.get_referrers(test_dset.event_data)), sys.getsizeof(data), sys.getsizeof(test_dset.f)))
            pprint_ntuple(psutil.swap_memory())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true',dest='generate',default=False)
    parser.add_argument('--data_dir', type=str, dest='dir',default='/fast_scratch/WatChMaL/debug')
    parser.add_argument('--h5_name', type=str, dest='h5_name',default='dummy_dataset.h5')
    parser.add_argument('--n_samples', type=int, dest='n_samples',default=10000)
    parser.add_argument('--triumf_data', action='store_true',default=False,dest='use_triumf_path')
    parser.add_argument('--epochs', type=int,dest='epochs',default=1)
    parser.add_argument('--use_memmap', action='store_true',default=False,dest='use_mem_map')

    args = parser.parse_args()
    
    run_test(args)
