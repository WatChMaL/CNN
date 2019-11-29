"""
Script to compute the parameters of different normalization schemes and
save them to disk to be used by the dataloaders

Uses similar reading strategy as in normalize_hdf5.py which reads in the
event data in blocks of user-defined size and accumulates values to compute
the normalization parameters

Derived from normalize_hdf5.py

Author: Abhishek .
"""

import os
import argparse
import h5py
import numpy as np
from math import ceil, sqrt

# WatChMaL imports
import normalize_funcs

# Key for data dimension that requires normalization
NORM_CAT = 'event_data'

# Exception for when functions are called with null accumulator
ACC_EXCEPTION = Exception("Attempted to apply operation with null accumulator.")

# Large number to initialize min accumulators to
LARGE = 1e10

# Default bin number for histograms
BINS = 10000

def parse_args():
    parser = argparse.ArgumentParser(description="Normalizes data from an input HDF5 file and outputs to HDF5 file.")
    parser.add_argument("--input_file", '-in', dest="input_file", type=str, nargs=1,
                        help="path to dataset to normalize")
    parser.add_argument("--output_file", "-out", dest="output_file", type=str,
                        help="Absolute path to save the .npz file with the normalization parameters")
    parser.add_argument('--block_size', '-blk', dest="block_size", type=int, default=4096,
                        help="number of events to load into memory at once", required=False)
    parser.add_argument('--chrg_norm', '-cf', dest="chrg_norm_func", type=str, nargs=1, default=["identity"], 
                        help="normalization function to apply to charge data", required=False)
    parser.add_argument('--time_norm', '-tf', dest="time_norm_func", type=str, nargs=1, default=["identity"],
                        help="normalization function to apply to time data", required=False)
    args = parser.parse_args()
    return args

def compute_normalize_params(config):
    config.input_file = config.input_file[0]
    
    # Setup path to save the normalization parameters
    block_size = int(config.block_size)
    
    # Ensure specified input file exists, then open file
    assert os.path.isfile(config.input_file), "Invalid input file path provided: " + config.input_file
    print("Reading data from", config.input_file)
    infile = h5py.File(config.input_file, 'r')
    
    c_func = getattr(normalize_funcs, config.chrg_norm_func[0])
    t_func = getattr(normalize_funcs, config.time_norm_func[0])
    print("Event data normalization scheme: charge =", config.chrg_norm_func[0], "| timing =", config.time_norm_func[0])
    
    event_data = infile[NORM_CAT]
    chunk_length = event_data.shape[0]
    num_blocks_in_file = int(ceil(chunk_length / block_size))
    
    # Accumulators for functions that need them
    c_acc, t_acc = None, None
    
    # Read and process in chunks (apply=False)
    for iblock in range(num_blocks_in_file):
        block_begin=iblock*block_size
        block_end=(iblock+1)*block_size
        if block_end>chunk_length:
            block_end=chunk_length
        # Do necessary calculations while updating accumulators
        chrg_data = event_data[block_begin:block_end,:,:,:19]
        time_data = event_data[block_begin:block_end,:,:,19:]
        
        if c_func != "identity":
            c_acc = c_func(chrg_data, acc=c_acc, apply=False)
        if t_func != "identity":
            t_acc = t_func(time_data, acc=t_acc, apply=False)
            
        if iblock != 0:
            print('\r', end='')
        print('[', iblock+1, 'of', num_blocks_in_file, 'blocks parsed ]', end='')
    print('')
    
    np.savez(config.output_file, c_acc=c_acc, t_acc=t_acc, c_func=config.chrg_norm_func[0], t_func=config.time_norm_func[0])
    
# Main
if __name__ == "__main__":
    config = parse_args()
    assert hasattr(normalize_funcs, config.chrg_norm_func[0]) and hasattr(normalize_funcs, config.time_norm_func[0]), "Functions "+config.chrg_norm_func[0]+" and/or "+config.time_norm_func[0]+" are not implemented in normalize_funcs.py, aborting."
    compute_normalize_params(config)