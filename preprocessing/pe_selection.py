"""
Script to apply a simple selection cut using the total charge produced 
at the PMT photocathodes in an event which proportional to the total number
of photoelectrons observed in an event.

Author : Abhishek .
"""

import argparse
import h5py
import numpy as np
import os
from math import ceil

# Seed the NumPy generator with 42
np.random.seed(42)

# Global constants
CUT_THRESHOLD = 300.
EVENT_DATA_KEY = "event_data"

def parse_args():
    """Argument parser method with the parser initialization and argument parsing.
    
    Returns : config object to access the individual script arguments
    """
    parser = argparse.ArgumentParser(description="Apply the simple selection cut using the total charge produced in an event.")
    parser.add_argument("--input_file", "-infile", dest="input_file", type=str,
                        help="Absolute path to the .h5 dataset")
    parser.add_argument("--output_file", "-outfile", dest="output_file", type=str,
                        help="Absolute path to the selected index .npz file")
    parser.add_argument("--block_size", "-blk", dest="block_size", type=int, default=4096,
                        help="Number of event data to load into the memory at once", required=False)
    parser.add_argument("--train_ratio", "-train", dest="train_ratio", type=float, default=0.5,
                        help="Ratio of the total dataset size to use as training data", required=False)
    parser.add_argument("--val_ratio", "-val", dest="val_ratio", type=float, default=0.1,
                        help="Ratio of the total dataset size to use as validation data", required=False)
    args = parser.parse_args()
    return args

def select_idxs_total_chrg(input_file, block_size):
    """Select the indices of the events which pass the cut
    
    Args:
        input_file -- Absolute path to the .h5 dataset
        block_size -- Number of event data to load into the memory at once
        
    Returns : np array with the selected indices
    """
    
    # Selected indices to return
    select_idxs = []
    
    print("Reading data from", input_file)
    infile = h5py.File(input_file, 'r')
    
    event_data = infile[EVENT_DATA_KEY]
    num_events = event_data.shape[0]
    
    event_idxs = np.arange(num_events)
    
    num_blocks = int(ceil(num_events / block_size))
    
    for iblock in range(num_blocks):
        block_begin = iblock*block_size
        block_end = (iblock+1)*block_size
        if block_end > num_events:
            block_end = num_events
    
        chrg_data = event_data[block_begin:block_end,:,:,:19]
        idx_data = event_idxs[block_begin:block_end]
        
        sum_chrg_data = np.sum(chrg_data.reshape(chrg_data.shape[0], -1), axis=1).reshape(-1)
        select_idxs.extend(idx_data[sum_chrg_data >= CUT_THRESHOLD])
        
        if iblock != 0:
            print('\r', end='')
        print("{0} of {1} blocks processed.".format(iblock+1, num_blocks), end='')
    print("\nFinished processing {0} events. {1} events selected.".format(num_events, len(select_idxs)))
    
    return np.array(select_idxs)

def split_and_save_idxs(idxs, train_ratio, val_ratio, output_file):
    """Split the index array into train, test and validation subsets
    
    Args:
        idxs -- np array with the selected indices
        train_ratio -- Ratio of the total dataset size to use as training data
        val_ratio -- Ratio of the total dataset size to use as validation data
        output_file -- Absolute path to location to save the index .npz file
        
    Returns : None
    """
    num_train = int(len(idxs) * train_ratio)
    num_val = int(len(idxs) * val_ratio)
    
    train_idxs = idxs[:num_train]
    val_idxs = idxs[num_train:num_val]
    test_idxs = idxs[num_train+num_val:]
    
    np.savez(output_file, **{"train_idxs":train_idxs, "val_idxs":val_idxs, "test_idxs":test_idxs})
    
if __name__ == "__main__":
        config = parse_args()
        
        # Ensure specified input file exists, then open file
        assert os.path.isfile(config.input_file), "Invalid input file path provided: " + config.input_file
    
        # Ensure no index file already exists (prevents overwriting of existing index files)
        assert not os.path.isfile(config.output_file), "Index file already exists at :" + config.output_file
        
        # Ensure the consistency in the user-defined ratios
        assert config.train_ratio + config.val_ratio < 1.0, "Sum of the argument train_ratio and val_ratio should be less than 1.0"
    
        select_idxs = select_idxs_total_chrg(config.input_file, config.block_size)
        
        # Shuffle the indices in-place
        np.random.shuffle(select_idxs)
        
        split_and_save_idxs(select_idxs, config.train_ratio, config.val_ratio, config.output_file)