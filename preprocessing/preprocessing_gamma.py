#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import os

import h5py

# Directories for input and output files
in_dir = '/home/akajal/projects/rpp-tanaka-ab/machine_learning/data/IWCDmPMT/varyE/'
out_dir = '/home/akajal/projects/rpp-tanaka-ab/akajal/fmt_data/IWCDmPMT/varyE/'

# List of particles for the data files
particles = ['e-', 'mu-', 'gamma']
for particle in particles:
    
    in_path = in_dir + particle + '/IWCDmPMT_varyE_'  + particle + '_100-1000MeV_300k.h5'
    in_file = h5py.File(in_path, "r")
    
    out_path = out_dir + 'IWCDmPMT_varyE_' + particle +'_100-1000MeV_300k_fmt_test.h5'
    out_file = h5py.File(out_path, "x")
    
    # Set the shape of the output data
    out_energies_shape = in_file['energies'].shape
    out_positions_shape = in_file['positions'].shape
    
    # Check if the particle is gamma
    if particle == 'gamma':
        out_energies_shape = out_energies_shape[:1] + (1,)
        out_positions_shape = out_positions_shape[:1] + (1,) + out_positions_shape[2:]
  
    # Create the uncompressed datasets
    out_energies = out_file.create_dataset("energies",
                                           shape=(1000, 1),#shape=out_energies_shape,
                                           dtype=in_file['energies'].dtype)

    out_events = out_file.create_dataset("event_data",
                                         shape=(1000,16,40,38),#shape=in_file['event_data'].shape,
                                         dtype=in_file['event_data'].dtype)

    out_labels = out_file.create_dataset("labels",
                                         shape=(1000,),#shape=in_file['labels'].shape,
                                         dtype=in_file['labels'].dtype)

    out_positions = out_file.create_dataset("positions",
                                            shape=(1000,1,3),#shape=out_positions_shape,
                                            dtype=in_file['positions'].dtype)


    offset = 0
    write_block = 100 # 100 chosen arbitrarily as the write block size
    assert out_energies_shape[0] % write_block == 0
    
    #############
    print(out_energies.shape, out_events.shape, out_labels.shape, out_positions.shape)
    #############

    while(offset < 1000):#out_energies_shape[0]):
        
        out_energies[offset:offset+write_block] = np.sum(in_file['energies'][offset:offset+write_block], axis=1).reshape(-1,1)
        out_events[offset:offset+write_block] = in_file['event_data'][offset:offset+write_block]
        out_labels[offset:offset+write_block] = in_file['labels'][offset:offset+write_block]
        out_positions[offset:offset+write_block] = in_file['positions'][offset:offset+write_block][:,0].reshape(write_block, 1,-1)

        offset = offset + write_block
