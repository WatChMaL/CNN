# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
import os
import random
import h5py
import numpy as np

# Add the path to the parent directory to augment search for module
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

if par_dir not in sys.path:
    sys.path.append(par_dir)
trainval_path = '/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN'

print("PID: {}".format(os.getpid()))

print('Importing events from h5 file')
#Import the h5 data
original_data_path = os.path.join(trainval_path,"IWCDmPMT_4pi_fulltank_9M_trainval.h5")
f = h5py.File(original_data_path, "r")

#get the data - cannot load event_data directly as it is too large
hdf5_event_data = (f["event_data"])
# original_eventdata = np.memmap(original_data_path, mode="r", shape=hdf5_event_data.shape,
#                                     offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype)
original_eventids = np.array(f['event_ids'])
original_energies = np.array(f['energies'])
original_positions = np.array(f['positions'])
original_angles = np.array(f['angles'])
original_labels = np.array(f['labels'])

#create datasets in file for the information small enough to load all at once
print('Creating compressed datasets for small arrays')
compressed_h5 = h5py.File(os.path.join(trainval_path,'IWCDmPMT_4pi_fulltank_9M_trainval_compressed.h5'),'w')

compressed_h5.create_dataset('event_ids', data=original_eventids, compression="gzip")
compressed_h5.create_dataset('energies', data=original_energies, compression="gzip")
compressed_h5.create_dataset('positions', data=original_positions, compression="gzip")
compressed_h5.create_dataset('angles', data=original_angles, compression="gzip")
compressed_h5.create_dataset('labels', data=original_labels, compression="gzip")
compressed_h5.create_dataset('event_data', shape=(5026528, 40, 40, 38),
                                              chunks=(1,40,40,38),
                                              compression="gzip")
#load the event_data into the compressed dataset batch by batch
print('Beginning batch loading of eventdata dataset')
event_data = compressed_h5['event_data']
eof = False
first_idx = 0
last_idx = 5000
eof_index = hdf5_event_data.shape[0]

while not eof:
    minibatch = hdf5_event_data[first_idx:last_idx]
    event_data[first_idx:last_idx] = minibatch
    print("{}/{}".format(last_idx,eof_index))
    if last_idx == eof_index: eof = True
        
    first_idx = last_idx
    if last_idx + 5000 > eof_index:
        last_idx = eof_index
    else:
        last_idx = last_idx + 5000
    sys.stdout.flush()
compressed_h5.close()
