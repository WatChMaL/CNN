import numpy as np
from pathlib import Path
import argparse

import h5py

'''
Merges numpy arrays into an hdf5 file
'''

GAMMA = 0 # 0 is the label for gamma events

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges numpy arrays; outputs hdf5 file")
    parser.add_argument("input_file_list",
                        type=str, nargs=1,
                        help="txt file with a list of files to merge")
    parser.add_argument('output_file', type=str, nargs=1,
                        help="where do we put the output")
    
    parser.add_argument("--encoding", type=str,default='ASCII',help="specifies encoding to be used for loading numpy arrays saved with python 2")
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    # -- Parse arguments
    config = parse_args()

    #read in the input file list
    with open(config.input_file_list[0]) as f:
        files = f.readlines()

    #remove whitespace 
    files = [x.strip() for x in files] 
     
    # -- Check that files were provided
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging "+str(len(files))+" files")
    
    print("Files are:")
    print(files)


    # -- Start merging
    i = 0
    array_list = []
    

    total_rows = 0    

    prev_shape=None

    dtype_data_prev=None
    dtype_labels_prev=None
    dtype_energies_prev=None
    dtype_positions_prev=None
    
    dtype_PATHS_prev=None
    dtype_IDX_prev=None
    
    for file_name in files:
        print("Loading " + file_name)
        print(str(i)+"/"+str(len(files)))

        #check that we have a regular file
        if not Path(file_name).is_file():
            raise ValueError(
                file_name+" is not a regular file or does not exist")
        #the encoding is set by default to read files written out using python 2
        info = np.load(file_name,encoding=config.encoding)
        x_data = info['event_data']
        labels = info['labels']
        energies = info['energies']
        positions = info['positions']
        
        PATHS = info['PATHS']
        IDX = info['IDX']
        
        i += 1
        shape = x_data.shape
        
        print("Array shape" + str(shape))

        #check the shape compatibility
        if prev_shape is not None:
            if prev_shape[1:] != shape[1:]:
                raise ValueError(
                    "previous file data and this ({}) file data"
                    " don't have events with"
                    " the same data structure."
                    " shapes are: {} {}".format(file_name, prev_shape, shape))

        prev_shape=shape
        
        labels_shape=labels.shape
        if labels_shape[0] != shape[0]:
            raise ValueError(
                "Problem in file {}."
                " Number of events ({})"
                " and labels ({}) not equal".format(file_name,
                                                    shape[0],
                                                    labels[0]))

        if dtype_data_prev is not None:
            if x_data.dtype != dtype_data_prev or labels.dtype != dtype_labels_prev:
               raise ValueError("data types mismatch at file {}".format(
                   file_name))
               
        dtype_data_prev=x_data.dtype
        dtype_labels_prev=labels.dtype
        dtype_energies_prev=energies.dtype
        dtype_positions_prev=positions.dtype
        
        dtype_PATHS_prev=PATHS.dtype
        dtype_IDX_prev=IDX.dtype
           
        total_rows += shape[0]
        
            
        del x_data
        del labels
        del energies
        del positions
        del PATHS
        del IDX
        del info

    print("We have {} total events".format(total_rows))

    print("opening the hdf5 file\n")
    f=h5py.File(config.output_file[0],'w')

    #this will create unchunked (contiguous), uncompressed datasets,
    #that can be memmaped
    dset_labels=f.create_dataset("labels",
                                 shape=(total_rows,),
                                 dtype=dtype_labels_prev)
    
    dset_PATHS=f.create_dataset("PATHS",
                                shape=(total_rows,),
                                dtype=dtype_PATHS_prev)
    dset_IDX=f.create_dataset("IDX",
                              shape=(total_rows,),
                              dtype=dtype_IDX_prev)
    
    dset_event_data=f.create_dataset("event_data",
                                     shape=(total_rows,)+prev_shape[1:],
                                     dtype=dtype_data_prev)
    dset_energies=f.create_dataset("energies",
                                   shape=(total_rows, 1),
                                   dtype=dtype_energies_prev)
    dset_positions=f.create_dataset("positions",
                                    shape=(total_rows, 1, 3),
                                    dtype=dtype_positions_prev)

    
    i = 0
    j = 0
    print("Filling hdf5 datasets")

    offset=0
    
    for file_name in files:

        print("Loading " + file_name)
        print(str(i)+"/"+str(len(files)))
        
        info = np.load(file_name,encoding=config.encoding)
        x_data = info['event_data']
        labels = info['labels']
        
        energies = info['energies']
        positions = info['positions']
        
        # Process gamma events (adapted from preprocessing_gamma.py by Abhishek Kajal)
        for i, lab in enumerate(labels):
            if lab == GAMMA:
                energies[i] = np.sum(energies[i], axis=1).reshape(-1,1)
                positions[i] = positions[i].reshape(1, 1,-1)
        
        PATHS = info['PATHS']
        IDX = info['IDX']
        
        i += 1
        
        offset_next=offset+shape[0]
        
        print("Array shape" + str(shape))
        
        dset_event_data[offset:offset_next,:]=x_data
        dset_labels[offset:offset_next]=labels
        dset_energies[offset:offset_next,:]=energies
        dset_positions[offset:offset_next,:,:]=positions
        
        dset_PATHS[offset:offset_next]=PATHS
        dset_IDX[offset:offset_next]=IDX
        
        offset=offset_next
        del x_data
        del labels
        del energies
        del positions
        del PATHS
        del IDX
        del info


    # -- Save merged arrays
    print("Saving arrays")
    f.close()
    print("Done saving")

    # -- Finish
    print("Merging complete")
