import h5py, tables
import numpy as np
import os, sys
import time

def opener(path, flags):
    flag_list = [flags, os.O_DIRECT]
    return os.open(path, flags)

class Test_Dset:
    def __init__(self, tv_path, use_mem_map=False, use_tables=False, reopen_mem_map=False, driver=None):    
        self.reopen_mem_map = reopen_mem_map and use_mem_map 

        if use_tables:
            self.f = tables.open_file(tv_path, "r", driver="H5FD_SEC2")    
            self.hdf5_event_data = self.f.get_node("/event_data")
            self.n = self.hdf5_event_data.shape[0]   
        else:
            self.fd = open(tv_path, 'rb', opener=opener)
            self.f = h5py.File(self.fd, 'r', driver=driver,rdcc_nbytes=0)
            self.hdf5_event_data = self.f["event_data"]            
            self.n = self.hdf5_event_data.shape[0]    


        self.path=tv_path
        self.offset=self.hdf5_event_data.id.get_offset()
        self.dtype=self.hdf5_event_data.dtype
        self.shape=self.hdf5_event_data.shape

        if use_mem_map:
            self.event_data = np.memmap(tv_path, mode="r", shape=self.hdf5_event_data.shape,
                                            offset=self.hdf5_event_data.id.get_offset(), dtype=self.hdf5_event_data.dtype)
            print(f"Numpy memmap stride is {self.event_data.strides}")
        else: 
            self.event_data = self.hdf5_event_data 
        self.e = np.ones((1,40,40,19))

    def __getitem__(self, index):        
        if not isinstance(index, int): index = np.sort(index)

        if self.reopen_mem_map:
            self.event_data = np.memmap(self.path, mode="r", shape=self.shape,
                                            offset=self.offset, dtype=self.dtype)
        # self.e = np.array(self.event_data[index,:,:,:19])
        self.event_data.read_direct(self.e,source_sel=np.s_[index,:,:,:19],dest_sel=np.s_[:])

        # print(sys.getsizeof(self.e))
        # os.posix_fadvise(self.fd.fileno(), 0, self.f.id.get_filesize(), os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(self.fd.fileno(), self.hdf5_event_data.id.get_offset(), self.hdf5_event_data.id.get_storage_size(), os.POSIX_FADV_DONTNEED)
        # os.posix_fadvise(self.fd.fileno(), self.hdf5_event_data.id.get_offset()+index*486400, 486400, os.POSIX_FADV_DONTNEED)
        return self.e

    def __len__(self):
        return self.n

if __name__ == "__main__":
    dset = Test_Dset('/fast_scratch/WatChMaL/debug/dummy_dataset.h5')
    print(type(dset[12]))