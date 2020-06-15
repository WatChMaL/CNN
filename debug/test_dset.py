import h5py, tables
import numpy as np

class Test_Dset:
    def __init__(self, tv_path, use_mem_map=False, use_tables=False, reopen_mem_map=False):    
        self.reopen_mem_map = reopen_mem_map and use_mem_map   
        if use_tables:
            self.f = tables.open_file(tv_path, "r", driver="H5FD_SEC2")    
            hdf5_event_data = self.f.get_node("/event_data")
            self.n = hdf5_event_data.shape[0]   
        else:
            self.f = h5py.File(tv_path, 'r')
            hdf5_event_data = self.f["event_data"]
            self.n = hdf5_event_data.shape[0]    
        self.path=tv_path
        self.offset=hdf5_event_data.id.get_offset()
        self.dtype=hdf5_event_data.dtype
        self.shape=hdf5_event_data.shape

        if use_mem_map:
            self.event_data = np.memmap(tv_path, mode="r", shape=hdf5_event_data.shape,
                                            offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype)
        else: 
            self.event_data = hdf5_event_data 
        self.e = np.ones((512,40,40,19))

    def __getitem__(self, index):        
        if not isinstance(index, int): index = np.sort(index)

        if self.reopen_mem_map:
            self.event_data = np.memmap(self.path, mode="r", shape=self.shape,
                                            offset=self.offset, dtype=self.dtype)
        self.e = np.array(self.event_data[index,:,:,:19])
        return self.e

    def __len__(self):
        return self.n

if __name__ == "__main__":
    dset = Test_Dset('/fast_scratch/WatChMaL/debug/dummy_dataset.h5')
    print(type(dset[12]))