import numpy as np
import numpy.ma as ma

__all__ = ['horizontal_flip', 'vertical_flip', ]

horizontal_map_array_idxs=[0,11,10,9,8,7,6,5,4,3,2,1,12,17,16,15,14,13,18]
def horizontal_flip(data):
    return np.flip(data[horizontal_map_array_idxs,:,:,],axis=2)

vertical_map_array_idxs=[6,5,4,3,2,1,0,11,10,9,8,7,15,14,13,12,17,16,18]
def vertical_flip(data):
    return np.flip(data[vertical_map_array_idxs,:,:],axis=1)    