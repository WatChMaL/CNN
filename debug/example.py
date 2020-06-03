import os
import sys 
import time, tqdm
import numpy as np
import gc

@profile
def use_memory():
    a = []
    a.append(np.random.rand(1000,1000))
    for i in tqdm.tqdm(range(15)):
        a.append(np.random.rand(1000,1000))
        time.sleep(1)
    print(len(gc.get_objects()))
if __name__ == '__main__':
    use_memory()
