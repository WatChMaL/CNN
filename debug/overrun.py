import numpy as np
import time 

n = 50000
last = time.time()
for i in range(1000):
    x = np.ones((n,n))
    t = time.time()
    print(t-last)
    print(i)
    last = t
