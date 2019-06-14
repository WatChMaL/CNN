"""
Set of tools for transforming 19-layer mPMT data into a pixel grid array for matplotlib

Author: Julian Ding
"""

import numpy as np

# 10x10 square represents one mPMT
# List of top-left pixel positions (row,col) for 2x2 grids representing PMTs 0 to 18
POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,1), #5
           (0,4), #6
           (1,6), #7
           (2,8), #8
           (4,8), #9
           (6,8), #10
           (7,6), #11
           # Inner ring
           (6,4), #12
           (5,2), #13
           (3,2), #14
           (2,4), #15
           (3,6), #16
           (5,6), #17
           (4,4)] #18

# Function to get a 2D list (i.e. a list of lists, NOT a 2D numpy array)
# of mPMT subplots (numpy arrays) representing a single event
def get_mpmt_grid(data):
    assert(len(data.shape) == 3 and data.shape[2] == 19)
    rows = data.shape[0]
    cols = data.shape[1]
    grid = []
    for row in range(rows):
        subgrid = []
        for col in range(cols):
            pmts = data[row, col]
            mpmt = make_mpmt(pmts)
            subgrid.append(mpmt)
        grid.append(subgrid)
    return grid

# Function to get a 2D array of pixels representing a single event
def plot_single_image(data, padding=1):
    assert(len(data.shape) == 3 and data.shape[2] == 19)
    rows = data.shape[0]
    cols = data.shape[1]
    # Make empty output pixel grid
    output = np.zeroes(((10+padding)*rows-padding, (10+padding*cols)-padding))
    i, j = 0, 0
    for row in range(rows):
        for col in range(cols):
            pmts = data[row, col]
            output = tile(output, (i, j), pmts)
            j += 10+padding
        i += 10+padding
        j = 0
    return output

# Helper function to generate a 10x10 array representing an mPMT module
def make_mpmt(pmt_array):
    mpmt = np.zeros((10, 10))
    for i, val in enumerate(pmt_array):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val
    return mpmt
            
# Helper function to tile a canvas with mpmt subplots
def tile(canvas, ul, pmts):
    # First, create 10x10 grid representing single mPMT
    mpmt = make_mpmt(pmts)
        
    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]