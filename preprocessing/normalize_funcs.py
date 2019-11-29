"""
Module with different normalization functions and their implmentations for
both the accumulators (computing the normalization parameters) and the appliers
(apply the normalization) scheme to the input data

Derived from normalize_hdf5.py

Author : Abhishek .
""" 
import numpy as np
from math import sqrt

# =================== GLOBAL VARIABLES ====================

# Exception for when functions are called with null accumulator
ACC_EXCEPTION = Exception("Attempted to apply operation with null accumulator.")

# Large number to initialize min accumulators to
LARGE = 1e10

# Default bin number for histograms
BINS = 10000

# =================== NORMALIZATION FUNCTION CANDIDATES ====================
    
# Identity function that returns the input dataset
def identity(data, acc=None, apply=False):
    if apply:
        return data
    else:
        return acc

# Function that removes all data from a dataset by setting everything to 0
def set_zero(data, acc=None, apply=False):
    if apply:
        return np.zeros(data.shape)
    else:
        return acc

# Function that divides every entry in data array by the (non-zero) mean of the data
# acc = [current sum of events seen, number of events seen]
def divide_by_mean(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return data / (acc[0]/acc[1])
    else:
        # Calculate mean of all non-zero hits in chunk
        flat_data = data.reshape(-1,1)
        nonzero = flat_data[flat_data != 0]
        if acc is None:
            acc = [0, 0]
        acc[0] += np.sum(nonzero)
        acc[1] += nonzero.size
        return acc

# Function that divides every entry in a data array by the max of the data
def divide_by_max(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return data / acc
    else:
        if acc is None:
            acc = 0
        return max(np.amax(data), acc)
    
# Function that divides every entry in a data array by the max of the data scaled by some factor
def divide_by_max_scaled(data, scale=2, acc=None, apply=False):
    check_data(data)
    if apply:
        return 2*divide_by_max(data, acc=acc, apply=apply)
    else:
        return divide_by_max(data, acc=acc, apply=apply)

# Function that scales a dataset logarithmically: x = log(x+1)
def scale_log(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return np.log(data+1)
    else:
        return identity(data)

# Function that removes offsets in data by setting the lowest non-zero hit value as the new zero
def remove_offset_min(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            offset = data - acc
            return offset.clip(0)
    else:
        # Initialize accumulator if necessary
        if acc is None:
            acc = LARGE
        # Find minimum nonzero value and save lowest so far
        flat_data = data.reshape(-1,1)
        nonzero = flat_data[flat_data != 0]
        return min(np.amin(nonzero), acc)
    
# Function that removes offsets in data by setting the mode (peak) non-zero hit value to 0
# Accumulator: acc[0] = histogram of seen data, acc[1] = upper bound of hist range
def remove_offset_mode(data, bins=BINS, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            mode_idx = np.arange(acc[0].size)[acc[0] == np.amax(acc[0])][0]
            mode = mode_idx*acc[1]/bins
            # Subtract minimum value from every nonzero value
            return data - mode
    else:
        # Find mode non-zero hit value by binning data and selecting highest-frequency bin
        flat_data = data.reshape(-1,1)
        nonzero = flat_data[flat_data != 0]
        if acc is None:
            acc = [None, None]
        # Set upper bound on range if unset
        if acc[1] is None:
           acc[1] = 2*np.amax(nonzero)
        # Append to histogram
        if acc[0] is None:
            acc[0] = np.histogram(nonzero, bins=bins, range=(0, acc[1]))[0]
        else:
            acc[0] += np.histogram(nonzero, bins=bins, range=(0, acc[1]))[0]
        return acc

# Function that removes offsets in data by setting the mode (peak) non-zero hit value to 0 (clips hits < 0 to 0)
def remove_offset_mode_clip(data, bins=BINS, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            out = remove_offset_mode(data, bins=bins, acc=acc, apply=True)
            return out.clip(0)
    else:
        return remove_offset_mode(data, bins=bins, acc=acc, apply=False)
    
# Function that finds FWHM of data and divides every data point by FWHM (assumes data is roughly one large peak)
def divide_by_FWHM(data, scale=1, bins=BINS, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            half_max = np.amax(acc[0])/2
            edges = np.arange(acc[0].size)[acc[0] >= half_max]
            # Find left and right edges of FWHM in histogram
            llim = edges[0]*acc[1]/bins
            hlim = edges[-1]*acc[1]/bins
            fwhm = abs(hlim - llim)
            # Divide data by FWHM
            return data/(scale*fwhm)
    else:
        # Same accumulator use as remove_offset_mode
        return remove_offset_mode(data, bins=bins, acc=acc, apply=False)
    
# Function that finds the FWHM of the log plot of data and divides every point by this "log FWHM"
# (assumes data is roughly one large peak)
def divide_by_log_FWHM(data, scale=0.02, bins=BINS, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            half_max_log = np.log(np.amax(acc[0]))/2
            edges = np.arange(acc[0].size)[acc[0] >= half_max_log]
            # Find left and right edges of FWHM in histogram
            llim = edges[0]*acc[1]/bins
            hlim = edges[-1]*acc[1]/bins
            fwhm = abs(hlim - llim)
            # Divide data by FWHM
            return data/(scale*fwhm)
    else:
        # Same accumulator use as remove_offset_mode
        return remove_offset_mode(data, bins=bins, acc=acc, apply=False)

# Function that applies the transformation: f(x) = tanh(x) + 1
def tanh_plus_one(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return np.tanh(data) + 1
    else:
        return acc
    
# Function that applies the transformation: f(x) = sigmoid(x)
def sigmoid(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return 1/(1+np.exp(-data))
    else:
        return acc

# Temporary function for shifting data by an arbitrary amount
def offset_arbitrary(data, offset=950, acc=None, apply=False):
    check_data(data)
    if apply:
        return (data-offset).clip(0)
    else:
        return acc
    
# Function that applies min-max normalization: f(x) = [x-min(data)]/[max(data)-min(data)]
def min_max(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            minimum = acc[0]
            maximum = acc[1]
            return (data-minimum)/(maximum-minimum)
    else:
        if acc is None:
            acc = [LARGE, 0]
        curr_min = acc[0]
        curr_max = acc[1]
        return [min(np.amin(data), curr_min), max(np.amax(data), curr_max)]

# Function that applies Z-score normalization: f(x) = [x-mean(data)]/stdev(data)
# Implementation note: sum((xi-u)^2) = (x1^2 + ... + xi^2) + 2u(x1 + ... + xi) + Nu^2
#                   -> keep accumulator for (x1^2 + ... + xi^2) and another for (x1 + ... + xi)
#                                            ^sum of squared hits                ^direct sum of hits
def z_score(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            sum_direct = acc[0]
            sum_squared = acc[1]
            num = acc[2]
            
            mean = sum_direct/num
            stdev = sqrt((sum_squared + 2*mean*sum_direct + num*mean**2)/num)
            
            return (data-mean)/stdev
    else:
        if acc is None:
            acc = [0, 0, 0]
        acc[0] += np.sum(data)
        acc[1] += np.sum(data**2)
        acc[2] += data.size
        return acc
    
# =============== Function compositions =================
    
def offset_divide_by_mean(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return divide_by_mean(offset_arbitrary(data, apply=True), acc=acc, apply=True)
    else:
        return divide_by_mean(data, acc=acc, apply=False)

def offset_divide_by_max(data, acc=None, apply=False):
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            return divide_by_max(offset_arbitrary(data, apply=True), acc=acc, apply=True)
    else:
        return divide_by_max(data, acc=acc, apply=False)

def offset_scale_log(data, acc=None, apply=False):
    check_data(data)
    if apply:
        return scale_log(offset_arbitrary(data, apply=True), apply=True)
    else:
        return acc

# Applies the transformation: tanh((x-mode_x)/FWHM) + 1
def tanh_minus_mode_divided_by_FWHM(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            x_minus_mode = remove_offset_mode(data, acc=acc, apply=apply)
            del data
            x_minus_mode_divide_FWHM = divide_by_FWHM(x_minus_mode, acc=acc, apply=apply)
            del x_minus_mode
            return tanh_plus_one(x_minus_mode_divide_FWHM, apply=apply)
    else:
        # Accumulator for remove_offset_mode is the same as for divide_by_FWHM
        return remove_offset_mode(data, acc=acc, apply=False)

# Applies the transformation: tanh((x-mode_x)/log_FWHM) + 1
def tanh_minus_mode_divided_by_log_FWHM(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            x_minus_mode = remove_offset_mode(data, acc=acc, apply=apply)
            del data
            x_minus_mode_divide_FWHM = divide_by_log_FWHM(x_minus_mode, acc=acc, apply=apply)
            del x_minus_mode
            return tanh_plus_one(x_minus_mode_divide_FWHM, apply=apply)
    else:
        # Accumulator for remove_offset_mode is the same as for divide_by_log_FWHM
        return remove_offset_mode(data, acc=acc, apply=False)

# Applies the transformation: 2*sigmoid((x-mode_x)/FWHM)
def two_sigmoid_minus_mode_divided_by_FWHM(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            x_minus_mode = remove_offset_mode(data, acc=acc, apply=apply)
            del data
            x_minus_mode_divide_FWHM = divide_by_FWHM(x_minus_mode, acc=acc, apply=apply)
            del x_minus_mode
            return 2*sigmoid(x_minus_mode_divide_FWHM, apply=apply)
    else:
        # Accumulator for remove_offset_mode is the same as for divide_by_FWHM
        return remove_offset_mode(data, acc=acc, apply=False)

# Applies the transformation: 2*sigmoid((x-mode_x)/log_FWHM)
def two_sigmoid_minus_mode_divided_by_log_FWHM(data, acc=None, apply=False):
    check_data(data)
    if apply:
        if acc is None:
            raise ACC_EXCEPTION
        else:
            x_minus_mode = remove_offset_mode(data, acc=acc, apply=apply)
            del data
            x_minus_mode_divide_FWHM = divide_by_log_FWHM(x_minus_mode, acc=acc, apply=apply)
            del x_minus_mode
            return 2*sigmoid(x_minus_mode_divide_FWHM, apply=apply)
    else:
        # Accumulator for remove_offset_mode is the same as for divide_by_log_FWHM
        return remove_offset_mode(data, acc=acc, apply=False)

# =================== HELPER FUNCTIONS ====================
    
# Helper function to check input data shape
def check_data(data):
    assert len(data.shape) == 4 and (data.shape[1:] == (16, 40, 19) or data.shape[1:] == (40, 40, 19)), "Invalid data shape (required: n, 16, 40, 19 or n, 40, 40, 19), (provided: {0}) aborting".format(data.shape)