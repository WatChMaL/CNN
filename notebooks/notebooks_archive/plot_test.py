# Imports
import math
import os
import sys
import pandas as pd
import numpy as np

# Add the path to the parent directory to augment search for module
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)
    
# Import the custom plotting module
from plot_utils import plot_utils
import random

# Label dict - Dictionary mapping integer labels to str
label_dict = {0:"gamma", 1:"e", 2:"mu"}

# Using the absolute path
run_id = "20190619_041500"
model_name = "ConvaeNet"
dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"

# Setup the path to the training log file
training_log = dump_dir + "log_train.csv"

# Plot the training progress of the current model
"""plot_utils.plot_vae_training([training_log], [model_name], {model_name:["red", "blue"]},
                             show_plot=True, save_path="1.eps")"""


# Plot using the downsample intervals
plot_utils.plot_vae_training([training_log], [model_name], {model_name:["red", "blue"]},
                             downsample_interval=32, show_plot=True,
                             save_path="2.eps")

print("Good")