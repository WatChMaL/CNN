# Imports
import math
import os
import sys
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Add the path to the parent directory to augment search for module
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)
    
# Import the custom plotting module
from plot_utils import plot_utils
import random
import torch

# Label dict - Dictionary mapping integer labels to str
label_dict = {0:"gamma", 1:"e", 2:"mu"}

np.set_printoptions(threshold=np.inf)

def plot_event(run_id, iteration, mode):
    
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    
    if mode is "validation":
        np_arr_path = dump_dir + "val_iteration_" + str(iteration) + ".npz"
    else:
        np_arr_path = dump_dir + "iteration_" + str(iteration) + ".npz"
    
    # Load the numpy array
    np_arr = np.load(np_arr_path)
    np_event, np_recon, np_labels, np_energies, np_predicted_labels, np_predicted_energies = np_arr["events"], np_arr["recon"], np_arr["labels"], np_arr["energies"], np_arr["predicted_labels"], np_arr["predicted_energies"]

    i = random.randint(0, np_labels.shape[0]-1)
    plot_utils.plot_actual_vs_recon(np_event[i], np_recon[i], 
                                    label_dict[np_labels[i]], np_energies[i].item(),
                                    label_dict[np_predicted_labels[i]], np_predicted_energies[i].item(),
                                    show_plot=True)

    plot_utils.plot_charge_hist(torch.tensor(np_event).permute(0,2,3,1).numpy(),
                                np_recon, iteration, num_bins=200)
    
def plot_old_events(run_id, iteration, mode):
    
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    
    if mode is "validation":
        np_arr_path = dump_dir + "val_iteration_" + str(iteration) + ".npz"
    else:
        np_arr_path = dump_dir + "iteration_" + str(iteration) + ".npz"
    
    # Load the numpy array
    np_arr = np.load(np_arr_path)
    np_event, np_recon, np_labels, np_energies = np_arr["events"], np_arr["recon"], np_arr["labels"], np_arr["energies"]

    i = random.randint(0, np_labels.shape[0]-1)
    plot_utils.plot_actual_vs_recon(np_event[i], np_recon[i], 
                                    label_dict[np_labels[i]], np_energies[i].item(),
                                    label_dict[np_labels[i]], np_energies[i].item(),
                                    show_plot=True)

    plot_utils.plot_charge_hist(torch.tensor(np_event).permute(0,2,3,1).numpy(),
                                np_recon, iteration, num_bins=200)
    
def plot_log(run_id, model_name, iteration, variant, mode):

    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"

    # Setup the path to the training log file
    if mode is "training":
        log = dump_dir + "log_train.csv"
    elif mode is "training_validation":
        log = dump_dir + "log_val.csv"
    elif mode is "validation":
        log = dump_dir + "valid_validation_log.csv"
    elif mode is "validation_training":
        log = dump_dir + "train_validation_log.csv"
    else:
        print("mode has to be one of training, training_validation, validation, validation_training")
        return None
    
    downsample_interval = 32 if mode is "training" else None

    if variant is "AE":
        plot_utils.plot_ae_training([log], [model_name], {model_name:["red"]},
                                 downsample_interval=downsample_interval, show_plot=True)
    elif variant is "VAE":
        plot_utils.plot_vae_training([log], [model_name], {model_name:["red", "blue"]},
                                 downsample_interval=downsample_interval, show_plot=True)
        
    if iteration is not None:
        plot_event(run_id, iteration, mode=mode)
        
def plot_samples(run_id, model_dir, trained):
    
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    model_status = "trained" if trained is True else "untrained"
    np_arr_path = dump_dir + "samples/" + model_dir + "/" + model_status + "_samples.npy"
    
    np_arr = np.load(np_arr_path, allow_pickle=True)
    i, j = random.randint(0, np_arr.shape[0]-1), random.randint(0, np_arr.shape[0]-1)

    plot_utils.plot_actual_vs_recon(np_arr[i][0][0], np_arr[j][0][0], 
                                    label_dict[np_arr[i][1].item()], np_arr[i][2][0],
                                    show_plot=True)

    plot_utils.plot_charge_hist(np_arr[i][0][0],
                                np_arr[j][0][0], 0, num_bins=200)
    
def plot_new_samples(run_id, model_dir, trained):
    
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    model_status = "trained" if trained is True else "untrained"
    np_arr_path = dump_dir + "samples/" + model_dir + "_" + model_status + ".npz"
    
    np_arr = np.load(np_arr_path)
    np_samples, np_labels, np_energies = np_arr["samples"], np_arr["predicted_labels"], np_arr["predicted_energies"]

    i, j = random.randint(0, np_labels.shape[0]-1), random.randint(0, np_labels.shape[0]-1)
    plot_utils.plot_actual_vs_recon(np_samples[i], np_samples[j], 
                                    label_dict[np_labels[i]], np_energies[i].item(),
                                    label_dict[np_labels[j]], np_energies[j].item(),
                                    show_plot=True)
    
    plot_utils.plot_charge_hist(np_samples[i],
                                np_samples[j], 0, num_bins=200)
    
    
# Method to print out the comparison values
def print_vae_metrics(run_id):
    
    # Using the absolute path
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    train_val_log, valid_val_log = dump_dir + "train_validation_log.csv", dump_dir + "valid_validation_log.csv"
    
    # Print the average metrics on the training subset
    log_df = pd.read_csv(train_val_log)
            
    # Extract the loss values from the csv file
    loss_values = log_df["loss"].values
    mse_loss_values  = log_df["recon_loss"].values
    kl_loss_values = log_df["kl_loss"].values
    
    # Print out the average values
    print("Printing metrics over the training subset :")
    print("Average total loss : {0}".format(np.mean(loss_values)))
    print("Average mse loss : {0}".format(np.mean(mse_loss_values)))
    print("Average kl loss : {0}\n\n".format(np.mean(kl_loss_values)))
    
    # Print the average metrics on the training subset
    log_df = pd.read_csv(valid_val_log)
            
    # Extract the loss values from the csv file
    loss_values = log_df["loss"].values
    mse_loss_values  = log_df["recon_loss"].values
    kl_loss_values = log_df["kl_loss"].values
    
    # Print out the average values
    print("Printing metrics over the validation subset :")
    print("Average total loss : {0}".format(np.mean(loss_values)))
    print("Average mse loss : {0}".format(np.mean(mse_loss_values)))
    print("Average kl loss : {0}".format(np.mean(kl_loss_values)))
    
# Method to print out the comparison values for the classifier and energy regressor
def print_cl_metrics(run_id):
    
    # Using the absolute path
    dump_dir = "/home/akajal/WatChMaL/VAE/dumps/" + run_id + "/"
    train_val_log, valid_val_log = dump_dir + "train_validation_log.csv", dump_dir + "valid_validation_log.csv"
    
    # Print the average metrics on the training subset
    log_df = pd.read_csv(train_val_log)
    
    # Print out the average values
    print("Printing metrics over the training subset :")
    print("Average ce loss : {0}".format(np.mean(log_df["ce_loss"].values)))
    print("Average mse loss : {0}".format(np.mean(log_df["mse_loss"].values)))
    print("Average accuracy : {0}\n\n".format(np.mean(log_df["accuracy"].values)))
    
    # Print the average metrics on the training subset
    log_df = pd.read_csv(valid_val_log)
            
    # Print out the average values
    print("Printing metrics over the validation subset :")
    print("Average ce loss : {0}".format(np.mean(log_df["ce_loss"].values)))
    print("Average mse loss : {0}".format(np.mean(log_df["mse_loss"].values)))
    print("Average accuracy : {0}\n\n".format(np.mean(log_df["accuracy"].values)))