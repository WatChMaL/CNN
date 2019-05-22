"""
Source code borrowed from https://github.com/WatChMaL/UVicWorkshopPlayground/blob/master/B/notebooks/utils/utils.py
Edited by : Abhishek .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


# Set the style
plt.style.use("classic")


# Plot a confusion matrix
def plot_confusion_matrix(label,prediction,class_names, save_path=None):
    """
    Args: label ... 1D array of true label value, the length = sample size
          prediction ... 1D array of predictions, the length = sample size
          class_names ... 1D array of string label for classification targets, the length = number of categories
    """
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value  = np.max([np.max(np.unique(label)),np.max(np.unique(label))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(prediction, label,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=16)
    ax.set_yticklabels(class_names,fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()
        
# Plot the softmax curve for a given particle
def plot_softmax_curve(softmax, labels, labels_dict=None, particle=None, save_path=None, bandwidth=0.25):
    """
    Args: softmax ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels ... 1D array of true labels
          labels_dict ... Dictionary with the keys as particle types and values as col indices in the
                          softmax array
          particle ... string for the particle type e.g. "gamma", "e", "mu"
          save_path[optional] ... path to save the plot to
          bandwidth[optional] ... bandwidth to use for the kernel density estimation, default uses the
                                  silverman method to compute the bandwidth to use
    """
    assert softmax.any() != None
    assert labels.any() != None
    assert labels_dict != None
    assert particle != None
    
    # Estimate the density of the data using kernel density estimation
    density_estimates = {}
    curr_softmax = softmax[labels == labels_dict[particle],:]
    for key in labels_dict.keys():
        density_estimates[key] = gaussian_kde(curr_softmax[:, labels_dict[key]], bw_method="silverman")
        
    # Fix colors for each particle type
    color_dict = {"gamma":"r", "e":"b", "mu":"g"}
    
    x_ax = np.linspace(0, 1.0, 500)
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    
    ax.tick_params(axis="both", labelsize=20)
    
    for key in labels_dict.keys():
            ax.fill_between(x_ax, 0, density_estimates[key](x_ax), alpha=0.5, color=color_dict[key])
            ax.semilogy(x_ax, density_estimates[key](x_ax), label=key + " softmax distribution")

    ax.grid(True)
    ax.set_xlabel("Softmax value", fontsize=20)
    ax.set_ylabel("Density Estimate ( log scaled )", fontsize=20)
    
    plt.legend(loc="lower left")
    
    
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()
        
# Plot the softmax outputs as histogram for a given true particle events
def plot_softmax_histogram(softmax, labels, labels_dict=None, particle=None, save_path=None, num_bins=50):
    """
    Args: softmax ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels ... 1D array of true labels
          labels_dict ... Dictionary with the keys as particle types and values as col indices in the
                          softmax array
          particle ... string for the particle type e.g. "gamma", "e", "mu"
          save_path[optional] ... path to save the plot to
          num_bins[optional] ... number of bins to use in the histogram
    """
    assert softmax is not None and softmax.any() != None
    assert labels is not None and labels.any() != None
    assert labels_dict != None
    assert particle != None
    
    # Get the softmax values for the given true particle label
    curr_softmax = softmax[labels == labels_dict[particle],:]
    
    # Fix colors for each particle type
    color_dict = {"gamma":"r", "e":"b", "mu":"g"}
    
    # Plot the histograms for each of the particle classes
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    ax.tick_params(axis="both", labelsize=20)
    
    for key in labels_dict.keys():
        plt.hist(curr_softmax[:,labels_dict[key]], bins=num_bins, density=True,
                 label=key, color=color_dict[key], alpha=0.5, stacked=True)
        
    ax.grid(True)
    ax.set_xlabel("Softmax value", fontsize=20)
    ax.set_ylabel("Normalized count", fontsize=20)
    
    plt.legend(loc="upper left")
    plt.yscale("log")
    
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()
        
# Plot the softmax outputs for a given particle
def plot_particle_histogram(softmax, labels, labels_dict=None, particle=None, save_path=None, num_bins=50):
    """
    Args: softmax ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels ... 1D array of true labels
          labels_dict ... Dictionary with the keys as particle types and values as col indices in the
                          softmax array
          particle ... string for the particle type e.g. "gamma", "e", "mu"
          save_path[optional] ... path to save the plot to
          num_bins[optional] ... number of bins to use in the histogram
    """
    assert softmax is not None and softmax.any() != None
    assert labels is not None and labels.any() != None
    assert labels_dict != None
    assert particle != None
    
    # Fix colors for each particle type
    color_dict = {"gamma":"r", "e":"b", "mu":"g"}
    
    # Initialize the plot and corresponding parameters
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    ax.tick_params(axis="both", labelsize=20)
    
    for event_type in labels_dict.keys():
        label_to_use = r"$\{0}$ events".format(event_type) if event_type is not "e" else r"${0}$ events".format(event_type)
        curr_softmax = softmax[labels == labels_dict[event_type],labels_dict[particle]]
        plt.hist(curr_softmax, bins=num_bins, density=False, 
                 label= label_to_use,
                 color=color_dict[event_type], alpha=0.5, stacked=True)
        
    ax.grid(True)
    if particle is not "e":
        ax.set_xlabel(r"Classifier softmax output : $P(\{0})$".format(particle), fontsize=20)
    else:
        ax.set_xlabel(r"Classifier softmax output : $P(e)$".format(particle), fontsize=20)
    
    ax.set_ylabel("Count (Log scaled)", fontsize=20)
    
    plt.legend(loc="upper left")
    plt.yscale("log")
    
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()
        
# Plot the ROC curve for the model performance
def plot_ROC_curve(softmax, labels, class_dict, save_path=None):
    """
    Args: softmax ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels ... 1D array of true labels
          class_dict ... Dictionary with the keys as col indices in the softmax np array and values as particle
                         type ( str )
          save_path[optional] ... path to save the plot to
    """
    assert softmax.any() != None
    assert labels.any() != None
    assert class_dict != None
    assert softmax.shape[0] == labels.shape[0]
    
    # Compute the number of classes
    n_classes = len(class_dict.keys())
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold, roc_auc = dict(), dict(), dict(), dict()
    
    # Binarize the labels
    binary_labels = label_binarize(labels, classes=[0,1,2])
    
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(binary_labels[:,i], softmax[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all the ROC curves
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    
    plt.title(r"ROC Curve for $e, \mu$ and $\gamma$")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.tick_params(axis="both", labelsize=20)
    
    # Fix colors for each particle type
    color_dict = {"gamma":"r", "e":"b", "mu":"g"}
    
    for i, key in zip(range(n_classes), class_dict.keys()):
        if key is not "e":
            label_to_use = r"$\{0}$, AUC = ${1:0.3f}$".format(key,roc_auc[i])
        else:
            label_to_use = r"${0}$, AUC = ${1:0.3f}$".format(key,roc_auc[i])
            
        plt.plot(fpr[i], tpr[i], linewidth=2.0,
                 marker='.', markersize=8.0,
                 color=color_dict[key], label=label_to_use)
    
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    plt.legend(loc="lower right")
    
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()