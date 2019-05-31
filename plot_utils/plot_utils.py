"""
Source code borrowed from https://github.com/WatChMaL/UVicWorkshopPlayground/blob/master/B/notebooks/utils/utils.py
Edited by : Abhishek .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


# Set the style
plt.style.use("classic")

# Fix the colour scheme for each particle type
color_dict = {"gamma":"r", "e":"b", "mu":"g"}

# Function to convert from the true particle energies to visible energies
def convert_to_visible_energy(energies, labels):
    
    """
    convert_to_visible_energy(energies, labels)
    
    Purpose : Convert the true event energies to visible energy collected by the PMTs
    
    Args: energies ... 1D array of event energies, the length = sample size
          labels   ... 1D array of true label value, the length = sample size
    """
    
    # Convert true particle energies to visible energies
    m_mu = 105.7
    m_e = 0.511
    m_p = 0.511

    # Constant for the inverse refractive index of water
    beta = 0.75

    # Denominator for the scaling factor to be used for the cherenkov threshold
    dem = math.sqrt(1 - beta**2)
    
    # Perform the conversion from true particle energy to visible energy
    for i in range(len(energies)):
        if(labels[i] == 0):
            energies[i] = max((energies[i] - (m_e / dem) - (m_p / dem)), 0)
        elif(labels[i] == 1):
            energies[i] = max((energies[i] - (m_e / dem)), 0)
        elif(labels[i] == 2):
            energies[i] = max((energies[i] - (m_mu / dem)), 0)
        
    return energies

# Function to plot the energy distribution over a given dataset
def plot_event_energy_distribution(energies, labels, label_dict=None, dset_type="full", show_plot=False, save_path=None):
    
    """
    plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, save_path=None)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: energies            ... 1D array of event energies, the length = sample size
          labels              ... 1D array of true label value, the length = sample size
          labels_dict         ... Dictionary with the keys as event types and values as labels, default=None
          dset_type           ... String describing the type of dataset (full, train, validation, train), default="full"
          show_plot[optional] ... Boolean to determine whether to display the plot, default=False
          save_path[optional] ... Path to save the plot as an image, default=None
    """
    # Assertions
    assert label_dict != None
    
    # Extract the event energies corresponding to given event types
    energies_dict = {}
    for key in label_dict.keys():
        energies_dict[key] = energies[labels==label_dict[key]]
        
    fig, axes = plt.subplots(3,1,figsize=(16,12))
    plt.subplots_adjust(hspace=0.6)
    
    for label in energies_dict.keys():
        label_to_use = r"$\{0}$".format(label) if label is not "e" else r"${0}$".format(label)
        
        axes[label_dict[label]].hist(energies_dict[label], bins=50, density=False, label=label_to_use, alpha=0.8,
                        color=color_dict[label])
        axes[label_dict[label]].tick_params(labelsize=20)
        axes[label_dict[label]].legend(prop={"size":20})
        axes[label_dict[label]].grid(True, which="both", axis="both")
        axes[label_dict[label]].set_ylabel("Frequency", fontsize=20)
        axes[label_dict[label]].set_xlabel("Event Visible Energy (MeV)", fontsize=20)
        axes[label_dict[label]].set_xlim(0, max(energies)+20)
        axes[label_dict[label]].set_title("Energy distribution for " + label_to_use + " over the " + dset_type + " dataset",
                             fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the plot frame
        plt.close() # Close the opened window if any


# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, show_plot=False,
                          save_path=None):
    
    """
    plot_confusion_matrix(labels, predictions, energies, class_names, min_energy, max_energy, save_path=None)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          energies            ... 1D array of event energies, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
          min_energy          ... Minimum energy for the events to consider
          max_energy          ... Maximum energy for the events to consider
          show_plot[optional] ... Boolean to determine whether to display the plot
          save_path[optional] ... Path to save the plot as an image
    """
    
    # Create a mapping to extract the energies in
    energy_slice_map = [False for i in range(len(energies))]
    for i in range(len(energies)):
        if(energies[i] >= min_energy and energies[i] < max_energy):
                energy_slice_map[i] = True
                
    # Filter the CNN outputs based on the energy intervals
    labels = labels[energy_slice_map]
    predictions = predictions[energy_slice_map]
    
    if(show_plot or save_path is not None):
        fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
        num_labels = len(class_names)
        max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
        assert max_value < num_labels
        mat,_,_,im = ax.hist2d(predictions, labels,
                               bins=(num_labels,num_labels),
                               range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

        # Normalize the confusion matrix
        mat = mat.astype("float") / mat.sum(axis=0)[:, np.newaxis]

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=20) 
        
        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(class_names,fontsize=20)
        ax.set_yticklabels(class_names,fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_xlabel('Prediction',fontsize=20)
        ax.set_ylabel('True Label',fontsize=20)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                        ha="center", va="center", fontsize=20,
                        color="white" if mat[i,j] > (0.5*mat.max()) else "black")
        fig.tight_layout()
        plt.title("Confusion matrix, " + r"${0} \leq E < {1}$".format(min_energy, max_energy), fontsize=20) 
   
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the plot frame
        plt.close() # Close the opened window if any
        
        
# Plot the classifier for a given event type for several true event types
def plot_classifier_response(softmaxes, labels, energies, labels_dict=None, event_dict=None, min_energy=0,
                             max_energy=1000, num_bins=100, show_plot=False, save_path=None):
    
    """
    plot_classifier_response(softmaxes, labels, energies, labels_dict=None, event_dict=None, min_energy=0,
                             max_energy=1000, num_bins=100, show_plot=False, save_path=None)
                             
    Purpose : Plot the classifier softmax response for a given event type for several true event types
    
    Args: softmaxes             ... 2D array of softmax output, length = sample size, dimensions = (n_samples, n_classes)
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          labels_dict           ... Dictionary with the keys as event types and values as column indices in the softmax
                                    array, default=None
          event_dict            ... Dictionary with the key as the event type e.g. "gamma", "e", "mu" and value as the
                                    column index in the 2-D softmax array, default=None
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          num_bins[optional]    ... Number of bins to use per histogram, default=100
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
          
    """
    
    assert softmaxes is not None and softmaxes.any() != None
    assert labels is not None and labels.any() != None
    assert labels_dict != None
    assert event_dict != None
    
    
    # Initialize the plot and corresponding parameters
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")
    ax.tick_params(axis="both", labelsize=20)
    
    # Get the current event type from the event dict
    event = list(event_dict.keys())[0]
    
    for event_type in labels_dict.keys():
        
        label_to_use = r"$\{0}$ events".format(event_type) if event_type is not "e" else r"${0}$ events".format(event_type)
        
        # Get the softmax values for the given true event label
        label_map = [False for i in range(len(labels))]
        for i in range(len(labels)):
            if( labels[i] == labels_dict[event_type] ):
                label_map[i] = True
        
        # Get the softmax values for the given true event label
        curr_softmax = softmaxes[label_map]

        # Get the energy values for the given true event label
        curr_energies = energies[label_map]

        # Create a mapping to extract the energies in
        energy_slice_map = [False for i in range(len(curr_energies))]
        for i in range(len(curr_energies)):
            if(curr_energies[i] >= min_energy and curr_energies[i] < max_energy):
                    energy_slice_map[i] = True

        # Filter the CNN outputs based on the energy intervals
        curr_softmax = curr_softmax[energy_slice_map]
        curr_softmax = curr_softmax[:,event_dict[event]]
        
        if(curr_softmax.shape[0] <= 0):
            return None, None, None
        else:
            values, bins, patches = plt.hist(curr_softmax, bins=num_bins, density=False,
                                             label= label_to_use, color=color_dict[event_type],
                                             alpha=0.5, stacked=True)
        
    if save_path is not None or show_plot:
        ax.grid(True)
        if event is not "e":
            ax.set_xlabel(r"Classifier softmax output : $P(\{0})$".format(event), fontsize=20)
        else:
            ax.set_xlabel(r"Classifier softmax output : $P(e)$".format(event), fontsize=20)

        ax.set_ylabel("Count (Log scaled)", fontsize=20)
        plt.yscale("log")

        ax.set_xlim(0,1)

        plt.legend(loc="upper left", prop={"size":20}, bbox_to_anchor=(1.04,1))
        
        plt.title(r"${0} \leq E < {1}$".format(min_energy, max_energy), fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
        
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the current figure
        plt.close() # Close the opened window
        
    return values, bins, patches
        
# Plot the ROC curve for one vs another class
def plot_ROC_curve_one_vs_one(softmaxes, labels, energies, index_dict, label_0, label_1, min_energy, max_energy,
                              show_plot=False, save_path=None):
    """
    plot_ROC_curve_one_vs_one(softmaxes, labels, energies, index_dict, label_0, label_1, min_energy, max_energy,
                              show_plot=False, save_path=None)
                              
    Purpose : Plot the Reciver Operating Characteristic (ROC) curve given the softmax values and true labels
                              
    Args: softmaxes             ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          index_dict            ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmax array
          label_0               ... String identifying the first event type
          label_1               ... String identifying the second event type
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    assert softmaxes.any() != None
    assert labels.any() != None
    assert index_dict != None
    assert softmaxes.shape[0] == labels.shape[0]
    
    # Create a mapping to extract the energies in
    energy_slice_map = [False for i in range(len(energies))]
    for i in range(len(energies)):
        if(energies[i] >= min_energy and energies[i] < max_energy):
                energy_slice_map[i] = True
                
    # Filter the CNN outputs based on the energy intervals
    curr_softmax = softmaxes[energy_slice_map]
    curr_labels = labels[energy_slice_map]
    
    # Extract the useful softmax and labels from the input arrays
    softmax_0 = curr_softmax[curr_labels==index_dict[label_0]]# or 
    labels_0 = curr_labels[curr_labels==index_dict[label_0]] #or 
    
    softmax_1 = curr_softmax[curr_labels==index_dict[label_1]]
    labels_1 = curr_labels[curr_labels==index_dict[label_1]]
    
    # Add the two arrays
    softmax = np.concatenate((softmax_0, softmax_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)
    
    # Binarize the labels
    binary_labels_1 = label_binarize(labels, classes=[index_dict[label_0], index_dict[label_1]])
    binary_labels_0 = 1 - binary_labels_1

    # Compute the ROC curve and the AUC for class corresponding to label 0
    fpr_0, tpr_0, threshold_0 = roc_curve(binary_labels_0, softmax[:,index_dict[label_0]])
    roc_auc_0 = auc(fpr_0, tpr_0)
    
    # Compute the ROC curve and the AUC for class corresponding to label 1
    fpr_1, tpr_1, threshold_1 = roc_curve(binary_labels_1, softmax[:,index_dict[label_1]])
    roc_auc_1 = auc(fpr_1, tpr_1)
    
    if show_plot or save_path is not None:
        # Plot the ROC curves
        fig, ax = plt.subplots(figsize=(16,9),facecolor="w")
        ax.tick_params(axis="both", labelsize=20)

        ax.plot(fpr_0, tpr_0, color=color_dict[label_0],
                 label=r"$\{0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0) if label_0 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_0, roc_auc_0),
                 linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=color_dict[label_0])

        ax.plot(fpr_1, tpr_1, color=color_dict[label_1], 
                 label=r"$\{0}$, AUC ${1:0.3f}$".format(label_1, roc_auc_0) if label_1 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_1, roc_auc_0),
                 linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=color_dict[label_1])

        ax.grid(True)
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.set_title(r"${0} \leq E < {1}$".format(min_energy, max_energy), fontsize=20)
        ax.legend(loc="upper right", prop={"size":20}, bbox_to_anchor=(1.04,1))

    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the current figure
        plt.close() # Close the opened window
        
        
    return fpr_0, tpr_0, threshold_0, roc_auc_0, fpr_1, tpr_1, threshold_1, roc_auc_1

# Plot signal efficiency for a given event type at different energies
def plot_signal_efficiency(softmaxes, labels, energies, index_dict=None, event=None,
                           average_efficiencies=[0.2, 0.5, 0.8], energy_interval=25,
                           min_energy=100, max_energy=1000, num_bins=100, show_plot=False,
                           save_path=None):
    
    """
    plot_signal_efficiency(softmaxes, labels, energies, index_dict=None, event=None, thresholds=None,
                           energy_interval=5, energy_min=100, energy_max=1000, num_bins=100, save_path=None)
                           
    Purpose : Plot the signal efficiency vs energy for several thresholds
    
    Args: softmaxes             ... 2D array of softmax output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          index_dict            ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmax arrayy
          event                 ... String identifier for the event for which to plot the signal efficiency
          average_efficiencies  ... 1D array with the average efficiency values for which to plot the signal efficiency
                                    vs energy plot, default=[0.2, 0.5, 0.8] 
          energy_interval       ... Energy interval to be used to calculate the response curve and calculating the signal                 
                                    efficiency, default=5
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          num_bins              ... Number of bins to use in the classifier response histogram ( should be greater than 100
                                    to prevent 0 values
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    # Assertions to check for valid inputs
    assert softmaxes.any() != None
    assert labels.any() != None
    assert energies.any() != None
    assert index_dict != None
    assert event != None
    assert len(average_efficiencies) >= 3
    assert num_bins >= 100
    
    # Calculate the threshold here according to the desired average efficiencies
    _, _, threshold_0, _, _, tpr_1, threshold_1, _ = plot_ROC_curve_one_vs_one(softmaxes, labels, energies,
                                                                     {"gamma":0,"e":1}, "gamma",
                                                                      "e", 0, 1000, show_plot=False)
    
    threshold_index_dict = {}

    for tpr_value in average_efficiencies:
        index_list = []
        for i in range(len(tpr_1)):
            if(math.fabs(tpr_1[i]-tpr_value) < 0.001):
                index_list.append(i)
        index = index_list[math.ceil(len(index_list)/2)]
        threshold_index_dict[tpr_value] = index

    thresholds = []
    for key in threshold_index_dict.keys():
        thresholds.append(round(threshold_1[threshold_index_dict[key]], 2))
    
    # Get the energy intervals to plot the signal efficiency against ( replace with max(energies) ) 
    energy_lb = [min_energy+(energy_interval*i) for i in range(math.ceil((max_energy-min_energy)/energy_interval))]
    energy_ub = [energy_low+energy_interval for energy_low in energy_lb]
    
    # Local color dict
    local_color_dict = {}
    local_color_dict[thresholds[0]] = "green"
    local_color_dict[thresholds[1]] = "blue"
    local_color_dict[thresholds[2]] = "red"
    
    # Epsilon to ensure the plots are OK for low efficiency thresholds
    epsilon = 0.0001
    
    # Plot the signal efficiency vs energy
    fig = plt.figure(figsize=(32,18), facecolor="w")
        
    for threshold, efficiency in zip(thresholds, average_efficiencies):
        
        # Values to be plotted at the end
        signal_efficiency = []
        energy_values = []
        
        # Value for the previous non-zero events
        prev_non_zero_efficiency = 0.0
    
        # Iterate over the energy intervals computing the efficiency
        for energy_lower, energy_upper in zip(energy_lb, energy_ub):
            values, bins, _ = plot_classifier_response(softmaxes, labels, energies,
                                                      {event:index_dict[event]},
                                                      {event:index_dict[event]},
                                                      energy_lower, energy_upper,
                                                      num_bins=num_bins, show_plot=False)
            
            total_true_events = np.sum(values)
            num_true_events_selected = np.sum(values[bins[:len(bins)-1] > threshold-epsilon])

            curr_interval_efficiency = num_true_events_selected/total_true_events if total_true_events > 0 else 0

            if(curr_interval_efficiency == 0):
                curr_interval_efficiency = prev_non_zero_efficiency
            else:
                prev_non_zero_efficiency = curr_interval_efficiency

            # Add two times once for the lower energy bound and once for the upper energy bound
            signal_efficiency.append(curr_interval_efficiency)
            signal_efficiency.append(curr_interval_efficiency)

            # Add the lower and upper energy bounds
            energy_values.append(energy_lower)
            energy_values.append(energy_upper)

            label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)

        plt.plot(energy_values, signal_efficiency, color=local_color_dict[threshold], linewidth=2.0,
                 marker=".", markersize=6.0, markerfacecolor=local_color_dict[threshold], label=label_to_use)

    if(event is not "e"):
             title = r"Signal Efficiency vs Energy for $\{0}$ events.".format(event)
    else:
             title = r"Signal Efficiency vs Energy for ${0}$ events.".format(event)
             
    plt.title(title, fontsize=20)
    plt.grid(True)
             
    plt.xlim([min_energy, max_energy])
    plt.ylim([0, 1.05])
    plt.tick_params(axis="both", labelsize=20)
             
    plt.xlabel("Event Visible Energy (MeV)", fontsize=20)
    plt.ylabel("Signal Efficiency", fontsize=20)
    
    plt.legend(prop={"size":20}, bbox_to_anchor=(1.04,1), loc="upper left")
        
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.clf() # Clear the current figure
        plt.close() # Close the opened window
        
# Plot background rejection for a given event
def plot_background_rejection(softmaxes, labels, energies, index_dict=None, event=None,
                              average_efficiencies=[0.2, 0.5, 0.8], energy_interval=5,
                              min_energy=100, max_energy=1000, num_bins=100,
                              show_plot=False, save_path=None):
    
    """
    plot_background_rejection(softmaxes, labels, energies, index_dict=None, event=None,
                              average_efficiencies=[0.2, 0.5, 0.8], energy_interval=5,
                              min_energy=100, max_energy=1000, num_bins=100,
                              show_plot=False, save_path=None)
                           
    Purpose : Plot the background rejection vs energy for several thresholds
    
    Args: softmaxes             ... 2D array of softmaxes output, length = sample size, dimensions = n_samples, n_classes
          labels                ... 1D array of true labels
          energies              ... 1D array of visible event energies
          index_dict            ... Dictionary with the keys as event type (str) and values as the column indices 
                                    in the np softmaxes array
          event                 ... String identifier for the event for which to plot the background rejection
          average_efficiencies  ... 1D array with the average efficiency values for which to plot the signal efficiency
                                    vs energy plot, default=[0.2, 0.5, 0.8] 
          energy_interval       ... Energy interval to be used to calculate the response curve and calculating the signal                 
                                    efficiency, default=5
          min_energy            ... Minimum energy for the events to consider, default=0
          max_energy            ... Maximum energy for the events to consider, default=1000
          show_plot[optional]   ... Boolean to determine whether to show the plot, default=False
          save_path[optional]   ... Path to save the plot to, format='eps', default=None
    """
    
    # Assertions to check for valid inputs
    assert softmaxes.any() != None
    assert labels.any() != None
    assert energies.any() != None
    assert len(average_efficiencies) >= 3
    assert index_dict != None
    assert event != None
    assert num_bins >= 100
    
    # Calculate the threshold here according to the desired average efficiencies
    _, _, threshold_0, _, _, tpr_1, threshold_1, _ = plot_ROC_curve_one_vs_one(softmaxes, labels, energies,
                                                                     {"gamma":0,"e":1}, "gamma",
                                                                      "e", 0, 1000, show_plot=False)
    
    threshold_index_dict = {}

    for tpr_value in average_efficiencies:
        index_list = []
        for i in range(len(tpr_1)):
            if(math.fabs(tpr_1[i]-tpr_value) < 0.001):
                index_list.append(i)
        index = index_list[math.ceil(len(index_list)/2)]
        threshold_index_dict[tpr_value] = index

    thresholds = []
    for key in threshold_index_dict.keys():
        thresholds.append(round(threshold_1[threshold_index_dict[key]], 2))
    
    # Get the energy intervals to plot the signal efficiency against ( replace with max(energies) ) 
    energy_lb = [min_energy+(energy_interval*i) for i in range(math.ceil((max_energy-min_energy)/energy_interval))]
    energy_ub = [energy_low+energy_interval for energy_low in energy_lb]
    
    # Local color dict
    local_color_dict = {}
    local_color_dict[thresholds[0]] = "green"
    local_color_dict[thresholds[1]] = "blue"
    local_color_dict[thresholds[2]] = "red"
    
    # Epsilon to ensure the plots are OK for low efficiency thresholds
    epsilon = 0.0001
    
    # Plot the background rejection vs energy
    fig = plt.figure(figsize=(32,18), facecolor="w")
    
    for threshold, efficiency in zip(thresholds, average_efficiencies):
    
        # Initialize the dictionary to hold the background rejection values
        background_rejection_dict = {}
        for key in index_dict.keys():
            if(key != event):
                background_rejection_dict[key] = []
    
        energy_values = []
    
        # List of all the keys for background rejection
        background_rejection_keys = list(background_rejection_dict.keys())
    
        # Add an extra color to the color dict for total background rejection
        color_dict["total"] = "black"
    
        # Iterate over the energy intervals to compute the background rejection
        for key in background_rejection_dict.keys():

            # Value for the previous non-zero events
            prev_non_zero_rejection = 0.0

            # Initialize the dict to pass
            if( key == "total" ):
                pass_dict = index_dict.copy()
                del pass_dict[event]
            else:
                pass_dict = {key:index_dict[key]}

            for energy_lower, energy_upper in zip(energy_lb, energy_ub):

                values, bins, _ = plot_classifier_response(softmaxes, labels, energies, pass_dict,
                                                          {event:index_dict[event]},
                                                          energy_lower, energy_upper, 
                                                          num_bins=num_bins, show_plot=False)

                # Find the number of false events rejected
                total_false_events = np.sum(values)
                num_false_events_rejected = np.sum(values[bins[:len(bins)-1] < threshold])

                curr_interval_rejection = num_false_events_rejected/total_false_events if total_false_events > 0 else 0

                if(curr_interval_rejection == 0):
                    curr_interval_rejection = prev_non_zero_rejection
                else:
                    prev_non_zero_rejection = curr_interval_rejection

                # Add two times once for the lower energy bound and once for the upper energy bound
                background_rejection_dict[key].append(curr_interval_rejection)
                background_rejection_dict[key].append(curr_interval_rejection)

                # If the key is the last key in the dict
                if( key == background_rejection_keys[len(background_rejection_keys)-1] ):

                    # Add the lower and upper energy bounds
                    energy_values.append(energy_lower)
                    energy_values.append(energy_upper)
                    
        for key in background_rejection_keys:
            
            label_to_use = None
            if( key == "total" ):
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)
            elif( key == "e" ):
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)
            else:
                label_to_use = r"Average signal efficiency = {0}, Threshold = {1:0.3f}".format(efficiency, threshold)

            plt.plot(energy_values, background_rejection_dict[key], color=local_color_dict[threshold], linewidth=2.0,
                     marker=".", markersize=6.0, markerfacecolor=local_color_dict[threshold], label=label_to_use)
        
    # Delete the total key from the color dict
    del color_dict["total"]
             
    if event is not "e" and key is not "e":
        title = r"$\{0}$ Background rejection vs Energy for selecting $\{1}$ events.".format(key, event)
    elif event is "e":
        title = r"$\{0}$ Background rejection vs Energy for selecting ${1}$ events.".format(key, event)
    elif key is "e":
        title = r"${0}$ Background rejection vs Energy for selecting $\{1}$ events.".format(key, event)
             
    plt.title(title, fontsize=20)
    plt.grid(True)
             
    plt.xlim([min_energy, max_energy])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis="both", labelsize=20)
             
    plt.xlabel("Event visible energy (MeV)", fontsize=20)
    plt.ylabel("Background rejection", fontsize=20)
    plt.legend(prop={"size":20}, bbox_to_anchor=(1.04,1), loc="upper left")
        
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=300)
    else:
        plt.show()
    