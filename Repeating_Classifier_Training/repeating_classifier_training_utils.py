# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc
import os,sys
import matplotlib.gridspec as gridspec


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def disp_learn_hist_smoothed(location, losslim=None, window_train=400,window_val=40,show=True):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'
    
    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)

    epoch_train    = moving_average(np.array(train_log_csv.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_csv.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_csv.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_csv.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_csv.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_csv.loss),window_val)

    epoch_val_uns    = np.array(val_log_csv.epoch)
    accuracy_val_uns = np.array(val_log_csv.accuracy)
    loss_val_uns     = np.array(val_log_csv.loss)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(epoch_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch_val, loss_val, label='Average validation loss', color='blue')

    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(epoch_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch_val, accuracy_val, label='Average validation accuracy', color='red')    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)

    lines  = line11+ line12+ line21+ line22

    labels = [l.get_label() for l in lines]
    
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1,prop={'size' : 6})
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return

    return fig

# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, class_names,title=None):
    
    """
    plot_confusion_matrix(labels, predictions, class_names)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
       
 
    """
    
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(predictions, labels,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

    # Normalize the confusion matrix
    mat = mat.astype("float") / mat.sum(axis=0)

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
    if title is not None: 
        ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    plt.show()

# Plot multiple ROC curves on the same figure
def plot_multiple_ROC(fprs, tprs, thresholds, label_0, label_1, png_name='roc_plot',title='ROC Curve'):
    
    min_energy = 0
    max_energy = 1000
    
    fig, ax = plt.subplots(figsize=(16,9),facecolor="w")
    ax.tick_params(axis="both", labelsize=20)
    
    model_colors = [np.random.rand(3,) for i in fprs]
    
    for j in np.arange(len(fprs)):
        fpr = fprs[j]
        tpr = tprs[j]
        threshold = thresholds[j]
     
        roc_auc = auc(fpr, tpr)

        inv_fpr = []
        for i in fpr:
            inv_fpr.append(1/i) if i != 0 else inv_fpr.append(1/1e-5)

        tnr = 1. - fpr

        # TNR vs TPR plot

        ax.plot(tpr, inv_fpr,
                 label=r"${1:0.3f}$: $\{0}$, AUC ${1:0.3f}$".format((j),label_0, roc_auc) if label_0 is not "e" else r"${0}$, AUC ${1:0.3f}$".format(label_0, roc_auc),
                 linewidth=1.0, marker=".", markersize=4.0, markerfacecolor=model_colors[j])

        # Show coords of individual points near x = 0.2, 0.5, 0.8
        todo = {0.2: True, 0.5: True, 0.8: True}
        for xy in zip(tpr, inv_fpr, tnr):
            xy = (round(xy[0], 4), round(xy[1], 4), round(xy[2], 4))
            xy_plot = (round(xy[0], 4), round(xy[1], 4))
            for point in todo.keys():
                if xy[0] >= point and todo[point]:
                    ax.annotate('(%s, %s, %s)' % xy, xy=xy_plot, textcoords='data', fontsize=18, bbox=dict(boxstyle="square", fc="w"))
                    todo[point] = False

            ax.grid(True, which='both', color='grey')
            xlabel = r"$\{0}$ signal efficiency".format(label_0) if label_0 is not "e" else r"${0}$ signal efficiency".format(label_0)
            ylabel = r"$\{0}$ background rejection".format(label_1) if label_1 is not "e" else r"${0}$ background rejection".format(label_1)

            ax.set_xlabel(xlabel, fontsize=20) 
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_title(title)
            ax.legend(loc="upper right", prop={"size":20})

            plt.margins(0.1)
        plt.yscale("log")
        
    plt.savefig(os.path.join(os.getcwd(),png_name), bbox_inches='tight')    
    
    plt.show()

    plt.clf() # Clear the current figure
    plt.close() # Close the opened window
        
        
    return fpr, tpr, threshold, roc_auc

def prep_roc_data(softmaxes,labels, energies,softmax_index_dict,label_0,label_1,threshold=None):
    """
    prep_roc_data(softmaxes,labels, energies,index_dict,threshold=None)

    Purpose : Prepare data for plotting the ROC curves. If threshold is not none, filters 
    out events with energy greater than threshold. Returns true positive rates, positive 
    rates, and thresholds for plotting the ROC curve.

    Args: labels              ... 1D array of true label value, the length = sample size
          softmaxes           ... array of resnet softmax output, the 0th dim= sample size
          energies            ... 1D array of true event energies, the length = sample 
                                  size
          softmax_index_dict  ... Dictionary pointing to label integer from particle name
          label_0 and label_1 ... Labels indicating which particles to use
    """    
    if threshold is not None:
        low_energy_idxs = np.where(energies < threshold)
        rsoftmaxes = softmaxes[low_energy_idxs]
        rlabels = labels[low_energy_idxs]
        renergies = energies[low_energy_idxs]
    else:
        rsoftmaxes = softmaxes
        rlabels = labels
        renergies = energies
        
    g_test_indices = np.where(rlabels==softmax_index_dict['gamma'])[0]
    e_test_indices = np.where(rlabels==softmax_index_dict['e'])[0]
    m_test_indices = np.where(rlabels==softmax_index_dict['mu'])[0]
    
    g_labels = rlabels[g_test_indices]
    e_labels = rlabels[e_test_indices]
    m_labels = rlabels[m_test_indices]

    g_softmaxes = rsoftmaxes[g_test_indices]
    e_softmaxes = rsoftmaxes[e_test_indices]
    m_softmaxes = rsoftmaxes[m_test_indices]

    g_energies = renergies[g_test_indices]
    e_energies = renergies[e_test_indices]
    m_energies = renergies[m_test_indices]
    
    e_softmax_0 = e_softmaxes[e_labels==softmax_index_dict[label_0]] 
    mu_softmax_0 = m_softmaxes[m_labels==softmax_index_dict[label_0]]
    gamma_softmax_0 = g_softmaxes[g_labels==softmax_index_dict[label_0]]

    e_labels_0 = e_labels[e_labels==softmax_index_dict[label_0]] 
    mu_labels_0 = m_labels[m_labels==softmax_index_dict[label_0]]
    gamma_labels_0 = g_labels[g_labels==softmax_index_dict[label_0]]

    e_softmax_1 = e_softmaxes[e_labels==softmax_index_dict[label_1]] 
    mu_softmax_1 = m_softmaxes[m_labels==softmax_index_dict[label_1]]
    gamma_softmax_1 = g_softmaxes[g_labels==softmax_index_dict[label_1]]

    e_labels_1 = e_labels[e_labels==softmax_index_dict[label_1]] 
    mu_labels_1 = m_labels[m_labels==softmax_index_dict[label_1]]
    gamma_labels_1 = g_labels[g_labels==softmax_index_dict[label_1]]
    
    total_softmax = np.concatenate((e_softmax_0, e_softmax_1, mu_softmax_0, mu_softmax_1), axis=0)
    total_labels = np.concatenate((e_labels_0, e_labels_1, mu_labels_0, mu_labels_1), axis=0)
    
    return roc_curve(total_labels, total_softmax[:,softmax_index_dict[label_0]], pos_label=softmax_index_dict[label_0])


def disp_multiple_learn_hist(locations,losslim=None,show=True,titles=None,best_only=False,leg_font=10):
    ncols = len(locations) if len(locations) < 3 else 3
    nrows = math.ceil(len(locations)/3)
    fig = plt.figure(facecolor='w',figsize=(12,nrows*4))
    gs = gridspec.GridSpec(nrows,ncols,figure=fig)
    axes = []
    for i,location in enumerate(locations):
        train_log=location+'/log_train.csv'
        val_log=location+'/log_val.csv'        
        train_log_csv = pd.read_csv(train_log)
        val_log_csv  = pd.read_csv(val_log)
        
        if best_only:
            best_idxs = [0]
            best_epoch=0
            best_loss = val_log_csv.loss[0]
            for idx,loss in enumerate(val_log_csv.loss):
                if loss < best_loss: 
                    best_loss=loss
                    best_idxs.append(idx)
                    best_epoch=val_log_csv.epoch[idx]
            val_log_csv = val_log_csv.loc[best_idxs]
            if titles is not None:
                titles[i] = titles[i] + ", Best Val Loss ={loss:.4f}@Ep.{epoch:.2f}".format(loss=best_loss,epoch=best_epoch)
                
        ax1=fig.add_subplot(gs[i],facecolor='w') if i ==0 else fig.add_subplot(gs[i],facecolor='w',sharey=axes[0])
        ax1.set_xlim(0,train_log_csv.epoch.max())
        axes.append(ax1)
        line11 = ax1.plot(train_log_csv.epoch, train_log_csv.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
        line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')
        if losslim is not None:
            ax1.set_ylim(None,losslim)
        if titles is not None:
            ax1.set_title(titles[i])
        ax2 = ax1.twinx()
        line21 = ax2.plot(train_log_csv.epoch, train_log_csv.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
        line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

        ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
        ax1.tick_params('x',colors='black',labelsize=18)
        ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
        ax1.tick_params('y',colors='b',labelsize=18)


        ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
        ax2.tick_params('y',colors='r',labelsize=18)
        ax2.set_ylim(0.,1.05)

        lines  = line11 + line12 + line21 + line22
        labels = [l.get_label() for l in lines]
        leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1,prop={'size':leg_font})
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor('white')
    gs.tight_layout(fig)
    return fig


# Function to plot a confusion matrix
def plot_multiple_confusion_matrix(label_arrays, prediction_arrays, class_names,titles=None):
    fig = plt.figure(facecolor='w',figsize=(16,8))
    gs = gridspec.GridSpec(math.ceil(len(label_arrays)/3),3,figure=fig)
    axes = []
    
    """
    plot_confusion_matrix(labels, predictions, class_names)
    
    Purpose : Plot the confusion matrix for a series of test outputs.
    
    Args: label_arrays        ... array of 1D arrays of true label value, the length = sample size
          predictions         ... array of 1D arrays of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
       
 
    """
    for i,labels in enumerate(label_arrays):
        predictions = prediction_arrays[i]
        ax=fig.add_subplot(gs[i],facecolor='w')
        num_labels = len(class_names)
        max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
        assert max_value < num_labels
        mat,_,_,im = ax.hist2d(predictions, labels,
                               bins=(num_labels,num_labels),
                               range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

        # Normalize the confusion matrix
        mat = mat.astype("float") / mat.sum(axis=0)

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
        if titles is not None: 
            ax.set_title(titles[i])

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                        ha="center", va="center", fontsize=20,
                        color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    gs.tight_layout(fig)
    return fig


def load_test_output(location,index_path):
    """
    load_test_output(location,index_path)
    
    Purpose : Load output of a test run on the full h5 test set, 
              remove FiTQun flagged/failed events, and return a dict of results.
    
    Args: location            ... string, path of the directory containing the test 
                                  output eg. '/home/cmacdonald/CNN/dumps/20200525_152544'
          index_path          ... string, path of directory containing indices of FiTQun failed and flagged files
       
 
    """
    test_dump_np = np.load(location, allow_pickle=True)
    
    res_predictedlabels = np.concatenate(list([batch_array for batch_array in test_dump_np['predicted_labels']]))
    res_softmaxes  = np.concatenate(list([batch_array for batch_array in test_dump_np['softmax']]))
    res_labels   = np.concatenate(list([batch_array for batch_array in test_dump_np['labels']]))
    res_energies = np.concatenate(list([batch_array for batch_array in test_dump_np['energies']]))
    res_rootfiles = np.concatenate(list([batch_array for batch_array in test_dump_np['rootfiles']]))
    res_eventids = np.concatenate(list([batch_array for batch_array in test_dump_np['eventids']]))
    res_angles = np.concatenate(list([batch_array for batch_array in test_dump_np['angles']]))
    
    failed_idxs = np.load(os.path.join(index_path, 'fq_failed_idxs.npz'),allow_pickle=True)['failed_indices_pointing_to_h5_test_set'].astype(int)
    flagged_idxs = np.load(os.path.join(index_path, 'fq_flagged_idxs.npz'),allow_pickle=True)['arr_0'].astype(int)
    
    sres_predictedlabels = np.delete(res_predictedlabels,failed_idxs)
    sres_softmaxes  = np.delete(res_softmaxes,failed_idxs,0)
    sres_labels  = np.delete(res_labels,failed_idxs)
    sres_energies = np.delete(res_energies,failed_idxs)
    sres_rootfiles = np.delete(res_rootfiles,failed_idxs)
    sres_eventids = np.delete(res_eventids,failed_idxs)
    sres_angles = np.delete(res_angles,failed_idxs,0)
    
    filtered_res_predictedlabels = np.delete(sres_predictedlabels,flagged_idxs)
    filtered_res_softmaxes  = np.delete(sres_softmaxes,flagged_idxs,0)
    filtered_res_labels  = np.delete(sres_labels,flagged_idxs)
    filtered_res_energies = np.delete(sres_energies,flagged_idxs)
    filtered_res_rootfiles = np.delete(sres_rootfiles,flagged_idxs)
    filtered_res_eventids = np.delete(sres_eventids,flagged_idxs)
    filtered_res_angles = np.delete(sres_angles,flagged_idxs,0)
    
    return{'filtered_predictions':filtered_res_predictedlabels,
            'filtered_softmaxes':filtered_res_softmaxes,
            'filtered_labels':filtered_res_labels,
            'filtered_energies':filtered_res_energies,
            'filtered_rootfiles':filtered_res_rootfiles,
            'filtered_eventids':filtered_res_eventids,
            'filtered_angles':filtered_res_angles          
          }


def parametrized_ray_point(x,y,z,theta,phi,t):
    return x + np.sin(theta)*np.cos(phi)*t,y + np.sin(theta)*np.sin(phi)*t, z + np.cos(theta)*t


def distance_to_wall(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
       
    """
    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False
    sols = []
    #Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        shared_expression = np.sqrt(-np.sin(theta)**2*(-275282+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi)
                                     +shared_expression)
            if radial_parameter_sol_1 > 0: sols.append(radial_parameter_sol_1)
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi)
                                     +shared_expression)
            if radial_parameter_sol_2 > 0: sols.append(radial_parameter_sol_2)
        except:
            pass
    try:
        cap_parameter_sol_top = (521 - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(521+z)/np.cos(theta)
        if cap_parameter_sol_top > 0: sols.append(cap_parameter_sol_top)
        if cap_parameter_sol_bottom > 0: sols.append(cap_parameter_sol_bottom)
    except:
        pass
    sols = np.sort(sols)
    x_int,y_int,z_int = parametrized_ray_point(x,y,z,theta,phi,sols[0])
    return np.sqrt((x-x_int)**2+(y-y_int)**2+(z-z_int)**2)
