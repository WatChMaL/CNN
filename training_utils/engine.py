'''
Author: Wojciech Fedorko
Collaborators: Julian Ding, Abhishek Kajal
'''
# ======================== TEST IMPORTS =====================================
import collections
import sys
# ===========================================================================

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import time
import numpy as np

from io_utils.data_handling import WCH5Dataset
from io_utils import ioconfig 
from plot_utils.notebook_utils import CSVData
from plot_utils.plot_utils import plot_confusion_matrix

from training_utils.doublepriorityqueue import DoublePriority

GAMMA, ELECTRON, MUON = 0, 1, 2

ROOT_DUMP = 'ROOTS.txt'

EVENT_CLASS = {GAMMA : 'gamma', ELECTRON : 'electron', MUON : 'muon'}

class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model, config):
        self.model = model
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]

            print("Main GPU: "+self.devids[0])
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)

                print("CUDA is available")
            else:
                self.device=torch.device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=torch.device("cpu")

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None

        # NOTE: The functionality of this block is coupled to the implementation of WCH5Dataset in the iotools module
        self.dset=WCH5Dataset(config.path,
                              config.val_split,
                              config.test_split,
                              shuffle=config.shuffle,
                              reduced_dataset_size=config.subset)

        self.train_iter=DataLoader(self.dset,
                                   batch_size=config.batch_size_train,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices))
        
        self.val_iter=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        
        self.test_iter=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))

        self.dirpath=config.dump_path + time.strftime("%Y%m%d_%H%M%S") + "/"
        
        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        self.config=config
        
        # Save a copy of the config in the dump path
        ioconfig.saveConfig(self.config, self.dirpath + "config_file.ini")


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)
                        
            # Prediction
            #print("this is the data size before permuting: {}".format(data.size()))
            self.data = self.data.permute(0,3,1,2)
            #print("this is the data size after permuting: {}".format(data.size()))
            prediction = self.model(self.data)
            # Training
            loss = -1
            loss = self.criterion(prediction,self.label)
            self.loss = loss
            
            softmax    = self.softmax(prediction).cpu().detach().numpy()
            prediction = torch.argmax(prediction,dim=-1)
            accuracy   = (prediction == self.label).sum().item() / float(prediction.nelement())        
            prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

    def backward(self):
        self.optimizer.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.optimizer.step()
        
    # ========================================================================
    def train(self, epochs=3.0, report_interval=10, valid_interval=1000, save_interval=1000):
        # CODE BELOW COPY-PASTED FROM [HKML CNN Image Classification.ipynb]
        # (variable names changed to match new Engine architecture. Added comments and minor debugging)
        
        # Keep track of the validation accuracy
        best_val_acc = 0.0
        continue_train = True
        
        # Prepare attributes for data logging
        self.train_log, self.val_log = CSVData(self.dirpath+"log_train.csv"), CSVData(self.dirpath+"val_test.csv")
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        iteration = 0
        # Training loop
        while ((int(epoch+0.5) < epochs) and continue_train):
            print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            j = 0
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                
                # Data and label
                self.data = data[0]
                self.label = data[1].long()
                
                # Move the data and labels on the GPU
                self.data = self.data.to(self.device)
                self.label = self.label.to(self.device)
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                # Log/Report
                #
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                self.train_log.write()
                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    #print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (iteration,epoch,res['loss'],res['accuracy']))#
                    pass
                    
                # more rarely, run validation
                if (i+1)%valid_interval == 0:
                    with torch.no_grad():
                        self.model.eval()
                        val_data = next(iter(self.val_iter))
                        
                        # Data and label
                        self.data = val_data[0]
                        self.label = val_data[1].long()
                        
                        res = self.forward(False)
                        self.val_log.record(['iteration','epoch','accuracy','loss'],[iteration,epoch,res['accuracy'],res['loss']])
                        self.val_log.write()
                    self.model.train()
                    
                    if(res["accuracy"]-best_val_acc > 1e-03):
                        continue_train = True
                        best_val_acc = res["accuracy"]
                    else:
                        continue_train = True
                        
                if epoch >= epochs:
                    break
                    
                print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' %(iteration,epoch,res['loss'],res['accuracy']))
                    
                # Save on the given intervals
                if(i+1)%save_interval == 0:
                    with torch.no_grad():
                        
                        self.model.eval()
                        val_data = next(iter(self.val_iter))
                        
                        # Data and label
                        self.data = val_data[0]
                        self.label = val_data[1].long()
                        
                        res = self.forward(False)
                        
                        if(res["accuracy"]-best_val_acc > 1e-03):
                            self.save_state(curr_iter=0)
                            continue_train = True
                            best_val_acc = res["accuracy"]
                        else:
                            continue_train = True
                    self.save_state(curr_iter=iteration)
        self.val_log.close()
        self.train_log.close()
    
    # ========================================================================

    # Function to test the model performance on the validation
    # dataset ( returns loss, acc, confusion matrix )
    def validate(self, plt_worst=0, plt_best=0):
        """
        Test the trained model on the validation set.
        
        Parameters: None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
        
        # Run number
        run = 8
        
        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes, energies = [],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for val_data in iter(self.val_iter):
                
                sys.stdout.write("val_iterations : " + str(val_iterations) + "\n")
                
                self.data, self.label, index, batch_energies = val_data[0:4]
                
                self.label = self.label.long()

                # Run the forward procedure and output the result
                result = self.forward(False)
                val_loss += result['loss']
                val_acc += result['accuracy']
                
                # Add item to priority queues if necessary
                
                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")
                
                # Add the local result to the final result
                labels.extend(self.label)
                predictions.extend(result['prediction'])
                softmaxes.extend(result["softmax"])
                energies.extend(batch_energies)
                
                val_iterations += 1
                
        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        # If requested, dump list of root files + indices to dump_path directory
        if pushing:
            root_path = (os.path.dirname(self.config.path)+'/' if self.config.root is None else self.config.root)+ROOT_DUMP
            plot_path = self.config.dump_path+"extreme_events/"
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            wl_lo = open(plot_path+'list_lo.txt', 'w+')
            wl_hi = open(plot_path+'list_hi.txt', 'w+')
            worst, best = [], []
            for i in range(len(queues)):
                q = queues[i]
                # Lowest softmax are worst
                worst.extend(q.getsmallest())
                # Highest softmax are best
                best.extend(q.getlargest())
                
            root_list = open(root_path, 'r')
            root_files = [l.strip() for l in root_list.readlines()]
                
            for event in worst:
                wl_lo.write(str(event[0])+' '+root_files[event[1]]+' '+str(event[2])+'\n')
            for event in best:
                wl_hi.write(str(event[0])+' '+root_files[event[1]]+' '+str(event[2])+'\n')
            
            wl_lo.close()
            wl_hi.close()
            root_list.close()
            
            print("Dumped lists of extreme events at", plot_path)
        
        np.save("labels" + str(run) + ".npy", np.hstack(labels))
        np.save("energies" + str(run) + ".npy", np.hstack(energies))
        np.save("predictions" + str(run) + ".npy", np.hstack(predictions))
        np.save("softmax" + str(run) + ".npy", np.array(softmaxes))
        # If requested, dump root file visualization script outputs to dump_path directory
            
    # Function to test the model performance on the test
    # dataset ( returns loss, acc, confusion matrix )
    
    def test(self):
        r"""Test the trained model on the test dataset.
        
        Parameters: None
        
        Outputs : 
            total_test_loss = accumulated validation loss
            avg_test_loss = average validation loss
            total_test_acc = accumulated validation accuracy
            avg_test_acc = accumulated validation accuracy
            
        Returns : None
        """
        # Variables to output at the end
        test_loss = 0.0
        test_acc = 0.0
        test_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Extract the event data and label from the DataLoader iterator
            for test_data in iter(self.test_iter):
                
                sys.stdout.write("\r\r\r" + "test_iterations : " + str(test_iterations))
                
                self.data, self.label = test_data[0:2]
                self.label = self.label.long()
                
                counter = collections.Counter(self.label.tolist())
                sys.stdout.write("\ncounter : " + str(counter))

                # Run the forward procedure and output the result
                result = self.forward(False)
                test_loss += result['loss']
                test_acc += result['accuracy']
                
                test_iterations += 1
         
        print("\nTotal test loss : ", test_loss,
              "\nTotal test acc : ", test_acc,
              "\nAvg test loss : ", test_loss/test_iterations,
              "\nAvg test acc : ", test_acc/test_iterations)
        
    def get_top_bottom_softmax(self, n_top=5, n_bottom=5, event_type=None, label_dict=None):
        r"""Return the events with the highest and lowest softmax scores for
            visualizing the model performance
        
        Parameters: None
        
        Outputs : 
            n_top = number of events with the highest softmax score to return
                    for the given event type
            n_bottom = number of events with the lowest softmax score to return
                       for the given event type
            event_type = type of neutrino event to get the event data for
            label_dict = dictionary that maps the event type to the labels
                         used in the label tensor
            
            
        Returns : Numpy array of event data for the events with the highest
                  and lowest softmax score
        
        """
        
        # Variables to add or remove events
        softmax_top = np.array([-1 for i in range(n_top)])
        softmax_bottom = np.array([2 for i in range(n_bottom)])
        
        # Iterate over the validation set to get the desired events
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
             # Extract the event data and label from the DataLoader iterator
            for val_data in iter(self.val_iter):
                
                self.data, self.label = val_data[0:2]
                self.label = self.label.long()
                
                print(self.data.shape)
                print(self.label.shape)
                
                # Use only the labels and event for the given event type
                self.data = self.data[self.label == label_dict[event_type]]
                self.label = self.label[self.label == label_dict[event_type]]
                
                print(self.data.shape)
                print(self.label.shape)
            
                result = self.forward(False)
                
                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")
                
                # Sort the softmax output to get the indices for the top and 
                # bottom events to return or save
                softmax_indices_sorted = np.argsort(result["softmax"][:,label_dict[event_type]])
                
                # Get the indices for the top and bottom events
                softmax_top_n = softmax_indices_sorted[softmax_indices_sorted.shape[0]-n_top:]
                softmax_bottom_n = softmax_indices_sorted[:n_bottom]
                
                # Append the local top and bottom items to the global top and bottom items
                softmax_top = np.append(softmax_top,
                                        result["softmax"][softmax_top_n,label_dict[event_type]])
                softmax_bottom = np.append(softmax_bottom,
                                        result["softmax"][softmax_bottom_n,label_dict[event_type]])
                
                # Sort the global top and bottom softmax array and get the top and bottom sections
                #softmax_top 
                
                
        
        
    # ========================================================================
    
            
    def save_state(self, curr_iter=0):
        filename = self.dirpath + str(self.config.model[1]) + ".pth"
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_state(self, weight_file):
        weight_file = self.config.restore_state
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f,map_location="cuda:0" if (self.config.device == 'gpu') else 'cpu')
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
        print('Restoration complete.')
            
