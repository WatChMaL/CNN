# Comparison of FiTQun and ResNet-18 Results on Test Events


## Intro

To assess ResNet-18's performance and potential use, we needed to compare its results with the results of fiTQun when run on the same events. The test events used came from the original un-split (into trainval and testing) dataset `/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M.h5` - and the indices `/fast_scratch/WatChMaL/data/IWCD_fulltank_300_pe_idxs.npz` were used to isolate the test events from the other events in the h5 file.

Due to how the simulated data was created, the results of each of these methods were mapped to each other using the root files and eventids for each event. The root file points to a specific file of events, and the eventid specifies which event within the file that result corresponds to.

To make it easier to match up the events between fiTQun and ResNet, several files of indices were saved.

## Table of Contents

### 1. [FiTQun Results](#fitqun_results)
### 2. [Mapping ResNet-18 to FiTQun](#mapping)
### 3. [Comparing Classifier Output](#results)
### 4. [Energy Binning](#energy)
### 5. [Position Binning](#position)

## FiTQun Results <a id="fitqun_results"></a>


The fiTQun results for the test set are stored in 3 separate files, one each for electron, muon, and gamma events:
1. `/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_e-.npz`
2. `/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_mu-.npz`
3. `/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_gamma-.npz`

Each result has 8 tags associated with it: `filename`, `eventid`, `flag`, `momentum`, `time`, `nLL`, `position`, and `direction`.

`filename` and `eventid` have one element per event, which are the filename (the fitqun output root filenames, which are related to the wcsim output root file names) and the event ID in the original file (zero indexed)

`flag`, `momentum`, `time` and `nLL` each have two values per event, while `position` and `direction` have two sets of three values (3D vectors) each. The two values for all of these things are because it does two fits, one assuming electron and one assuming muon. The first in the array is assuming it is an electron event, and the second is assuming it is a muon event.

`flag` is related to whether fiTQun decides it's a good event.

`nLL` is the negative log-likelihood - basically equivalent to the classifier output.
`momentum`, `position`, `direction` and `time` are the reconstructed particle's initial properties and reconstructed event start time.


## Mapping Resnet-18 to FiTQun <a id="mapping"></a>

The script `ResNet-18 to FiTQun.ipynb` was used to map the results of the ResNet-18 model for the test events to fiTQun's results using their root files and eventids. This script saved the softmax output at `/home/ttuinstr/VAE/Comparing_ResNet_and_FiTQun/resnet_softmaxes.npz`, as well as appropriately labelled npz files for the corresponding predicted labels, eventids, rootfiles, and energies of each event. These npz files have the same event order as the test events from the original h5 file (when isolated using the previously mentioned indices).

Next, indices that separate these ordered events into electron, muon, and gamma events are applied. These are `test_indices_e.npz`, `test_indices_mu.npz`, and `test_indices_gamma.npz`. After the events have been separated, `map_indices_e.npz` (or `map_indices_e_all.npz` to include events that have been flagged by fiTQun) are used to map the ResNet electron events to the fiTQun electron events, and same for the similarly labelled muon and gamma indices. At this point, the ResNet-18 events will be in the same order as the fiTQun events.

## Comparing Classifier Output <a id="results"></a>

The classification output of ResNet-18 and fiTQun is directly compared in the notebook `Direct Result Comparison`. For fiTQun, the difference in nLL values for the muon and electron hypothesis is used as input to sklearn's roc function, while ResNet-18 uses the softmax output. The fpr and tpr from this are then used to overlay ROC curves. A confusion matrix is also made for each method. 

## Energy Binning <a id="energy"></a>

Two notebooks were primarily used for energy binning - `ResNet-18 Energy Binning` and `fiTQun Energy Binning`. These scripts were used to bin events by energy intervals and plot their background rejection at a fixed efficiency. `fiTQun Energy Binning` provided additional functionality where the events were binned by energy, and then for each energy bin the events were further binned by their radius from the center of the tank and their background rejection plotted at a fixed efficiency.

## Position Binning <a id="position"></a>

Two notebooks were primarily used for position binning - `ResNet-18 Position Binning` and `fiTQun Position Binning`. These scripts were used to bin events by intervals of their radius from the center of the tank and plot their background rejection at a fixed efficiency. `fiTQun Position Binning` also has True vs. Reconstructed Position histogram using the squared radius of the particle form the center of the tank.

The notebook `fiTQun Barrel Containment` was used to assess any difference in performance when the data was restricted to events that hit only in the barrel of the tank (not hitting the end caps).


## Flagged Events

About 22% of the events that passed through fiTQun were flagged as having failed reconstruction. These events were further explored in the notebook `Plotting Flagged Events`, where the events images were found to be low energy and often originating near the tank edge, resulting in black event images.

```bash

```
