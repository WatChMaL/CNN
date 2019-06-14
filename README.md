# Water Cherenkov Machine Learning Variational AutoEncoder (WatChMaL-VAE)


## Description

Python implementation of the training engine and framework to build, train and test VAE models for Water Cherenkov Detectors.

## Table of Contents

### 1. [Directory Layout](#directory_layout)
### 2. [Installation](#installation)
### 3. [Usage](#usage)
### 4. [Credits](#credits)
### 5. [License](#license)

## Directory Layout <a id="directory_layout"></a>

```bash
.
+-- config                              # Configuration files
  +-- engine_config                     # Configuration files for the training engine
    +-- test_resnet.ini
    +-- test_kazunet.ini
    +-- test_kvaenet.ini
  +-- preprocessing_config              # Configuration files for data pre-processing
    +-- merge_config.ini
+-- io_utils                            # Tools to handle the user inputs, dataset and models
  +-- __init__.py
  +-- arghandler.py
  +-- data_handling.py
  +-- ioconfig.py
  +-- modelhandler.py
+-- models                              # PyTorch implementation of various CNN-VAE models
  +-- abhinet.py
  +-- convonlynet.py
  +-- densenet.py
  +-- resnet.py
  +-- kazunet.py
  +-- kvaenet.py
  +-- resnet.py
  +-- vaenet.py
+-- notebooks                           # Jupyter Notebooks for offline analysis
+-- plot_utils                          # Tools for visualizing model performance and dataset features
  +-- mpmt_visual.py
  +-- notebook_utils.py
  +-- plot_utils.py
+-- postprocessing                      # Tools for post-processing the outputs from the models
+-- preprocessing                       # Tools for pre-processing the dataset
  +-- merge_h5.py
  +-- merge_numpy_arrays_hdf5.py 
  +-- preprocessing_gamma.py
+-- root_utils                          # Tools for interacting with the ROOT files from the WCSim simulations
  +-- display_list.py
  +-- event_disp_and_dump.py
  +-- event_disp_and_dump_arg_utils.py
  +-- event_display.py
  +-- event_dump.py
  +-- event_dump_cedar_start.py
  +-- event_dump_one.py
  +-- pos_utils.py
+-- training_utils                      # Tools for training, validating and testing the models
  +-- doublepriorityqueue.py
  +-- engine.py
  +-- engine_vae.py
+-- README.md                           # README documentation for the repository
+-- USER_GUIDE.md                       # User guide on how to use the library
+-- watchmal.py                         # Main script to run the engine for the classifier
+-- watchmal_vae.py                     # Main script to run the engine for the VAE
```

## Installation <a id="installation"></a>

### Requirements

The following Python standard, machine learning and deep learning libraries are required for the functionality of the framework :

1. [PyTorch](https://pytorch.org/)
2. [NumPy](https://www.numpy.org/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
4. [Matplotlib](https://matplotlib.org/users/installing.html)
5. [h5py](https://www.h5py.org/)
6. [PyROOT](https://root.cern.ch/pyroot)

To download the repository use :

`git clone https://github.com/WatChMaL/VAE.git`

## Usage <a id="usage"></a>

Sample configuration files for the training engine and framework are provided in **config/engine_config**.

Examples :

```
# Train and validate a classifier using the sample configuration file

python3 watchmal.py -l test_resnet
```

```
# Train and validate a variational autoencoder using the sample configuration file

python3 watchmal_vae.py -l test_kvaenet
```

More in-depth usage options and configurations are described in **USER_GUIDE.md**.

## Credits <a id="credits"></a>

## License <a id="license"></a>
