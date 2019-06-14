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
+-- config
  +-- engine_config
    +-- test_resnet.ini
    +-- test_kazunet.ini
    +-- test_kvaenet.ini
  +-- preprocessing_config
    +-- merge_config.ini
+-- io_uitls
  +-- __init__.py
  +-- arghandler.py
  +-- data_handling.py
  +-- ioconfig.py
  +-- modelhandler.py
+-- models
  +-- abhinet.py
  +-- convonlynet.py
  +-- densenet.py
  +-- resnet.py
  +-- kazunet.py
  +-- kvaenet.py
  +-- resnet.py
  +-- vaenet.py
+-- notebooks
+-- plot_utils
  +-- mpmt_visual.py
  +-- notebook_utils.py
  +-- plot_utils.py
+-- postprocessing
+-- preprocessing
  +-- merge_h5.py
  +-- merge_numpy_arrays_hdf5.py
  +-- preprocessing_gamma.py
+-- root_utils
  +-- display_list.py
  +-- event_disp_and_dump.py
  +-- event_disp_and_dump_arg_utils.py
  +-- event_display.py
  +-- event_dump.py
  +-- event_dump_cedar_start.py
  +-- event_dump_one.py
  +-- pos_utils.py
+-- training_utils
  +-- doublepriorityqueue.py
  +-- engine.py
  +-- engine_vae.py
+-- README.md
+-- USER_GUIDE.md
+-- watchmal.py
+-- watchmal_vae.py
```

## Installation <a id="installation"></a>

### Requirements

## Usage <a id="usage"></a>

## Credits <a id="credits"></a>

## License <a id="license"></a>
