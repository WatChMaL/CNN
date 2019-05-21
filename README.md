# Water Cherenkov Machine Learning (WatChMaL) - Convolutional Neural Network (CNN)
Codebase for training and performance evaluation of CNNs using simulated neutrino weak interaction event data formatted as an image.

## User Guide
To start the program, download the repository and navigate to the parent folder, `CNN/`, then enter on terminal/cmd
```python3 watchmal.py #flags and arguments```
There is an extensive list of flags which can be used to tune the training engine, detailed below. Every flag has valid default behaviour and thus none of the flags need to be specified to run the program; however, the data path default `.` is probably invalid for any particular case.
### Setup
- `-h` prints out the help dialogue for all flags onto the terminal window. There is no config option for this flag.
- `-m #name #constructor'` specifies an architecture to train on. Make sure the selected architecture exists in `models/`. A list of available architectures is printed on the terminal for convenience. The config option for this flag is `model`.
- `-pms #space-delimited list of named arguments` specifies a list of arguments to pass to the CNN constructor. Make sure the arguments are valid for the selected constructor. A list of arguments taken by each constructor is printed on the terminal for convenience. The config option for this flag is `params`.
- `-dev #cpu/gpu` sets the engine to offload work to the CPU or GPU. If GPU is selected, you must also specify a list of GPUs. The config option for this flag is `device`.
- `-gpu #space-delimited list of gpus (ints)` gives the engine a list of GPUs to train on. If no GPUs are given, the training engine defaults to running on the CPU. The config option for this flag is `gpu_list`.
- `-do #train #test #val` instructs the engine to run training, testing, and validation tasks. The engine can run any subset of these tasks and runs them all by default. The config option for this flag is `tasks`.

### Data Handling
- `-pat #path` specifies the path to the labeled dataset which the engine will train, test, and validate on. HDF5 is the only supported data format at the moment. The config option for this flag is `path`.
- `-sub #integer` specifies a subset of the dataset located at `path` to use, which can be useful for making faster training runs. By default, all of the data is used. The config option for this flag is `subset`.
- `-vas #float between 0 and 1` specifies the fraction of the dataset to use for validation. By default this is set to `0.1`. The config option for this flag is `val_split`.
- `-tes #float between 0 and 1` specifies the fraction of the dataset to use for testing. By default this is set to `0.1`. The config option for this flag is `test_split`.
- There is no option to specify the fraction of the dataset to use for training. This fraction is the remainder of the dataset that is outside the validation and test splits (i.e. `train_split = 1 - val_split - test_split`).
- `-epo #float` specifies the number of epochs to train the data over. This number does not have to be a whole number. By default this is set to `1.0`. The config option for this flag is `epochs`.
- `-tnb #integer` specifies the batch size during training. By default this is set to `20`. The config option for this flag is `batch_size_train`.
- `-vlb #integer` specifies the batch size during validation. By default this is set to `1000`. The config option for this flag is `batch_size_val`.
- `-tsb #integer` specifies the batch size during testing. By default this is set to `1000`. The config option for this flag is `batch_size_test`.
- Note: the batch size should never exceed the dataset size.
- `-sap #path` specifies the directory into which to save the training engine output data. This directory will be located inside `USER/` and has a default name of `save_path`. The config option for this flag is `save_path`.
- `-dsc #description` specifies a subdirectory under `save_path` to save data from a particular run. By default this is set to `data_description`. The config option for this flag is `data_description`.
- `-ret #state file` specifies the path to a state file from which to restore the weights in the neural net. By default the state is not loaded. The config option for this flag is `restore_state`.

### Config File Management
- `-l #config file` specifies a config file to load settings from. By default no config file is loaded and settings are interpreted from the specified flags. If this flag is specified but other flags conflict with the settings in the conflict file, the flags given on the commandline will override the respective settings in the config file. The config option for this flag is `load`.
- `-s #config file` specifies the name of a config file to save settings to (overwrite enabled). By default no config file is saved. The config option for this flag is `cfg`.
Note that you can manually write a config file and load it with the `-l` flag as an alternative to using commandline flags. The syntax for the config file is
```
[config]
option1 = string1
option2 = string2
...
```
By default the config file extension is `.ini`.