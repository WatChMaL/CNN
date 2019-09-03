"""
watchmal_vae.py

Main script to pass the command-line arguments and run training_utils/engine_vae.py

Author : Abhishek .

Note : Parts of the source code borrowed from WatChMaL/CNN/watchmal.py
"""

"""
WELCOME TO WatChMaL, USER

START PROGRAM HERE

watchmal.py: Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

import os
from datetime import datetime

import training_utils.engine_vae as net
import io_utils.arghandler as arghandler
import io_utils.ioconfig as ioconfig
import io_utils.modelhandler as modelhandler
import torch

# Global list of arguments to request from commandline
ARGS = [arghandler.Argument('model', list, list_dtype=str, flag='-m',
                            default=['resnet', 'resnet18'], help='Specify neural net architecture. Default is resnet18.'),
        arghandler.Argument('params', list, list_dtype=str, flag='-pms',
                            default=['num_input_channels=38', 'num_classes=3'], help='Specify args to pass to neural net constructor.'),
        arghandler.Argument('device', str, '-dev',
                            default='cpu', help='Enter cpu to use CPU resources or gpu to use GPU resources.'),
        arghandler.Argument('gpu_list', list, list_dtype=int, flag='-gpu',
                            help='List of available GPUs.'),
        arghandler.Argument('path', str, '-pat',
                            default='', help='Path to training dataset.'),
        arghandler.Argument('root', str, '-roo',
                            default=None, help='Path to ROOT file list directory if outside data directory.'),
        arghandler.Argument('subset', int, '-sub',
                            default=None, help='Number of data from training set to use.'),
        arghandler.Argument('shuffle', bool, '-shf',
                            default=True, help='Specify whether or not to shuffle training dataset. Default is True.'),
        arghandler.Argument('cl_train_split', float, '-clt',
                            default=0.2, help='Fraction of dataset used in classifier validation.'),
        arghandler.Argument('cl_val_split', float, '-clv',
                            default=0.1, help='Fraction of dataset used in classifier validation.'),
        arghandler.Argument('vae_val_split', float, '-vav',
                            default=0.1, help='Fraction of dataset used in VAE validation.'),
        arghandler.Argument('test_split', float, '-tes',
                            default=0.1, help='Fraction of dataset used in testing for both the classifier and VAE.'),
        arghandler.Argument('epochs', float, '-epo',
                            default=1.0, help='Number of training epochs to run.'),
        arghandler.Argument('batch_size_train', int, '-tnb',
                            default=20, help='Batch size for training.'),
        arghandler.Argument('batch_size_val', int, '-vlb',
                            default=1000, help='Batch size for validation.'),
        arghandler.Argument('batch_size_test', int, '-tsb',
                            default=1000, help='Batch size for testing.'),
        arghandler.Argument('tasks',list, list_dtype=str, flag='-do',
                            default=['train', 'test', 'valid'], help='Specify list of tasks: "train" = run training; "test" = run testing; "valid" = run validation. Default behaviour runs all tasks.'),
        arghandler.Argument('worst', int, flag='-wst',
                            default=0, help='Specify the number of WORST-identified events to dump root file references to at the end of validation.'),
        arghandler.Argument('best', int, flag='-bst',
                            default=0, help='Specify the number of BEST-identified events to dump root file references to at the end of validation.'),
        arghandler.Argument('dump_path', str, '-dmp',
                            default='dumps', help='Specify path to dump data to. Default is dumps.'),
        arghandler.Argument('load', str, '-l',
                            default=None, help='Specify config file to load from. No action by default.'),
        arghandler.Argument('restore_state', str, '-ret',
                            default=None, help='Specify a state file to restore the neural net to the training state from a previous run. No action by default'),
        arghandler.Argument('cfg', str, '-s',
                            default=None, help='Specify name for destination config file. No action by default.'),
        arghandler.Argument('githash', str, '-git',
                            default=None, help='git-hash for the latest commit'),
        arghandler.Argument('num_samples', int, '-nsm',
                            default=64, help='Number of samples to generate from the VAE. Only works is model.variant is VAE.')]

ATTR_DICT = {arg.name : ioconfig.ConfigAttr(arg.name, arg.dtype,
                                            list_dtype = arg.list_dtype if hasattr(arg, 'list_dtype') else None) for arg in ARGS}

if __name__ == '__main__':
    
    # Find script execution runtime
    start_time = datetime.now()
    
    # Intro message :D
    print("""[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WatChMaL)
    \tCollaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n""")
    
    # Reflect available models
    print('CURRENT AVAILABLE ARCHITECTURES')
    config = arghandler.parse_args(ARGS)
    
    # Do not overwrite any attributes specified by commandline flags
    for ar in ARGS:
        if getattr(config, ar.name) != ar.default:
            ATTR_DICT[ar.name].overwrite = False
            
    # Load from file
    if config.load is not None:
        ioconfig.loadConfig(config, config.load, ATTR_DICT)
        config.cfg = None
        
    # Check attributes for validity
    for task in config.tasks:
        assert(task in ['train', 'test', 'valid', 'sample', 'generate', 'interpolate'])
        
    # Add the git-hash from the latest commit to the config
    git_hash = os.popen("git rev-parse HEAD").read()
    config.githash = git_hash[:len(git_hash)-1]
    
    # Save to file
    if config.cfg is not None:
        ioconfig.saveConfig(config, config.cfg)
        
    # Set save directory to under USER_DIR
    config.dump_path = config.dump_path+('' if config.dump_path.endswith('/') else '/')
        
    # Select requested model
    print('Selected architecture:', config.model)
    
    # Make sure the specified arguments can be passed to the model
    params = ioconfig.to_kwargs(config.params)
    modelhandler.check_params(config.model[0], params)
    constructor = modelhandler.select_model(config.model)
    model = constructor(**params)
    
    """
    # Plot the computational graph of the model
    model_dot = make_dot(model, params=dict(model.named_parameters()))
    print(type(model_dot))
    """
    
    # Finally, construct the neural net
    nnet = net.EngineVAE(model, config, model.variant, model.train_type)

    # Do some work...
    if config.restore_state is not None:
        nnet.restore_state(config.restore_state)
           
    if 'sample' in config.tasks:
        nnet.sample(num_samples=config.num_samples, trained=False)
    if 'generate' in config.tasks:
        print("Generating pre-training latent vectors")
        nnet.generate_latent_vectors("pre")
    if 'train' in config.tasks:
        print("Number of epochs :", config.epochs)
        nnet.train(epochs=config.epochs)
    if 'generate' in config.tasks:
        print("Generating post-training latent vectors")
        nnet.generate_latent_vectors("post")
    if 'test' in config.tasks:
        nnet.validate(subset="test")
    if 'valid' in config.tasks:
        nnet.validate(subset="train")
        nnet.validate(subset="validation")
    if 'sample' in config.tasks:
        nnet.sample(num_samples=config.num_samples, trained=True)
    if 'interpolate' in config.tasks:
        nnet.interpolate(subset="test", event_type="mu", angle_1=0, energy_1=800,
                         angle_2=-1.57, energy_2=800, intervals=10, num_neighbors=1024,
                         trained=True)
        
        
    # Print script execution time
    print("Time taken to execute the script : {0}".format(datetime.now() - start_time))