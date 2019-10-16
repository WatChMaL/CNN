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
import torchviz

# Global list of arguments to request from commandline
ARGS = [arghandler.Argument('model', list, '-mdl', list_dtype=str,
                            default=['resnet', 'resnet18'],
                            help='Specify model architecture. Default = resnet resnet18.'),
        arghandler.Argument('params', list, '-pms', list_dtype=str,
                            default=None,
                            help='Soecify model constructer args. Default = None'),
        arghandler.Argument('device', str, '-dev',
                            default='cpu',
                            help='Choose either \'cpu\' or \'gpu\'. Default = \'cpu\''),
        arghandler.Argument('gpu_list', list, '-gpu', list_dtype=int,
                            default=0,
                            help='Indices of the device GPUs to utilize. E.g. 0 1. Default = 0.'),
        arghandler.Argument('path', str, '-pth',
                            default=None,
                            help='Absolute path for the training dataset. Default = None'),
        arghandler.Argument('subset', int, '-sbt',
                            default=None, 
                            help='Number of samples to use from the entire dataset. Default = None'),
        arghandler.Argument('shuffle', bool, '-shf',
                            default=True, 
                            help='Specify whether or not to shuffle the dataset. Default = True.'),
        arghandler.Argument('cl_ratio', float, '-clr',
                            default=0.1,
                            help='Fraction of dataset to be used for classifier training and validation. Default = 0.1'),
        arghandler.Argument('val_split', float, '-vst',
                            default=0.1, 
                            help='Fraction of entire dataset to be used for validation. Default = 0.1'),
        arghandler.Argument('test_split', float, '-tst',
                            default=0.1,
                            help='Fraction of dataset to be used for testing. Default = 0.1'),
        arghandler.Argument('epochs', float, '-epc',
                            default=10.0,
                            help='Number of training epochs. Default=10.0'),
        arghandler.Argument('batch_size_train', int, '-bstn',
                            default=128, help='Training dataset batch size. Default=128.'),
        arghandler.Argument('batch_size_val', int, '-bsvl',
                            default=1000, help='Batch size for validation.'),
        arghandler.Argument('batch_size_test', int, '-bstt',
                            default=1000, help='Batch size for testing.'),
        arghandler.Argument('tasks', list, list_dtype=str, flag='-d',
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



if __name__ == '__main__':
    
    # Find script execution runtime
    start_time = datetime.now()
    
    # Intro message :D
    print("""[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WatChMaL)
          \tCollaborators: Wojciech Fedorko, Julian Ding, Abhishek Abhishek\n""")
    
    # Reflect available models
    print("Current available architectures")
    config = arghandler.parse_args(ARGS)
    
    # Do not overwrite any attributes specified by commandline flags
    for ar in ARGS:
        if getattr(config, ar.name) != ar.default:
            ATTR_DICT[ar.name].overwrite = False
            
    # Load from file
    if config.load != None:
        ioconfig.loadConfig(config, config.load, ATTR_DICT)
        config.cfg = None
        
    # Check attributes for validity
    for task in config.tasks:
        assert(task in ['train', 'test', 'valid', 'sample', 'generate', 'interpolate', 'train_ssl'])
        
    # Add the git-hash from the latest commit to the config
    git_hash = os.popen("git rev-parse HEAD").read()
    config.githash = git_hash[:len(git_hash)-1]
    
    # Save to file
    if config.cfg != None:
        ioconfig.saveConfig(config, config.cfg)
        
    # Set save directory to under USER_DIR
    config.dump_path = config.dump_path+('' if config.dump_path.endswith('/') else '/')
        
    # Select requested model
    print('Selected architecture : ', config.model)
     
    # Make sure the specified arguments can be passed to the model
    params = ioconfig.to_kwargs(config.params)
    modelhandler.check_params(config.model[0], params)
    constructor = modelhandler.select_model(config.model)
    model = constructor(**params)
    
    # Finally, construct the neural net
    nnet = net.EngineVAE(model, config)

    # Do some work...
    if config.restore_state != None:
        nnet.restore_state(mode="default")
    
    if 'train' in config.tasks:
        print("Number of epochs :", config.epochs)
        nnet.train(epochs=config.epochs)
        
    if 'train_ssl' in config.tasks:
        print("Number of epochs :", config.epochs)
        nnet.train_ssl(epochs=config.epochs, report_interval=50,
                       num_validations=10000)
        
    if 'generate' in config.tasks:
        print("Generating pre-training latent vectors")
        nnet.generate_latent_vectors("post")
        
    if 'test' in config.tasks:
        if config.restore_state != None:
            nnet.restore_state(mode="default")
        else:
            nnet.restore_state(mode="best")
        nnet.validate(subset="test", num_dump_events=config.num_samples)
        
    if 'valid' in config.tasks:
        if config.restore_state != None:
            nnet.restore_state(mode="default")
        else:
            nnet.restore_state(mode="best")
        nnet.validate(subset="validation", num_dump_events=config.num_samples)
        
    if 'sample' in config.tasks:
        nnet.sample(num_samples=config.num_samples, trained=True)
        
    if 'interpolate' in config.tasks:
        nnet.interpolate(subset="test", event_type="mu", angle_1=-3.14, energy_1=200,
                         angle_2=-3.14, energy_2=800, intervals=5, num_neighbors=256,
                         trained=True)
        
        
    # Print script execution time
    print("Time taken to execute the script : {0}".format(datetime.now() - start_time))