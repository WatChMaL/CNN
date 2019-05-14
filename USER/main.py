"""
MUST START PROGRAM IN USER DIRECTORY FOR IMPORTS TO WORK

Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

# Make sure all custom modules can be seen by the compiler
import os
import sys

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)
    
# Let's begin...
import training_utils.engine as net
import models.resnet as resnet
import argparse
import configparser

# Global dictionary of of config arguments (key : dtype)
ARGS = {'device' : 'str',
        'gpu' : 'bool',
        'gpu_list' : 'list int',
        'path' : 'str',
        'val_split' : 'float',
        'test_split' : 'float',
        'batch_size_train' : 'int',
        'batch_size_val' : 'int',
        'batch_size_test': 'int',
        'save_path' : 'str',
        'data_description' : 'str',
        'load' : 'str',
        'cfg' : 'str'}

def loadConfig(config):
    file = config.load
    if file == None:
        return
    print('Requested load from:', file)
    print('Scanning', os.getcwd(), 'for configuration file...')
    # Get list of valid config files in current directory
    cFiles = [f for f in os.listdir(os.getcwd()) if (os.path.splitext(f)[1] == '.ini')]
    if len(cFiles) > 0:
        print('Config files found:', cFiles)
        if file in cFiles:
            parser = configparser.ConfigParser()
            parser.read(file)
            # Make sure the file has all the requisite config options
            keys = parser.items('config')
            assert(set([x[0] for x in keys]).issubset(set([k for k in ARGS])))            
            print('Loading from', file)
            for (item, option) in keys:
                argtype = ARGS[item].split()[0]
                option = parser.get('config', item)
                if argtype == 'str':
                    setattr(config, item, option)
                elif argtype == 'int':
                    setattr(config, item, int(option))
                elif argtype == 'float':
                    setattr(config, item, float(option))
                elif argtype == 'bool':
                    setattr(config, item, (option == 'True'))
                elif argtype == 'list':
                    listtype = ARGS[item].split()[1]
                    # Note: int is the only list type at the moment
                    if listtype == 'int' and option is not None:
                        print(option)
                        print(type(option))
                        setattr(config, item, option)
                else:
                    print(option, 'is not a valid config option, ignoring...')
            print(file, 'loaded!')
        else:
            print(file, 'not found, aborting.')
            return
    else:
        print('No config files found.')
        return
    
def saveConfig(config):
    outFile = config.cfg
    if outFile is None:
        return
    print('Saving config file as', outFile)
    conf = configparser.ConfigParser()
    conf.add_section('config')
    # Store all config attributes in ConfigParser
    for x in dir(config):
        if not (x.startswith('_') or x.startswith('__')):
            item = vars(config)[x]
            if type(item) == list:
                listStr = ''
                for t in item:
                    listStr += ' ' + str(t)
                item = listStr
            conf.set('config', str(x), str(item))
    with open(outFile, 'w+') as configFile:
        conf.write(configFile)
    print('Config file saved in', os.getcwd())

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-device', dest='device', default='cpu',
                        required=False, help='Enter cpu to use CPU resources or gpu to use GPU resources.')
    parser.add_argument('-gpu', nargs='+', dest='gpu_list', default=None,
                        required=False, help='List of available GPUs')
    parser.add_argument('-path', dest='path', default='',
                        required=False, help='Path to training dataset.')
    parser.add_argument('-vs', dest='val_split', default=0.1,
                        required=False, help='Fraction of dataset used in validation. Note: requires vs + ts < 1')
    parser.add_argument('-ts', dest='test_split', default=0.1,
                        required=False, help='Fraction of dataset used in testing. Note: requires vs + ts < 1')
    parser.add_argument('-tnb', dest='batch_size_train', default=20,
                        required=False, help='Batch size for training.')
    parser.add_argument('-vlb', dest='batch_size_val', default=1000,
                        required=False, help='Batch size for validating.')
    parser.add_argument('-tsb', dest='batch_size_test', default=1000,
                        required=False, help='Batch size for testing.')
    parser.add_argument('-save', dest='save_path', default='save_path',
                        required=False, help='Specify path to save data to. Default is save_path')
    parser.add_argument('-desc', dest='data_description', default='DATA DESCRIPTION',
                        required=False, help='Specify description for data.')
    parser.add_argument('-l', dest='load', default=None,
                        required=False, help='Specify config file to load from. No action by default.')
    parser.add_argument('-s', dest='cfg', default=None,
                        required=False, help='Specify name for destination config file. No action by default.')
    
    config = parser.parse_args()

    print(config.device)
    print(config.gpu_list)
    
    if config.load is not None:
    	config.gpu_list = [int(x) for x in config.gpu_list]
    	if config.load is not None and not config.load.endswith('.ini'):
        	config.load += '.ini'
    	if config.cfg is not None and not config.cfg.endswith('.ini'):
        	config.cfg += '.ini'
        
    # Assertions to ensure flags follow rules:
    assert(config.device == 'cpu' or config.device == 'gpu')
    assert(config.val_split + config.test_split < 1)
    #assert(os.path.exists(config.save_path))
        
    return config

if __name__ == '__main__':
    print('[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WaTChMaL)')
    print('Collaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n')
    config = main()
    loadConfig(config)
    saveConfig(config)
    model = resnet.resnet18(num_input_channels=38, num_classes=3)
    nnet = net.Engine(model, config)
    nnet.restore_state("state2400")
    nnet.train(epochs=10.0, save_interval=1000)
    nnet.validate()
