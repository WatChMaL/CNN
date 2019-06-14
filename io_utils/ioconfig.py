"""
Module for config file loading/saving functionality

Author: Julian Ding
"""

import os
import configparser

# User directory name
CONFIG_DIR = 'config/engine_config/'

# Config file type
CFG_EXT = '.ini'

# Delimiters
DELIM = ' '
ARG_DELIM = '='
NAME_DELIM = '_'

# Class to encapsulate the necessary properties of a config object attribute
class ConfigAttr():
    def __init__(self, name, dtype, list_dtype=None, overwrite=True):
        self.name = name
        self.dtype = dtype
        self.list_dtype = list_dtype
        self.overwrite = overwrite
        
# Helper function to add an attribute to a config object
def add_attr(config, name, data_str, dtype, list_dtype=None):
    if data_str == 'None':
        setattr(config, name, None)
    elif dtype != list:
        setattr(config, name, dtype(data_str))
    elif list_dtype is not None:
        attrlist = [list_dtype(x) for x in data_str.split(DELIM)]
        setattr(config, name, attrlist)
    else:
        print('Load error encountered when parsing', data_str, 'as', dtype)

# Loads configuration from a file in CONFIG_DIR
def loadConfig(config, inFile, attr_dict):
    if not inFile.endswith(CFG_EXT):
        inFile += CFG_EXT
    print('Requested load from:', inFile)
    # Get list of valid config files in current directory
    cFiles = [f for f in os.listdir(CONFIG_DIR) if (os.path.splitext(f)[1] == '.ini')]
    if len(cFiles) > 0:
        print('Config files found:', cFiles)
        if inFile in cFiles:
            parser = configparser.ConfigParser()
            parser.read(CONFIG_DIR + inFile)
            keys = parser.items('config')
            print('Loading from', inFile)
            for (item, data_str) in keys:
                if item in attr_dict:
                    curr = attr_dict[item]
                    if curr.overwrite:
                        # If the item is a requested config attribute, parse string
                        add_attr(config, curr.name, data_str, curr.dtype,
                                 list_dtype=curr.list_dtype)
            print(inFile, 'loaded!')
        else:
            print(inFile, 'not found, aborting.')
    else:
        print('No config files found, aborting.')

# Saves config object to configuration file in CONFIG_DIR
def saveConfig(config, outFile):
    if not outFile.endswith(CFG_EXT):
        outFile += CFG_EXT
    print('Saving config file as', outFile)
    conf = configparser.ConfigParser()
    conf.add_section('config')
    # Store all config attributes in ConfigParser
    for x in dir(config):
        if not x.startswith('_'):
            item = vars(config)[x]
            if type(item) == list:
                listStr = ''
                for t in item:
                    listStr += DELIM + str(t)
                item = listStr
            conf.set('config', str(x), str(item))
    # Do not overwrite existing config files
    while os.path.isfile(outFile):
        n_outFile = outFile.split('.')[0]
        name_split = n_outFile.split(NAME_DELIM)
        n_outFile = name_split[0]+NAME_DELIM+('0' if len(name_split) == 1 else str(int(name_split[1])+1))+CFG_EXT
        print('Config file with name', outFile, 'already exists. Saving to:', n_outFile)
        outFile = n_outFile
    with open(outFile, 'w') as configFile:
        conf.write(configFile)
    print('Config file saved at', outFile)
    
# Function to convert a list of strings into a kwargs-interpretable dict
def to_kwargs(arglist):
    args = [arg.split(ARG_DELIM) for arg in arglist]
    return {x[0] : int(x[1]) for x in args}