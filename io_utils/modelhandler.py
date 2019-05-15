"""
Module for automated model selection

Author: Julian Ding
"""

import importlib
import os
import sys

MODELS_DIR = 'models'
models = importlib.import_module(MODELS_DIR)

# Prints list of all models and constructors associated with each model
def print_models():
    for name in models.__all__:
        print(name+':')
        # Suppress extraneous printing to console
        sys.stdout = open(os.devnull, 'w')
        curr_model = importlib.import_module(MODELS_DIR+'.'+name)
        sys.stdout = sys.__stdout__
        
        constructors = [x for x in dir(curr_model) if x.startswith(name)]
        for c in constructors:
            print('\t'+c)
    print('\n')
    
# Returns a model object corresponding to the specified model to load
# REQUIRES: All constructors across all models must SHARE THE SAME PARAMETERS,
#           otherwise this function completely breaks the program
def select_model(select_params):
    assert(len(select_params) == 2)
    name = select_params[0]
    constructor = select_params[1]
    # Make sure the specified model exists
    assert(name in models.__all__)
    model = importlib.import_module(MODELS_DIR+'.'+name)
    # Make sure the specified constructor exists
    assert(constructor in dir(model))
    # Return specified constructor
    return getattr(model, constructor)
    