"""
Engine class testing script
Author: Julian Ding
"""
import os
import sys

par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)

import training_utils.engine as engine
import models.resnet as resnet

# Preamble (will be done from parse_args() in __main__ in final script)
model = resnet.resnet18()

class CONFIG():
    pass

config = CONFIG()
config.gpu = False
config.path = '/project/6008045/machine_learning/data/IWCDmPMT/varyE/merged_IWCDmPMT_varyE.h5'
config.val_split = 0.33
config.test_split = 0.33
config.batch_size_train = 20
config.batch_size_val = 1000
config.batch_size_test = 1000
config.save_path = 'save_path'
config.data_description = 'Engine test data'

# Testing:
test_engine = engine.Engine(model, config)
test_engine.train()