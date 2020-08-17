"""
watchmal_cl.py

Main script to execute the training and evaluation of a fully supervised classifier
"""

# Standard python imports
from datetime import datetime

#from torchsummary import summary

# WatChMaL imports
from main.watchmal import handle_config, handle_model
from training_utils.engine_cl import EngineCL
import os, sys

# Global variables
_CL_TASKS = ['train', 'valid', 'test']

if __name__ == '__main__':
    print("PID: {}".format(os.getpid()))
    sys.stdout.flush()
    # For computing the wall clock time of execution
    start_time = datetime.now()

    # Parse the args and construct config
    config = handle_config()
    
    # Check the validity of tasks to perform
    for task in config.tasks:
        assert task in _CL_TASKS
        
    # Construct the model
    model = handle_model(config.model, config.model_params)
    # print(str(summary(model, (19,40,40), batch_size=512,device='cpu')))
    print(model)
    
    # Initialize the training engine
    engine = EngineCL(model, config)


    # Restore the model state if path given
    if config.restore_state is not None:
        engine.load_state(config.restore_state)
        
    # Do the user-specified engine tasks
    if 'train' in config.tasks:
        engine.train()
        
    if 'valid' in config.tasks:
        if engine.best_savepath is not None:
            print("Loading the best state from the training :")
            engine.load_state(engine.best_savepath)
            
        engine.validate("validation")
        
    if 'test' in config.tasks:
        if engine.best_savepath is not None:
            print("Loading the best state from the training :")
            engine.load_state(engine.best_savepath)
            
        engine.validate("test")
        
    # Print script execution time
    print("Time taken to execute the script : {0}".format(datetime.now() - start_time))
