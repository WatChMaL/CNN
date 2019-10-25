"""
watchmal_vae.py

Main script to execute the training and evaluation of a fully supervised classifier
"""

# Standard python imports
from datetime import datetime

# WatChMaL imports
from main.watchmal import handle_config, handle_model
from training_utils.engine_vae import EngineVAE
from io_utils.ioconfig import to_kwargs

# Global variables
_VAE_TASKS = ['train', 'valid', 'test', 'interpolate', 'sample']

if __name__ == '__main__':
    
    # For computing the wall clock time of execution
    start_time = datetime.now()

    # Parse the args and construct config
    config = handle_config()
    
    # Check the validity of tasks to perform
    for task in config.tasks:
        assert task in _VAE_TASKS
        
    # Construct the model
    model = handle_model(config.model, config.model_params)
    
    # Initialize the training engine
    engine = EngineVAE(model, config)
    
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
        
    if 'sample' in config.tasks:
        engine.sample(config.num_dump_events)
        
    if 'interpolate' in config.tasks:
        engine.interpolate()
        
    # Print script execution time
    print("Time taken to execute the script : {0}".format(datetime.now() - start_time))