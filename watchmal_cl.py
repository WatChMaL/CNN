"""
watchmal_cl.py

Main script to execute the training and evaluation of a fully supervised classifier
"""

# Standard python imports
from datetime import datetime

# WatChMaL imports
from main.watchmal import handle_config, handle_model
from training_utils.engine_cl import EngineCL

# Global variables
_CL_TASKS = ['train', 'valid', 'test']

if __name__ == '__main__':
    
    # For computing the wall clock time of execution
    start_time = datetime.now()

    # Parse the args and construct config
    config = handle_config()
    
    # Check the validity of tasks to perform
    for task in config.tasks:
        assert task in _CL_TASKS
        
    # Construct the model
    model = handle_model(config.model, config.params)
    
    # Initialize the training engine
    engine = EngineCL(model, config)
    
    # Restore the model state if path given
    if config.restore_state != None:
        engine.load_state(config.restore_state)
        
    # Do the user-specified engine tasks
    if 'train' in config.tasks:
        engine.train(config.epochs, config.report_interval, config.num_vals, config.num_val_batches)
        
    if 'valid' in config.tasks:
        engine.validate("validation", config.num_dump_events)
        
    if 'test' in config.tasks:
        engine.validate("test", config.num_dump_events)
        
    # Print script execution time
    print("Time taken to execute the script : {0}".format(datetime.now() - start_time))