"""
WELCOME TO WatChMaL, USER

START PROGRAM HERE

watchmal.py: Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

# TODO: Reduced dataset, specify epochs, training/validation (or both)

import training_utils.engine as net
import io_utils.arghandler as arghandler
import io_utils.ioconfig as ioconfig
import io_utils.modelhandler as modelhandler

# Global list of arguments to request from commandline
ARGS = [arghandler.Argument('model', list, list_dtype=str, flag='-m',
                            default=['resnet', 'resnet18'], help='Specify neural net architecture. Default is resnet18.'),
        arghandler.Argument('device', str, '-dev',
                            default='cpu', help='Enter cpu to use CPU resources or gpu to use GPU resources.'),
        arghandler.Argument('gpu_list', list, list_dtype=int, flag='-gpu',
                            help='List of available GPUs.'),
        arghandler.Argument('path', str, '-pat',
                            default='.', help='Path to training dataset.'),
        arghandler.Argument('subset', int, '-sub',
                            default=None, help='Number of data from training set to use.'),
        arghandler.Argument('val_split', float, '-vas',
                            default=0.1, help='Fraction of dataset used in validation.'),
        arghandler.Argument('test_split', float, '-tes',
                            default=0.1, help='Fraction of dataset used in testing. (Note: remaining fraction is used in training)'),
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
        arghandler.Argument('save_path', str, '-sap',
                            default='save_path', help='Specify path to save data to. Default is save_path.'),
        arghandler.Argument('data_description', str, '-dsc',
                            default='data_description', help='Specify description for data/name for data subdirectory.'),
        arghandler.Argument('load', str, '-l',
                            default=None, help='Specify config file to load from. No action by default.'),
        arghandler.Argument('restore_state', str, '-ret',
                            default=None, help='Specify a state file to restore the neural net to the training state from a previous run. No action by default'),
        arghandler.Argument('cfg', str, '-s',
                            default=None, help='Specify name for destination config file. No action by default.')]

ATTR_DICT = {arg.name : ioconfig.ConfigAttr(arg.name, arg.dtype,
                                            list_dtype = arg.list_dtype if hasattr(arg, 'list_dtype') else None) for arg in ARGS}

if __name__ == '__main__':
    # Intro message :)
    print("""[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WatChMaL)
\tCollaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n""")
    # Reflect available models
    print('CURRENT AVAILABLE ARCHITECTURES')
    modelhandler.print_models()
    config = arghandler.parse_args(ARGS)
    # Do not overwrite any attributes specified by commandline flags
    for ar in ARGS:
        if getattr(config, ar.name) != ar.default:
            ATTR_DICT[ar.name].overwrite = False
    # Load from file
    if config.load is not None:
        ioconfig.loadConfig(config, config.load, ATTR_DICT)
    # Check attributes for validity
    for task in config.tasks:
        assert(task in ['train', 'test', 'valid'])
    # Save to file
    if config.cfg is not None:
        ioconfig.saveConfig(config, config.cfg)
    # Select requested model
    print('Selected architecture:', config.model)
    constructor = modelhandler.select_model(config.model)
    model = constructor(num_input_channels=38, num_classes=3)
    # Finally, construct the neural net
    nnet = net.Engine(model, config)
    # Do some work...
    if config.restore_state is not None:
        nnet.restore_state(config.restore_state)
    if 'train' in config.tasks:
        nnet.train(epochs=config.epochs, save_interval=1000)
    if 'test' in config.tasks:
        nnet.test()
    if 'valid' in config.tasks:
        nnet.validate()
