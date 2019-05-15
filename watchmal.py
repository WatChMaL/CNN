"""
WELCOME TO WatChMaL, user

START PROGRAM HERE

watchmal.py: Script to pass commandline arguments from user to neural net framework.

Author: Julian Ding
"""

# TODO: Reduced dataset, specify epochs, training/validation (or both)

import training_utils.engine as net
import models.resnet as resnet
import iotools.arghandler as arghandler
import iotools.ioconfig as ioconfig

# Global list of arguments to request from commandline
ARGS = [arghandler.Argument('device', str, '-dev',
                            default='cpu', help='Enter cpu to use CPU resources or gpu to use GPU resources.'),
        arghandler.Argument('gpu_list', list, list_dtype=int, flag='-gpu',
                            help='List of available GPUs.'),
        arghandler.Argument('path', str, '-pat',
                            default='.', help='Path to training dataset.'),
        arghandler.Argument('subset', float, '-sub',
                            default=1, help='Fraction of training dataset to use.'),
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
        arghandler.Argument('save_path', str, '-sap',
                            default='save_path', help='Specify path to save data to. Default is save_path.'),
        arghandler.Argument('data_description', str, '-dsc',
                            default='data_description', help='Specify description for data/name for data subdirectory.'),
        arghandler.Argument('load', str, '-l',
                            default=None, help='Specify config file to load from. No action by default.'),
        arghandler.Argument('cfg', str, '-s',
                            default=None, help='Specify name for destination config file. No action by default.')]

ATTR_DICT = {arg.name : ioconfig.ConfigAttr(arg.name, arg.dtype,
                                            list_dtype = arg.list_dtype if hasattr(arg, 'list_dtype') else None) for arg in ARGS}

if __name__ == '__main__':
    print("""[HK-Canada] TRIUMF Neutrino Group: Water Cherenkov Machine Learning (WatChMaL)
\tCollaborators: Wojciech Fedorko, Julian Ding, Abhishek Kajal\n""")
    config = arghandler.parse_args(ARGS)
    # Do not overwrite any attributes specified by command line flags
    for ar in ARGS:
        if getattr(config, ar.name) != ar.default:
            ATTR_DICT[ar.name].overwrite = False
    if config.load is not None:
        ioconfig.loadConfig(config, config.load, ATTR_DICT)
    if config.cfg is not None:
        ioconfig.saveConfig(config, config.cfg)
        
    model = resnet.resnet18(num_input_channels=38, num_classes=3)
    nnet = net.Engine(model, config)
    nnet.restore_state("state2400")
    nnet.train(epochs=config.epochs, save_interval=1000)
    nnet.validate()
