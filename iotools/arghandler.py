"""
Module for commandline argument handling functionality

Author: Julian Ding
"""

import argparse

# Class to encapsulate all the attributes of an argument
class Argument():
    def __init__(self, name, dtype, flag,
                 list_dtype=None, default=None, required=False, help='default_help_string'):
        self.name = name
        self.dtype = dtype
        self.list_dtype= list_dtype
        self.flag = flag
        self.default = default
        self.required = required
        self.help = help
        
# Given a list of Argument objects, generate a config argparse.Namespace object
def parse_args(args_list):
    parser = argparse.ArgumentParser()
    for arg in args_list:
        parser.add_argument(arg.flag, dest=arg.name, default=arg.default,
                            required=arg.required, help=arg.help)
    return parser.parse_args()