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
        parser.add_argument(arg.flag, nargs = '+' if arg.dtype == list else 1, dest=arg.name,
                            default=arg.default, required=arg.required, help=arg.help)
    config = parser.parse_args()
    # Cast config attributes to correct data types
    for arg in args_list:
        curr = getattr(config, arg.name)
        if curr != arg.default and curr is not None:
            if arg.dtype != list:
                if type(curr) == list:
                    curr = curr[0]
                setattr(config, arg.name, arg.dtype(curr))
            else:
                attrlist = [arg.list_dtype(x) for x in curr]
                setattr(config, arg.name, attrlist)
    
    return config