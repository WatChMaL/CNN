"""
Script for setting up computecanada workspace for working with PyRoot

"""

import os,sys

commands = ["source /project/rpp-tanaka-ab/hk_software/nuPRISM/sourceme.sh",
            "module load python/2.7.14",
            "module load scipy-stack",
            "export PYTHONPATH=$ROOTSYS/../bindings/pyroot:$PYTHONPATH"]

def setup(start_args):
    for command in commands:
        os.system(command)
    dump_in = ''
    for arg in start_args:
        dump_in+=' '+arg
    os.system("python event_dump.py"+dump_in)
        
if __name__ == '__main__':
    setup(sys.argv[1:])