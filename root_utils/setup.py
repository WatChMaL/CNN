"""
Script for setting up computecanada workspace for working with PyRoot

"""

import subprocess

commands = ["source /project/rpp-tanaka-ab/hk_software/nuPRISM/sourceme.sh",
            "module load python/2.7.14",
            "module load scipy-stack",
            "export PYTHONPATH=$ROOTSYS/../bindings/pyroot:$PYTHONPATH"]

def setup():
    for command in commands:
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)