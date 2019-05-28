"""
Helper script to call event_display on a specified list of path, index pairs

Author: Julian Ding
"""

import subprocess
#from root_utils.setup import setup

def display_list(path_idx_tuple_list, output_dir):
    #setup()
    for (path, idx) in path_idx_tuple_list:
        splits = path.split('.')
        path = splits[0] + '.root'
        command = "python2 root_utils/event_display.py "+path+" "+output_dir+" "+str(idx)
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)