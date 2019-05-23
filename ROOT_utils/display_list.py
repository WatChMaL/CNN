"""
Helper script to call event_display on a specified list of path, index pairs

Author: Julian Ding
"""

import subprocess

def display_list(path_idx_tuple_list, output_dir):
    for (path, idx) in path_idx_tuple_list:
        command = "event_display.py "+path+" "+output_dir+" "+idx
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)