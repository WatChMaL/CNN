from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='dump WCSim data into numpy and/or create event display figures')
    parser.add_argument('input_file', type=str, nargs=1)
    parser.add_argument('output_file', type=str, nargs='?',default=None)
    parser.add_argument('--n_events_to_display', type=int, default=0)
    
    args = parser.parse_args()
    return args
