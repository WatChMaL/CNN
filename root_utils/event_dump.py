"""
Python 2 script for processing ROOT files into .npz files

Adapted from event_disp_and_dump by Wojciech Fedorko

To keep references to the original ROOT files, a list of 
absolute file paths is dumped in a text file in the output
directory.

Two indices are saved for every event in the output npz file:
one corresponding to the position of the ROOT file path in the
output dump file (ROOT.txt) and the other corresponding to the
event index within that ROOT file (ev).

Author: Julian Ding
"""

import numpy as np
from pos_utils import *

import ROOT
ROOT.gROOT.SetBatch(True)

import os, sys
from argparse import ArgumentParser

ROOT_DUMP = 'ROOTS.txt'

def get_args():
    parser = ArgumentParser(description='dump WCSim data into numpy .npz file')
    parser.add_argument('input_dir', type=str, nargs=1)
    parser.add_argument('output_dir', type=str, nargs=1)
    args = parser.parse_args()
    return args


def event_dump(config):
    config.input_dir = config.input_dir[0]
    config.output_dir = config.output_dir[0]
    config.input_dir += ('' if config.input_dir.endswith('/') else '/')
    config.output_dir += ('' if config.output_dir.endswith('/') else '/')
    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)
        
    # Create dump file here
    PATH_FILE = open(config.output_dir+ROOT_DUMP, 'ab+') # THIS IS HARD-CODED, MUST CORRESPOND IN event_display.py
    existing_paths = {} # Dictionary of abspath:index pairs
    for i, line in PATH_FILE.readlines():
        line = line.strip()
        existing_path[line] = i
    num_existing_paths = len(existing_paths.keys())
    
    files = [f for f in os.listdir(config.input_dir)
    if f.endswith('.root') and '_R0cm_' in f and not f.split('.')[0].endswith('_flat')]
    # This list is for input into merge_numpy_arrays_hdf5.py
    output_list = open(config.output_dir+'list.txt', 'a+')
    
    print "input directory: "+str(config.input_dir)
    print "input files ("+str(len(files))+"): "+str(files)
    print "output directory: "+str(config.output_dir)
    
    for input_file in files:
        
        print "\nNow processing "+input_file
        
        file_dir = config.input_dir+input_file
        file=ROOT.TFile(file_dir,"read")
        
        label=-1
        if "_gamma" in input_file:
            label=0
        elif "_e" in input_file:
            label=1
        elif "_mu" in input_file:
            label=2
        elif "_pi0" in input_file:
            label=3
        else:
            print "Unknown input file particle type"
            sys.exit()
            
        tree=file.Get("wcsimT")
        print(tree)
        print(type(tree))
        nevent=tree.GetEntries()
    
        print "number of entries in the tree: " + str(nevent)
    
        geotree=file.Get("wcsimGeoT");
    
        print "number of entries in the geometry tree: " + str(geotree.GetEntries())
    
        geotree.GetEntry(0)
        geo=geotree.wcsimrootgeom
        
        num_pmts=geo.GetWCNumPMT()
            
        # All data arrays are initialized here
        FILE_PATHS = []
        FILE_IDX = []
        
        ev_data=[]
        labels=[]
        pids=[]
        positions=[]
        directions=[]
        energies=[]
        n_trigs_displayed=0
        Eth = {22:0.786*2, 11:0.786, -11:0.786, 13:158.7, -13:158.7, 111:0.786*4}
            
        for ev in range(nevent):
            if ev%100 == 0:
                print "now processing event " +str(ev)
            
            tree.GetEvent(ev)
            wcsimrootsuperevent=tree.wcsimrootevent
    
            if ev%100 == 0:
                print "number of sub events: " + str(wcsimrootsuperevent.GetNumberOfEvents())
    
            wcsimrootevent = wcsimrootsuperevent.GetTrigger(0)
            tracks = wcsimrootevent.GetTracks()
            energy=[]
            position=[]
            direction=[]
            pid=[]
            for i in range(wcsimrootevent.GetNtrack()):
                if tracks[i].GetParenttype() == 0 and tracks[i].GetFlag() == 0 and tracks[i].GetIpnu() in Eth.keys():
                    pid.append(tracks[i].GetIpnu())
                    position.append([tracks[i].GetStart(0), tracks[i].GetStart(1), tracks[i].GetStart(2)])
                    direction.append([tracks[i].GetDir(0), tracks[i].GetDir(1), tracks[i].GetDir(2)])
                    energy.append(tracks[i].GetE())
            
            biggestTrigger = 0
            biggestTriggerDigihits = 0
            for index in range(wcsimrootsuperevent.GetNumberOfEvents()):
                wcsimrootevent = wcsimrootsuperevent.GetTrigger(index)
                ncherenkovdigihits = wcsimrootevent.GetNcherenkovdigihits()
                if ncherenkovdigihits > biggestTriggerDigihits:
                    biggestTriggerDigihits = ncherenkovdigihits
                    biggestTrigger = index
    
            wcsimrootevent=wcsimrootsuperevent.GetTrigger(biggestTrigger);
    
            wcsimrootevent=wcsimrootsuperevent.GetTrigger(index);
    
            if ev%100 == 0:
                print "event date and number: "+str(wcsimrootevent.GetHeader().GetDate())+" "+str(wcsimrootevent.GetHeader().GetEvtNum())
                
            ncherenkovhits     = wcsimrootevent.GetNcherenkovhits()
            ncherenkovdigihits = wcsimrootevent.GetNcherenkovdigihits()
    
            if ev%100 == 0:
                print "Ncherenkovdigihits "+str(ncherenkovdigihits)
    
            if ncherenkovdigihits == 0:
                print "event, trigger has no hits "+str(ev)+" "+str(index)
                continue
            
            np_pos_x=np.zeros((ncherenkovdigihits))
            np_pos_y=np.zeros((ncherenkovdigihits))
            np_pos_z=np.zeros((ncherenkovdigihits))
    
            np_dir_u=np.zeros((ncherenkovdigihits))
            np_dir_v=np.zeros((ncherenkovdigihits))
            np_dir_w=np.zeros((ncherenkovdigihits))
    
            np_cylloc=np.zeros((ncherenkovdigihits))
            np_cylloc=np_cylloc-1000
            
            np_q=np.zeros((ncherenkovdigihits))
            np_t=np.zeros((ncherenkovdigihits))
    
            np_pmt_index=np.zeros((ncherenkovdigihits),dtype=np.int32)
            
            """
            The index starts at 1 and counts up continuously with no gaps
            Each 19 consecutive PMTs belong to one mPMT module, so (index-1)/19 is the module number.
            The index%19 gives the position in the module: 1-12 is the outer ring, 13-18 is the inner ring, 0 is the centre PMT
            The modules are then ordered as follows:
            It starts by going round the second highest ring around the barrel, then the third highest ring, fourth highest ring, all the way down to the lowest ring (i.e. skips the highest ring). Then does the bottom end-cap, row by row (the first row has 6 modules, the second row has 8, then 10, 10, 10, 10, 10, 10, 8, 6). Then the highest ring around the barrel that was skipped before, then the top end-cap, row by row. I'm not sure why it has this somewhat strange order...
            WTF: actually it is: 2, 6, 8 10, 10, 12 and down again in the caps
            """
            
            for i in range(ncherenkovdigihits):
                wcsimrootcherenkovdigihit=wcsimrootevent.GetCherenkovDigiHits().At(i)
                
                hit_q=wcsimrootcherenkovdigihit.GetQ()
                hit_t=wcsimrootcherenkovdigihit.GetT()
                hit_tube_id=wcsimrootcherenkovdigihit.GetTubeId()-1
                
                np_pmt_index[i]=hit_tube_id
                np_q[i]=hit_q
                np_t[i]=hit_t
    
            np_module_index=module_index(np_pmt_index)
            np_pmt_in_module_id=pmt_in_module_id(np_pmt_index)
            
            np_wall_indices=np.where(is_barrel(np_module_index))
            
            np_q_wall=np_q[np_wall_indices]
            np_t_wall=np_t[np_wall_indices]
    
            np_wall_row, np_wall_col=row_col(np_module_index[np_wall_indices])
            np_pmt_in_module_id_wall=np_pmt_in_module_id[np_wall_indices]
    
            np_wall_data_rect=np.zeros((16,40,38))
            np_wall_data_rect[np_wall_row,
                              np_wall_col,
                              np_pmt_in_module_id_wall]=np_q_wall
            np_wall_data_rect[np_wall_row,
                              np_wall_col,
                              np_pmt_in_module_id_wall+19]=np_t_wall
    
            np_wall_data_rect_ev=np.expand_dims(np_wall_data_rect,axis=0)
            
            # This part updates the data arrays
            ev_data.append(np_wall_data_rect_ev)
            labels.append(label)
            pids.append(pid)
            positions.append(position)
            directions.append(direction)
            energies.append(energy)
            
            abs_path = os.path.abspath(file_dir)
            
            if not abs_path in existing_paths.keys():
                num_existing_paths += 1
                existing_paths[abs_path] = num_existing_paths
                PATH_FILE.write(abs_path+'\n')
            
            FILE_PATH_IDX.append(existing_paths[abs_path])
            FILE_EV_IDX.append(ev)
            
            wcsimrootsuperevent.ReInitialize()
    
        # Readying all data arrays for saving
        all_events=np.concatenate(ev_data)
        all_labels=np.asarray(labels)
        all_pids=np.asarray(pids)
        all_positions=np.asarray(positions)
        all_directions=np.asarray(directions)
        all_energies=np.asarray(energies)
        
        ALL_FILE_PATHS = np.asarray(FILE_PATH_IDX)
        ALL_FILE_IDX = np.asarray(FILE_EV_IDX)
        
        output_file = config.output_dir+input_file.split('.')[0]+('.npz')
        
        np.savez_compressed(output_file,event_data=all_events,labels=all_labels,pids=all_pids,positions=all_positions,directions=all_directions,energies=all_energies,
                            PATHS=ALL_FILE_PATHS, IDX=ALL_FILE_IDX)
        
        output_list.write(os.path.abspath(output_file)+'\n')
        
    # Close files on completion
    PATH_FILE.close()
    output_list.close()

if __name__ == '__main__':
    
    ROOT.gSystem.Load(os.environ['WCSIMDIR']+"/libWCSimRoot.so")
    config=get_args()
    event_dump(config)
