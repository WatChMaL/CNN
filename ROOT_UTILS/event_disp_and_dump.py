import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap as lsc
from mpl_toolkits.axes_grid1 import ImageGrid

from pos_utils import *
from event_disp_and_dump_arg_utils import get_args

import ROOT
ROOT.gROOT.SetBatch(True)

import os, sys


def event_disp_and_dump(config):

    config.input_file=config.input_file[0]
    print "input file: "+str(config.input_file)
    print "output file: "+str(config.output_file)
    print "n_events_to_display: "+str(config.n_events_to_display)
    
    
    norm=plt.Normalize()
    cm=matplotlib.cm.plasma
    cmaplist = [cm(i) for i in range(cm.N)]
    cm_cat_pmt_in_module = lsc.from_list('Custom cmap', cmaplist, cm.N)
    bounds_cat_pmt_in_module = np.linspace(0,19,20)
    norm_cat_pmt_in_module = matplotlib.colors.BoundaryNorm(bounds_cat_pmt_in_module, cm_cat_pmt_in_module.N)

    cm_cat_module_row = lsc.from_list('Custom cmap', cmaplist, cm.N)
    bounds_cat_module_row = np.linspace(0,16,17)
    norm_cat_module_row = matplotlib.colors.BoundaryNorm(bounds_cat_module_row, cm_cat_module_row.N)

    cm_cat_module_col = lsc.from_list('Custom cmap', cmaplist, cm.N)
    bounds_cat_module_col = np.linspace(0,40,41)
    norm_cat_module_col = matplotlib.colors.BoundaryNorm(bounds_cat_module_col, cm_cat_module_col.N)
    
    
    #file=ROOT.TFile("mu_500MeV_run700_wcsim.root","read")
    file=ROOT.TFile(config.input_file,"read")
    if config.output_file is None:
        config.output_file=config.input_file.replace(".root",".npz")
        print "set output file to: "+config.output_file
    
        
    #file=ROOT.TFile("WCSim_NuPRISM_10x8_mPMT_40perCent_tbugfix_mu_200to1200_0.root","read")
    label=-1
    if "_gamma" in config.input_file:
        label=0
    elif "_e" in config.input_file:
        label=1
    elif "_mu" in config.input_file:
        label=2
    elif "_pi0" in config.input_file:
        label=3
    else:
        print "Unknown input file particle type"
        sys.exit()
        
    tree=file.Get("wcsimT")

    nevent=tree.GetEntries()

    print "number of entries in the tree: " + str(nevent)

    

    geotree=file.Get("wcsimGeoT");
    

    print "number of entries in the geometry tree: " + str(geotree.GetEntries())

    geotree.GetEntry(0)
    geo=geotree.wcsimrootgeom
    
    num_pmts=geo.GetWCNumPMT()



    if config.n_events_to_display>0:
        np_pos_x_all_tubes=np.zeros((num_pmts))
        np_pos_y_all_tubes=np.zeros((num_pmts))
        np_pos_z_all_tubes=np.zeros((num_pmts))
        np_pmt_in_module_id_all_tubes=np.zeros((num_pmts))
        np_pmt_index_all_tubes=np.arange(num_pmts)
        np.random.shuffle(np_pmt_index_all_tubes)
        np_module_index_all_tubes=module_index(np_pmt_index_all_tubes)
        
        for i in range(len(np_pmt_index_all_tubes)):
        
            pmt_tube_in_module_id=np_pmt_index_all_tubes[i]%19
            np_pmt_in_module_id_all_tubes[i]=pmt_tube_in_module_id
            pmt=geo.GetPMT(np_pmt_index_all_tubes[i])
        
            np_pos_x_all_tubes[i]=pmt.GetPosition(2)
            np_pos_y_all_tubes[i]=pmt.GetPosition(0)
            np_pos_z_all_tubes[i]=pmt.GetPosition(1)

            #print "np_pos_z_all_tubes", np_pos_z_all_tubes
        
        np_pos_r_all_tubes=np.hypot(np_pos_x_all_tubes,np_pos_y_all_tubes)
        r_max=np.amax(np_pos_r_all_tubes)
        #np_r_max_all_tubes=np.full((num_pmts),r_max)

        np_wall_indices_ad_hoc=np.unique(np_module_index_all_tubes[np.where( (np_pos_z_all_tubes<499.0) & (np_pos_z_all_tubes>-499.0))[0]])
        np_bottom_indices_ad_hoc=np.unique(np_module_index_all_tubes[np.where( (np_pos_z_all_tubes<-499.0))[0]])
        np_top_indices_ad_hoc=np.unique(np_module_index_all_tubes[np.where( (np_pos_z_all_tubes>499.0))[0]])

        ##print "ad hoc wall indices: "
        ##print np_wall_indices_ad_hoc
        ##
        ##print "ad hoc bottom indices: "
        ##print np_bottom_indices_ad_hoc
        ##
        ##print "ad hoc top indices: "
        ##print np_top_indices_ad_hoc
        ##
        ##print "rearranged ad hoc barrel indices:"
        ##print rearrange_barrel_indices(np_wall_indices_ad_hoc)
        ##
        ##print "row and column from ad hod barrel indices:"
        ##print row_col(np_wall_indices_ad_hoc)
        
        #print "try on bottom:"
        #print row_col(np_bottom_indices_ad_hoc)
        
        #print "try on top:"
        #print row_col(np_top_indices_ad_hoc)
    
    
        np_pos_phi_all_tubes=np.arctan2(np_pos_y_all_tubes, np_pos_x_all_tubes)
        #np_pos_arc_all_tubes=np_pos_r_all_tubes*np_pos_phi_all_tubes
        np_pos_arc_all_tubes=r_max*np_pos_phi_all_tubes
        
        np_wall_indices=np.where(is_barrel(np_module_index_all_tubes))
        np_top_indices=np.where(is_top(np_module_index_all_tubes))
        np_bottom_indices=np.where(is_bottom(np_module_index_all_tubes))
        
        np_pmt_in_module_id_wall_tubes=np_pmt_in_module_id_all_tubes[np_wall_indices]
        np_pmt_in_module_id_top_tubes=np_pmt_in_module_id_all_tubes[np_top_indices]
        np_pmt_in_module_id_bottom_tubes=np_pmt_in_module_id_all_tubes[np_bottom_indices]
        
        np_pos_x_wall_tubes=np_pos_x_all_tubes[np_wall_indices]
        np_pos_y_wall_tubes=np_pos_y_all_tubes[np_wall_indices]
        np_pos_z_wall_tubes=np_pos_z_all_tubes[np_wall_indices]

        np_pos_x_top_tubes=np_pos_x_all_tubes[np_top_indices]
        np_pos_y_top_tubes=np_pos_y_all_tubes[np_top_indices]
        np_pos_z_top_tubes=np_pos_z_all_tubes[np_top_indices]

        np_pos_x_bottom_tubes=np_pos_x_all_tubes[np_bottom_indices]
        np_pos_y_bottom_tubes=np_pos_y_all_tubes[np_bottom_indices]
        np_pos_z_bottom_tubes=np_pos_z_all_tubes[np_bottom_indices]

        np_wall_row, np_wall_col=row_col(np_module_index_all_tubes[np_wall_indices])
        
        np_pos_phi_wall_tubes=np_pos_phi_all_tubes[np_wall_indices]
        np_pos_arc_wall_tubes=np_pos_arc_all_tubes[np_wall_indices]

        fig101 = plt.figure(num=101,clear=True)
        fig101.set_size_inches(10,8)
        ax101 = fig101.add_subplot(111)
        pos_arc_z_disp_all_tubes=ax101.scatter(np_pos_arc_all_tubes, np_pos_z_all_tubes, c=np_pmt_in_module_id_all_tubes,s=5,cmap=cm_cat_pmt_in_module,norm=norm_cat_pmt_in_module,marker='.')
        ax101.set_xlabel('arc along the wall')
        ax101.set_ylabel('z')
        cb_pos_arc_z_disp_all_tubes=fig101.colorbar(pos_arc_z_disp_all_tubes,ticks=range(20),pad=0.1)
        cb_pos_arc_z_disp_all_tubes.set_label("pmt in module")
        fig101.savefig("pos_arc_z_disp_all_tubes.pdf")

        fig102 = plt.figure(num=102,clear=True)
        fig102.set_size_inches(10,8)
        ax102 = fig102.add_subplot(111)
        pos_x_y_disp_all_tubes=ax102.scatter(np_pos_x_all_tubes, np_pos_y_all_tubes, c=np_pmt_in_module_id_all_tubes,s=5,cmap=cm_cat_pmt_in_module,norm=norm_cat_pmt_in_module,marker='.')
        ax102.set_xlabel('x')
        ax102.set_ylabel('y')
        cb_pos_x_y_disp_all_tubes=fig102.colorbar(pos_x_y_disp_all_tubes,ticks=range(20),pad=0.1)
        cb_pos_x_y_disp_all_tubes.set_label("pmt in module")
        fig102.savefig("pos_x_y_disp_all_tubes.pdf")
        
        fig103 = plt.figure(num=103,clear=True)
        fig103.set_size_inches(10,8)
        ax103 = fig103.add_subplot(111)
        pos_arc_z_disp_wall_tubes=ax103.scatter(np_pos_arc_wall_tubes, np_pos_z_wall_tubes, c=np_pmt_in_module_id_wall_tubes,s=5,cmap=cm_cat_pmt_in_module,norm=norm_cat_pmt_in_module,marker='.')
        ax103.set_xlabel('arc along the wall')
        ax103.set_ylabel('z')
        cb_pos_arc_z_disp_wall_tubes=fig103.colorbar(pos_arc_z_disp_wall_tubes,ticks=range(20),pad=0.1)
        cb_pos_arc_z_disp_wall_tubes.set_label("pmt in module")
        fig103.savefig("pos_arc_z_disp_wall_tubes.pdf")
        
        fig104 = plt.figure(num=104,clear=True)
        fig104.set_size_inches(10,8)
        ax104 = fig104.add_subplot(111)
        pos_arc_z_disp_wall_tubes=ax104.scatter(np_pos_arc_wall_tubes, np_pos_z_wall_tubes, c=np_wall_row,s=5,cmap=cm_cat_module_row,norm=norm_cat_module_row,marker='.')
        ax104.set_xlabel('arc along the wall')
        ax104.set_ylabel('z')
        cb_pos_arc_z_disp_wall_tubes=fig104.colorbar(pos_arc_z_disp_wall_tubes,ticks=range(16),pad=0.1)
        cb_pos_arc_z_disp_wall_tubes.set_label("wall module row")
        fig104.savefig("pos_arc_z_disp_wall_tubes_color_row.pdf")
        
        fig105 = plt.figure(num=105,clear=True)
        fig105.set_size_inches(10,8)
        ax105 = fig105.add_subplot(111)
        pos_arc_z_disp_wall_tubes=ax105.scatter(np_pos_arc_wall_tubes, np_pos_z_wall_tubes, c=np_wall_col,s=5,cmap=cm_cat_module_col,norm=norm_cat_module_col,marker='.')
        ax105.set_xlabel('arc along the wall')
        ax105.set_ylabel('z')
        cb_pos_arc_z_disp_wall_tubes=fig105.colorbar(pos_arc_z_disp_wall_tubes,ticks=range(40),pad=0.1)
        cb_pos_arc_z_disp_wall_tubes.set_label("wall module column")
        fig105.savefig("pos_arc_z_disp_wall_tubes_color_col.pdf")
        
        fig106 = plt.figure(num=106,clear=True)
        fig106.set_size_inches(10,8)
        ax106 = fig106.add_subplot(111)
        pos_x_y_disp_top_tubes=ax106.scatter(np_pos_x_top_tubes, np_pos_y_top_tubes, c=np_pmt_in_module_id_top_tubes,s=5,cmap=cm_cat_pmt_in_module,norm=norm_cat_pmt_in_module,marker='.')
        ax106.set_xlabel('x')
        ax106.set_ylabel('y')
        cb_pos_x_y_disp_top_tubes=fig106.colorbar(pos_x_y_disp_top_tubes,ticks=range(20),pad=0.1)
        cb_pos_x_y_disp_top_tubes.set_label("pmt in module")
        fig106.savefig("pos_x_y_disp_top_tubes.pdf")
        
        fig107 = plt.figure(num=107,clear=True)
        fig107.set_size_inches(10,8)
        ax107 = fig107.add_subplot(111)
        pos_x_y_disp_bottom_tubes=ax107.scatter(np_pos_x_bottom_tubes, np_pos_y_bottom_tubes, c=np_pmt_in_module_id_bottom_tubes,s=5,cmap=cm_cat_pmt_in_module,norm=norm_cat_pmt_in_module,marker='.')
        ax107.set_xlabel('x')
        ax107.set_ylabel('y')
        cb_pos_x_y_disp_bottom_tubes=fig107.colorbar(pos_x_y_disp_bottom_tubes,ticks=range(20),pad=0.1)
        cb_pos_x_y_disp_bottom_tubes.set_label("pmt in module")
        fig107.savefig("pos_x_y_disp_bottom_tubes.pdf")
        
        #plt.show()
        
    ev_data=[]
    labels=[]
    pids=[]
    positions=[]
    directions=[]
    energies=[]
    n_trigs_displayed=0
    Eth = {22:0.786*2, 11:0.786, -11:0.786, 13:158.7, -13:158.7, 111:0.786*4}
        
    for ev in range(nevent):
        if ev%100 == 0 or n_trigs_displayed<config.n_events_to_display:
            print "now processing event " +str(ev)
        
        tree.GetEvent(ev)
        wcsimrootsuperevent=tree.wcsimrootevent

        if ev%100 == 0 or n_trigs_displayed<config.n_events_to_display:
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

        if ev%100 == 0 or n_trigs_displayed<config.n_events_to_display:
            print "event date and number: "+str(wcsimrootevent.GetHeader().GetDate())+" "+str(wcsimrootevent.GetHeader().GetEvtNum())
            
        ncherenkovhits     = wcsimrootevent.GetNcherenkovhits()
        ncherenkovdigihits = wcsimrootevent.GetNcherenkovdigihits()

        if ev%100 == 0 or n_trigs_displayed<config.n_events_to_display:
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
            
            #if i<10:
            #    print "q t id: "+str(hit_q)+" "+str(hit_t)+" "+str(hit_tube_id)+" "

            pmt=geo.GetPMT(hit_tube_id)

            

            #if i<10:
            #    print "pmt tube no: "+str(pmt.GetTubeNo()) #+" " +pmt.GetPMTName()
            #    print "pmt cyl loc: "+str(pmt.GetCylLoc())

            #np_cylloc[i]=pmt.GetCylLoc()

            np_pos_x[i]=pmt.GetPosition(2)
            np_pos_y[i]=pmt.GetPosition(0)
            np_pos_z[i]=pmt.GetPosition(1)
              
            np_dir_u[i]=pmt.GetOrientation(2)
            np_dir_v[i]=pmt.GetOrientation(0)
            np_dir_w[i]=pmt.GetOrientation(1)

            np_q[i]=hit_q
            np_t[i]=hit_t

        np_module_index=module_index(np_pmt_index)
        np_pmt_in_module_id=pmt_in_module_id(np_pmt_index)
        
        np_wall_indices=np.where(is_barrel(np_module_index))
        np_top_indices=np.where(is_top(np_module_index))
        np_bottom_indices=np.where(is_bottom(np_module_index))

        if config.n_events_to_display>0:
            np_pos_r=np.hypot(np_pos_x,np_pos_y)
            np_pos_phi=np.arctan2(np_pos_y, np_pos_x)
            #np_pos_arc=np_pos_r*np_pos_phi
            np_pos_arc=r_max*np_pos_phi
            np_pos_arc_wall=np_pos_arc[np_wall_indices]
        
        np_pos_x_top=np_pos_x[np_top_indices]
        np_pos_y_top=np_pos_y[np_top_indices]
        np_pos_z_top=np_pos_z[np_top_indices]
        
        np_pos_x_bottom=np_pos_x[np_bottom_indices]
        np_pos_y_bottom=np_pos_y[np_bottom_indices]
        np_pos_z_bottom=np_pos_z[np_bottom_indices]
        
        np_pos_x_wall=np_pos_x[np_wall_indices]
        np_pos_y_wall=np_pos_y[np_wall_indices]
        np_pos_z_wall=np_pos_z[np_wall_indices]
        
        

        np_q_top=np_q[np_top_indices]
        np_t_top=np_t[np_top_indices]
        
        np_q_bottom=np_q[np_bottom_indices]
        np_t_bottom=np_t[np_bottom_indices]
        
        np_q_wall=np_q[np_wall_indices]
        np_t_wall=np_t[np_wall_indices]

        np_wall_row, np_wall_col=row_col(np_module_index[np_wall_indices])
        np_pmt_in_module_id_wall=np_pmt_in_module_id[np_wall_indices]

        np_wall_data_rect=np.zeros((16,40,38))
        #print "assigning"
        np_wall_data_rect[np_wall_row,
                          np_wall_col,
                          np_pmt_in_module_id_wall]=np_q_wall
        np_wall_data_rect[np_wall_row,
                          np_wall_col,
                          np_pmt_in_module_id_wall+19]=np_t_wall

        np_wall_data_rect_ev=np.expand_dims(np_wall_data_rect,axis=0)
        
        np_wall_q_max_module=np.amax(np_wall_data_rect[:,:,0:19],axis=-1)
        np_wall_q_sum_module=np.sum(np_wall_data_rect[:,:,0:19],axis=-1)
        

        #print "assigned"
        

            #if i<10:
            #    print "x: {}, y: {} z: {}, u: {}, v: {}, w: {}, q: {},  t: {}".format(np_pos_x[i],
            #                                                                          np_pos_y[i],
            #                                                                          np_pos_z[i],
            #                                                                          np_dir_u[i],
            #                                                                          np_dir_v[i],
            #                                                                          np_dir_w[i],
            #                                                                          np_q[i],
            #                                                                          np_t[i])


        max_q=np.amax(np_q)
        np_scaled_q=500*np_q/max_q

        np_dir_u_scaled=np_dir_u*np_scaled_q
        np_dir_v_scaled=np_dir_v*np_scaled_q
        np_dir_w_scaled=np_dir_w*np_scaled_q

        if n_trigs_displayed < config.n_events_to_display:
            
            """
            fig1 = plt.figure(num=1,clear=True)
            fig1.set_size_inches(10,8)
            ax1 = fig1.add_subplot(111, projection='3d',azim=35,elev=20)
            ev_disp=ax1.scatter(np_pos_x,np_pos_y,np_pos_z,c=np_q,s=2,alpha=0.4,cmap=cm,marker='.')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            cb_ev_disp=fig1.colorbar(ev_disp,pad=0.03)
            cb_ev_disp.set_label("charge")
            #plt.setp(ev_disp,markersize=2)
            #print ax.can_zoom()
            fig1.savefig("ev_disp_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig2 = plt.figure(num=2,clear=True)
            fig2.set_size_inches(10,8)
            ax2 = fig2.add_subplot(111, projection='3d',azim=35,elev=20)
            
            #print norm(np_t)
            #print len(np_pos_x), len(np_pos_y), len(np_pos_z), len(np_dir_u), len(np_dir_v), len(np_dir_w), len(np_q), len(np_t) 
            colors = plt.cm.spring(norm(np_t))
            ev_disp_q=ax2.quiver(np_pos_x,np_pos_y,np_pos_z,
                                 np_dir_u_scaled,np_dir_v_scaled,np_dir_w_scaled,
                                 colors=colors,alpha=0.4,cmap=cm)
            #ev_disp_q=ax2.quiver(np_pos_x,np_pos_y,np_pos_z,np_dir_u,np_dir_v,np_dir_w,length=1000,
            #                     color=np_t,cmap=plt.get_cmap("spring"))
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
            sm.set_array([])
            cb_ev_disp_2=fig2.colorbar(sm,pad=0.03)
            cb_ev_disp_2.set_label("time")
            fig2.savefig("ev_disp_quiver_ev_{}_trig_{}.pdf".format(ev,index))"""
            
            print("np_pos_arc_wall shape :", np_pos_arc_wall.shape)
            print("np_pos_arc_wall :", np_pos_arc_wall)
            
            print("np_pos_z_wall shape :", np_pos_z_wall.shape)
            print("np_pos_z_wall :", np_pos_z_wall)
            
            print("np_q_wall shape :", np_q_wall.shape)
            print("np_q_wall :", np_q_wall)
            
            fig3 = plt.figure(num=3,clear=True)
            fig3.set_size_inches(10,8)
            ax3 = fig3.add_subplot(111)
            ev_disp_wall=ax3.scatter(np_pos_arc_wall, np_pos_z_wall, c=np_q_wall,s=2,cmap=cm,marker='.')
            ax3.set_xlabel('arc along the wall')
            ax3.set_ylabel('z')
            cb_ev_disp_wall=fig3.colorbar(ev_disp_wall,pad=0.1)
            cb_ev_disp_wall.set_label("charge")
            fig3.savefig("ev_disp_wall_ev_{}_trig_{}.pdf".format(ev,index))
            
            """
            
            fig4 = plt.figure(num=4,clear=True)
            fig4.set_size_inches(10,8)
            ax4 = fig4.add_subplot(111)
            ev_disp_top=ax4.scatter(np_pos_x_top, np_pos_y_top, c=np_q_top,s=2,cmap=cm,marker='.')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            cb_ev_disp_top=fig4.colorbar(ev_disp_top,pad=0.1)
            cb_ev_disp_top.set_label("charge")
            fig4.savefig("ev_disp_top_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig5 = plt.figure(num=5,clear=True)
            fig5.set_size_inches(10,8)
            ax5 = fig5.add_subplot(111)
            ev_disp_bottom=ax5.scatter(np_pos_x_bottom, np_pos_y_bottom, c=np_q_bottom,s=2,cmap=cm,marker='.')
            ax5.set_xlabel('x')
            ax5.set_ylabel('y')
            cb_ev_disp_bottom=fig5.colorbar(ev_disp_bottom,pad=0.1)
            cb_ev_disp_bottom.set_label("charge")
            fig5.savefig("ev_disp_bottom_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig6 = plt.figure(num=6,clear=True)
            fig6.set_size_inches(10,4)
            ax6 = fig6.add_subplot(111)
            q_sum_disp=ax6.imshow(np.flip(np_wall_q_sum_module,axis=0), cmap=cm)
            #q_sum_disp=ax6.imshow(np_wall_q_sum_module, cmap=cm)
            ax6.set_xlabel('arc index')
            ax6.set_ylabel('z index')
            cb_q_sum_disp=fig6.colorbar(q_sum_disp,pad=0.1)
            cb_q_sum_disp.set_label("total charge in module")
            fig6.savefig("q_sum_disp_ev_{}_trig_{}.pdf".format(ev,index))
            
            
            fig7 = plt.figure(num=7,clear=True)
            fig7.set_size_inches(10,4)
            ax7 = fig7.add_subplot(111)
            q_max_disp=ax7.imshow(np.flip(np_wall_q_max_module,axis=0), cmap=cm)
            #q_max_disp=ax7.imshow(np_wall_q_max_module, cmap=cm)
            ax7.set_xlabel('arc index')
            ax7.set_ylabel('z index')
            cb_q_max_disp=fig7.colorbar(q_max_disp,pad=0.1)
            cb_q_max_disp.set_label("maximum charge in module")
            fig7.savefig("q_max_disp_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig8 = plt.figure(num=8,clear=True)
            fig8.set_size_inches(10,8)
            ax8 = fig8.add_subplot(111)
            plt.hist(np_q, 50, density=True, facecolor='blue', alpha=0.75)
            ax8.set_xlabel('charge')
            ax8.set_ylabel("PMT's above threshold")
            fig8.savefig("q_pmt_disp_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig9 = plt.figure(num=9,clear=True)
            fig9.set_size_inches(10,8)
            ax9 = fig9.add_subplot(111)
            plt.hist(np_t, 50, density=True, facecolor='blue', alpha=0.75)
            ax9.set_xlabel('time')
            ax9.set_ylabel("PMT's above threshold")
            fig9.savefig("t_pmt_disp_ev_{}_trig_{}.pdf".format(ev,index))
            
            fig10 = plt.figure(num=10,clear=True)
            fig10.set_size_inches(15,5)
            grid_q = ImageGrid(fig10, 111,  
                               nrows_ncols=(4, 5),
                               axes_pad=0.0,
                               share_all=True,
                               label_mode="L",
                               cbar_location="top",
                               cbar_mode="single",
            )
            for i in range(19):
                q_disp=grid_q[i].imshow(np.flip(np_wall_data_rect[:,:,i],axis=0), cmap=cm)
                q_disp=grid_q[19].imshow(np.flip(np_wall_q_max_module,axis=0), cmap=cm)
                grid_q.cbar_axes[0].colorbar(q_disp)
                                 
            fig10.savefig("q_disp_grid_ev_{}_trig_{}.pdf".format(ev,index))


        
            fig11 = plt.figure(num=11,clear=True)
            fig11.set_size_inches(15,5)
            grid_t = ImageGrid(fig11, 111,  
                               nrows_ncols=(4, 5),
                               axes_pad=0.0,
                               share_all=True,
                               label_mode="L",
                               cbar_location="top",
                               cbar_mode="single",
            )
            for i in range(19):
                t_disp=grid_t[i].imshow(np.flip(np_wall_data_rect[:,:,i+19],axis=0), cmap=cm)
        
                
        
            fig11.savefig("t_disp_grid_ev_{}_trig_{}.pdf".format(ev,index))

            n_trigs_displayed+=1
        
            #plt.show()"""

        ev_data.append(np_wall_data_rect_ev)
        labels.append(label)
        pids.append(pid)
        positions.append(position)
        directions.append(direction)
        energies.append(energy)
            
        #print "\n\n"
        
        wcsimrootsuperevent.ReInitialize()

    all_events=np.concatenate(ev_data)
    all_labels=np.asarray(labels)
    all_pids=np.asarray(pids)
    all_positions=np.asarray(positions)
    all_directions=np.asarray(directions)
    all_energies=np.asarray(energies)
    np.savez_compressed(config.output_file,event_data=all_events,labels=all_labels,pids=all_pids,positions=all_positions,directions=all_directions,energies=all_energies)
        #for i in range(ncherenkovhits):
        #    wcsimrootcherenkovhit=wcsimrootevent.GetCherenkovHits().At(i)
        #    tubeNumber=wcsimrootcherenkovhit.GetTubeID()
        #    if i<10:
        #        print "tube number: "+str(tubeNumber) 
                
if __name__ == '__main__':
    
    ROOT.gSystem.Load(os.environ['WCSIMDIR']+"/libWCSimRoot.so")
    config=get_args()
    event_disp_and_dump(config)
    
