import sys
import uproot
import awkward as ak
import numpy as np
import time
import ROOT
from helpers import acquire_lock, release_lock

def process_file(fileName, process):
    bins= np.load("/t3home/gcelotto/ggHbb/bkgEstimation/output/binsForHT.npy")
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    Jet_pt                      = branches["Jet_pt"]
    Muon_isTriggering           = branches["Muon_isTriggering"]
    # for mu9ip6 only
    #Muon_fired_HLT_Mu9_IP6      = branches["Muon_fired_HLT_Mu9_IP6"]
    Muon_pt                     = branches["Muon_pt"]
    Muon_dxy                    = branches["Muon_dxy"]
    Muon_dxyErr                 = branches["Muon_dxyErr"]
    
    mask_event = ak.sum(Muon_isTriggering, axis=-1)>0     # events where there is one muon that triggers the trigger path
    # uncomment following lines for mu9ip6
    #mask_event = ak.sum(Muon_fired_HLT_Mu9_IP6, axis=-1)>0     # events where there is one muon that triggers the trigger path
    #muon9ip6_mask = Muon_fired_HLT_Mu9_IP6[mask_event]>0        # muons that pass trigger in events where at least one muon passes
    # to select the leading then use Muon_pt[mask_event][muon9ip6_mask][:, 0:1] since they are ordered per pT

    # open hist for scale factor
    if process!='Data':
        print("Open histo for ", process)

        histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
        file = uproot.open(histPath)
        histogram = file["hist_scale_factor"]
        h = histogram.values()
        xedges = [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 20.0, 100.0]
        yedges = [0.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 500.0]
        #for mu9ip6
        #bin_coords = ak.zip({"x": Muon_pt[mask_event][muon9ip6_mask][:, 0:1], "y": abs(Muon_dxy[mask_event][muon9ip6_mask][:, 0:1]/Muon_dxyErr[mask_event][muon9ip6_mask][:, 0:1])})
        bin_coords = ak.zip({"x": Muon_pt[mask_event][:, 0:1], "y": abs(Muon_dxy[mask_event][:, 0:1]/Muon_dxyErr[mask_event][:, 0:1])})
        bin_indices_x = np.searchsorted(xedges, bin_coords.x, side="right") - 1
        bin_indices_x = ak.where(bin_indices_x<=9, bin_indices_x, 9)            # TEMPORARY no trig scale factor for pt > 100 GeV
        bin_indices_y = np.searchsorted(yedges, bin_coords.y, side="right") - 1
        bin_indices_y = ak.where(bin_indices_y<=8, bin_indices_y, 8)           # temporary

        # Find the bin content for each pair of bin coordinates
        print("Trigger SF in memory ... for", process)
        triggerSF = h[bin_indices_x, bin_indices_y]
    else:
        #for mu9ip6
        triggerSF = np.ones(np.sum(mask_event))
        

    ht = ak.sum(Jet_pt, axis=-1)
    #mu9ip6
    ht = ht[mask_event]
    

    assert len(ht) == len(triggerSF)
    



    numEventsTotal = 0

    if process != 'Data':
        lumiBlocks = f['LuminosityBlocks']
        numEventsTotal = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])
        
        print("Acquiring lock 1... for ", process)
        lock_fd = acquire_lock("/t3home/gcelotto/ggHbb/bkgEstimation/output/lockFile1_%s.lock"%process)
        numOld = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/output/mini_%s.npy" % process)
        numUpdate = numOld + numEventsTotal
        np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/mini_%s.npy" % process, numUpdate)
        release_lock(lock_fd)
        


    lock_fd = acquire_lock("/t3home/gcelotto/ggHbb/bkgEstimation/output/lockFile2_%s.lock"%process)
    counts = np.histogram(np.clip(ht, bins[0], bins[-1]), bins=bins, weights=triggerSF.reshape(-1) if process!='Data' else None)[0]
    countsOld = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/output/counts_%s.npy"%process)
    countsNew = countsOld + counts
    print(countsNew)
    np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/counts_%s.npy"%process, countsNew)
           
    release_lock(lock_fd)

   
    return 

if __name__ == "__main__":
    fileName = sys.argv[1]
    process = sys.argv[2] 
    
    process_file(fileName, process)


