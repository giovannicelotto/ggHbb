import sys
import uproot
import awkward as ak
import numpy as np
from helpersForLockers import acquire_lock, release_lock

def process_file(fileName, label):
    bins= np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/bins.npy")
    
    f =  uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    PV_npvs         = branches["PV_npvs"]
    #PV_npvsGood     = branches["PV_npvsGood"]
    
    if label!='Data':
        lumiBlocks = f['LuminosityBlocks']
        numEventsTotal = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])

        print("Acquiring lock 1... for ", label)
        lock_fd = acquire_lock("/t3home/gcelotto/ggHbb/PU_reweighting/output/lockFile1_%s.lock"%label)
        numOld = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/numEventsTotal_%s.npy"%label)
        numNew = numOld + numEventsTotal
        np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/numEventsTotal_%s.npy" % label, numNew)
        release_lock(lock_fd)
    
    counts = np.histogram(np.clip(PV_npvs, bins[0], bins[-1]), bins=bins)[0]
    print("Acquiring lock 2... for ", label)
    lock_fd = acquire_lock("/t3home/gcelotto/ggHbb/PU_reweighting/output/lockFile2_%s.lock"%label)
    countsOld = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/counts_%s.npy" % label)
    countsNew = countsOld + counts
    np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/counts_%s.npy" % label, countsNew)
    release_lock(lock_fd)

   
    return 

if __name__ == "__main__":
    fileName = sys.argv[1]
    label = sys.argv[2] 
    
    process_file(fileName, label)


