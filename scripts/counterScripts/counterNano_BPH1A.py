import uproot
import glob
import sys
import numpy as np
'''
    Compute number of NanoAOD entries of BPH 2018 1A
'''
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003"
fileNames = glob.glob(path+"/*/*.root")
print(len(fileNames))
totalEvents = 0
for fileName in fileNames:
    sys.stdout.write('\r')
    sys.stdout.write("%d/%d\t:\t%d"%(fileNames.index(fileName), len(fileNames), totalEvents))
    sys.stdout.flush()
    with uproot.open(fileName) as f:
        tree = f['Events']
        totalEvents += tree.num_entries
print("\n\n",totalEvents)
np.save("/t3home/gcelotto/ggHbb/outputs/N_BPH_Nano.npy", totalEvents)

