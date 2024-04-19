import uproot
import glob
import sys
import numpy as np
import pandas as pd
'''
    Compute number of NanoAOD entries of BPH 2018 1A
'''
def getFlatEntries():
    fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**/*.parquet")
    print("Number of Flat Data : ", len(fileNames))
    totalFlatEntries = 0
    for fileName in fileNames:
        f = pd.read_parquet(fileName)
        maxEntries = len(f)
        totalFlatEntries += maxEntries
        print("%d/%d\n\t\t"%(fileNames.index(fileName)+1, len(fileNames)), totalFlatEntries)
    np.save("/t3home/gcelotto/ggHbb/outputs/counters/N_BPH_Flat.npy", totalFlatEntries)
    return totalFlatEntries, len(fileNames)

def getNanoEntries():
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*"
    fileNames = glob.glob(path+"/*.root")
    print(len(fileNames))
    totalEvents = 0
    for fileName in fileNames:
        sys.stdout.write('\r')
        sys.stdout.write("%d/%d    :    %d"%(fileNames.index(fileName), len(fileNames), totalEvents))
        sys.stdout.flush()
        with uproot.open(fileName) as f:
            tree = f['Events']
            totalEvents += tree.num_entries
    print("\n\n",totalEvents)
    np.save("/t3home/gcelotto/ggHbb/outputs/counters/N_BPH_Nano.npy", totalEvents)


totalFlat, nFiles = getFlatEntries()
lumiPerEvent = 0.774 * nFiles / 1017 * 1./totalFlat
print("Lumi Per Event", lumiPerEvent, " fb-1")
np.save("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy", lumiPerEvent)

