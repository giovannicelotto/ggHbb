# %%
import pandas as pd
import numpy as np
import glob, re
import uproot
# %%
path ="/t3home/gcelotto/ggHbb/commonScripts/processes.csv"
dfProcess = pd.read_csv(path)

# %%
isMCList = [0, 1]
for isMC in isMCList:
    nanoPath = dfProcess.nanoPath[isMC]
    flatPath = dfProcess.flatPath[isMC]

    nanoCounter = 0
    nanoFileNumberLists = []
    flatFileNumberLists = []
    for nanoFileName in glob.glob(nanoPath+"/**/*.root", recursive=True)[:40]:
        fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1))
        nanoFileNumberLists.append(fileNumber)
        f = uproot.open(nanoFileName)
        tree = f['Events']
        nanoCounter = nanoCounter + tree.num_entries
    print("Process %s with %d nano" %(dfProcess.process[isMC], nanoCounter))
    flatCounter = 0
    for flatFileName in glob.glob(flatPath+"/**/*.parquet", recursive=True):
        fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', flatFileName).group(1))
        if fileNumber in nanoFileNumberLists:
            df = pd.read_parquet(flatFileName)
            flatCounter = flatCounter + len(df)
        else:
            continue
    print("Process %s with %d nano" %(dfProcess.process[isMC], nanoCounter))
    print("Process %s with %d flat" %(dfProcess.process[isMC], flatCounter))
    print("Efficiency %d / %d = %.1f%%" %(flatCounter, nanoCounter, flatCounter/nanoCounter*100))
        


# %%
