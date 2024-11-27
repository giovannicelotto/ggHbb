# %%
import glob, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import getDfProcesses
import mplhep as hep
hep.style.use("CMS")
# %%
dfProc = getDfProcesses()
isMC = 36
flatPath = dfProc.flatPath[isMC]
process = dfProc.process[isMC]
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched/%s"%process
fileNames_gm = glob.glob(path+"/*.parquet")

fileNumbers = []
for fileName in fileNames_gm:
    match = re.search(r'_(\d+)\.parquet$', fileName)
    if match:
        fileNumber = match.group(1)
        #print(f"File number: {fileNumber}")
        fileNumbers.append(int(fileNumber))
    else:
        pass
        #print("No match found.")

df_gm = pd.read_parquet(fileNames_gm)
# %%

fileNames = glob.glob(flatPath+"/*.parquet")
fileNamesFiltered = []
for fileName in fileNames:
    match = re.search(r'_(\d+)\.parquet$', fileName)
    if match:
        fileNumber = int(match.group(1))
        if fileNumber not in fileNumbers:
            print("Here", len(fileNames))
            #fileNames.remove(fileName)
            print( len(fileNames))
        else:
            fileNamesFiltered.append(fileName)
    else:
        assert False

df = pd.read_parquet(fileNamesFiltered)
# %%
print(len(fileNamesFiltered))
print(len(fileNames_gm))
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(20, 300, 101)
c_gm = np.histogram(df_gm.dijet_mass_2018, bins=bins)[0]
c = np.histogram(df.dijet_mass, bins=bins)[0]

#ax.hist(bins[:-1], bins=bins, weights=(c-c_gm), histtype='step')
ax.hist(bins[:-1], bins=bins, weights=(c), histtype='step', label='Dijet Chosen')
ax.hist(bins[:-1], bins=bins, weights=(c_gm), histtype='step', label='Jet GenMatched')
ax.set_title(process)
ax.legend()
# %%
