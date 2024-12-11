# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import cut, loadMultiParquet, getDfProcesses
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
import glob
from hist import Hist
import json
file_path = '/t3home/gcelotto/ggHbb/abcd/dynamicABCD/cutsComboCollector.json'


# %%
dfProcesses = getDfProcesses()
isMCList = [0, 1,
                2,
                3, 4, 5,
                6,7,8,9,10,11,
                12,13,14,
                15,16,17,18,19,
                20, 21, 22, 23, 36,
                #39    # Data2A
    ]
paths = dfProcesses.flatPath[isMCList]
paths[1] = paths[1]+"/others"
paths[0] = paths[0]+"/others"

nReal = 300
nMC = -1
columns = ['dijet_mass', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'sf', 'PU_SF']
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                          columns=columns,
                                                          returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = preprocessMultiClass(dfs=dfs)
for idx, df in enumerate(dfs):
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
    # make uinque data columns
if isMCList[-1]==39:
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
# remove the last element (data2a)
    dfs = dfs[:-1]
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))
# %%
# Open and read the JSON file
with open(file_path, 'r') as f:
    cutsComboCollector = json.load(f)
# %%
bins = np.linspace(40, 300, 20)
dfs_new =[]
for idx, (bmin, bmax) in enumerate(zip(bins[:-1], bins[1:])):
    dfs_b = cut(dfs, 'dijet_mass', bmin, bmax)
    maxIdx = np.argmax(cutsComboCollector[idx]["effSignal"])
    dfs_b = cut(dfs_b, 'jet1_btagDeepFlavB', cutsComboCollector[idx]["cut1"][maxIdx], None)
    dfs_b = cut(dfs_b, 'jet1_btagDeepFlavB', cutsComboCollector[idx]["cut2"][maxIdx], None)
    dfs_new.append(dfs_b)
# %%
dfs_rejected =[]
for idx, (bmin, bmax) in enumerate(zip(bins[:-1], bins[1:])):
    dfs_b = cut(dfs, 'dijet_mass', bmin, bmax)
    maxIdx = np.argmax(cutsComboCollector[idx]["effSignal"])
    dfs_b = cut(dfs_b, 'jet1_btagDeepFlavB', None, cutsComboCollector[idx]["cut1"][maxIdx])
    dfs_b = cut(dfs_b, 'jet1_btagDeepFlavB', None, cutsComboCollector[idx]["cut2"][maxIdx])
    dfs_rejected=dfs_rejected+dfs_b

# %%
fig, ax = plt.subplots(1, 1)

ax.hist(dfs_rejected.dijet_mass, bins=bins)

# %%
