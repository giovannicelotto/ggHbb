
# %%
import numpy as np
import pandas as pd
from functions import getDfProcesses_v2

dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
dfsData = []
lumi_tot = 0.
modelName = "Aug28_3_20p01"
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/%s"%modelName
for processName in dfProcessesData.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/dataframes_%s_%s.parquet"%(processName, modelName))
    dfsData.append(df)
    lumi_tot = lumi_tot + np.load(path+"/lumi_%s_%s.npy"%(processName, modelName))
# %%
dfsQCD = []
MCList_QCD = [23, 24, 25, 26, 27, 28, 29, 30,  31, 32, 33, 34]
for processName in dfProcessesMC.iloc[MCList_QCD].process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
    dfsQCD.append(df)

# %%
for idx, df in enumerate(dfsQCD):
    dfsQCD[idx].weight=dfsQCD[idx].weight*lumi_tot
# %%
dfQCD = pd.concat(dfsQCD)
dfData = pd.concat(dfsData)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
bins = np.linspace(70, 300, 21)
c=ax.hist(dfQCD.dijet_mass[dfQCD.PNN > 0.75], weights=dfQCD.weight[dfQCD.PNN > 0.75]*0.72, bins=bins, histtype='step', label='QCD MC', density=False)[0]
cData=ax.hist(dfData.dijet_mass[dfData.PNN > 0.75], bins=bins, histtype='step', label='Data', density=False)[0]

ax.legend()

fig, ax = plt.subplots(1,1)
ax.hist(bins[:-1], weights=cData-c,           bins=bins, histtype='step', label='QCD MC', density=False)[0]
ax.legend()
# %%
