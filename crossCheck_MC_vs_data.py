# %%
from functions import *
import matplotlib.pyplot as plt
import pandas as pd
# %%
isMCList =[23,24,25,26,27,28,29,30,31,32,33,34]
dfs, sumw = loadMultiParquet_v2(paths=isMCList, nMCs=-1, returnNumEventsTotal=True)
# %%
dfProcess = getDfProcesses_v2()[0]
for idx, df in enumerate(dfs):
    print(idx, dfProcess.xsection.iloc[isMCList].iloc[idx])
    dfs[idx]['weight'] = df.genWeight * df.btag_central * df.sf * df.PU_SF * dfProcess.xsection.iloc[isMCList].iloc[idx] * 1000 /sumw[idx]
dfMC = pd.concat(dfs)
# %%
dfsData, lumi = loadMultiParquet_Data_new(dataTaking=[1], nReals=-1)
# %%
# %%
bins = np.linspace(40, 300, 51)
fig, ax = plt.subplots(1, 1)
ax.hist(dfMC.dijet_mass, bins=bins, weights=dfMC.weight*1000)
ax.hist(dfMC.dfsData[0], bins=bins)