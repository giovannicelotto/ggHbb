# %%
from functions import getDfProcesses, loadMultiParquet, cut
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import pandas as pd
import numpy as np
# %%
isMCList = [40, 41, 42, 1, 43, 44]
dfProcesses = getDfProcesses()
flatPaths = dfProcesses.flatPath[isMCList]
dfs = loadMultiParquet(isMCList, nReal=-2, nMC=-1, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, columns=['dijet_mass', 'sf', 'PU_SF'])
# %%
dfs = cut(dfs, 'dijet_mass', 40, 300)
bins = np.linspace(40, 300, 101)
fig, ax = plt.subplots(1, 1)
mass = [50, 70, 100, 125, 200, 300]
for idx, df in enumerate(dfs):
    c = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    c = c/np.sum(c)
    ax.hist(bins[:-1], bins=bins, weights=c, histtype='step', linewidth=2, label='M = %d GeV'%mass[idx])
hep.cms.label()
ax.legend()
ax.set_xlim(40, 300)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Nomralized Counts")
# %%
