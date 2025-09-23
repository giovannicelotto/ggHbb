# %%
from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
MCList = [37]
filters = getCommonFilters(btagTight=False, cutDijet=False)
dfs, sumw = loadMultiParquet_v2(paths=MCList,  
                        nMCs=-1, 
                          returnNumEventsTotal=True, filters=filters)
# %%
dfsData, lumi = loadMultiParquet_Data_new(dataTaking=[17], nReals=1000, columns=None, 
                                          filters=filters)
# %%
dfProcess = getDfProcesses_v2()[0].iloc[MCList]
# %%
for idx, df in enumerate(dfs):
    dfs[idx]['weight'] = dfs[idx].genWeight * dfs[idx].btag_central * dfs[idx].PU_SF * dfs[idx].sf *lumi*dfProcess.xsection.iloc[idx]/sumw[idx] * 1000
# %%
cut(dfs, 'dijet_mass', 80, 170)
cut(dfsData, 'dijet_mass', 80, 170)
# Medium WP summary
print("="*40)
print("Medium WP Summary")
print("="*40)

# Signal
signal_all = dfs[0].weight.sum() * 41.6 / lumi
signal_tight = dfs[0][(dfs[0].jet1_btagTight == 1) & (dfs[0].jet2_btagTight == 1)].weight.sum() * 41.6 / lumi

# Data
data_all = len(dfsData[0]) * 41.6 / lumi
data_tight = len(dfsData[0][(dfsData[0].jet1_btagTight == 1) & (dfsData[0].jet2_btagTight == 1)]) * 41.6 / lumi

print(f"Signal (all events):   {signal_all:,.2f}")
print(f"Signal (tight jets):   {signal_tight:,.2f}")
print(f"Data   (all events):   {data_all:,}")
print(f"Data   (tight jets):   {data_tight:,}")
print("="*40)

# %%
