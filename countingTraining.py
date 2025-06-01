# %%
from functions import *
import pandas as pd
import numpy as np
# %%
dfs, lumi = loadMultiParquet_Data_new(dataTaking=[2], nReals=[600], filters=getCommonFilters(btagTight=True))
# %%
dfsMC, sumw = loadMultiParquet_v2(paths=[0], nMCs=-1, returnNumEventsTotal=True)
# %%
dfProcess = getDfProcesses_v2()[0]
# %%
dfMC = pd.concat(dfsMC)
dfMC['weight'] = dfMC.genWeight * dfMC.btag_central * dfMC.PU_SF * dfMC.sf * dfProcess.xsection.iloc[0] * lumi * 1000 /sumw[0]
# %%

num = dfMC[(dfMC.dijet_mass>100) & (dfMC.dijet_mass<150) & (dfMC.dijet_pt>160)].weight.sum()
den = len(dfs[0][(dfs[0].dijet_mass>100) & (dfs[0].dijet_mass<150)& (dfs[0].dijet_pt>160)])
# %%
num/den
# %%
