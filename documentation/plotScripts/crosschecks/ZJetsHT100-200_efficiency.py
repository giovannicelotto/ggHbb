# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import pandas as pd
from functions import  *
# %%

from functions import getCommonFilters

dfs, sumw = loadMultiParquet_v2(paths=[35], nMCs=-1, returnNumEventsTotal=True, filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=9))
# %%
df = dfs[0]
df['weight']  = df['flat_weight'] * 41.6 * df['xsection'] / sumw[0]
# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.hist(dfs[0]["dijet_mass"], bins=50, range=(0, 1000), histtype="step", label=f"ZJets HT100-200 (%d entries)"%(df.weight.sum()), weights=df.weight)
ax.legend()
# %%
dfs, sumw = loadMultiParquet_v2(paths=[35], nMCs=-1, returnNumEventsTotal=True, filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=12))
# %%
df = dfs[0]
df['weight']  = df['flat_weight'] * 41.6 * df['xsection'] / sumw[0]
# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.hist(dfs[0]["dijet_mass"], bins=50, range=(0, 1000), histtype="step", label=f"ZJets HT100-200 (%d entries)"%(df.weight.sum()), weights=df.weight)
ax.legend()
# %%
