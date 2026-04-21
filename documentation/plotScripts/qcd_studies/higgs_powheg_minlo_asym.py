# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import pandas as pd
from functions import loadMultiParquet_v2, getCommonFilters
# %%
filters= getCommonFilters(btagWP="M", cutDijet=False)
filters[0] = filters[0] + [('dijet_pt', '>=', 60), ('dijet_pt', '<', 120)]
filters[1] = filters[1] + [('dijet_pt', '>=', 60), ('dijet_pt', '<', 120)]
dfs, genw = loadMultiParquet_v2(paths=[0,37,46], returnNumEventsTotal=True,nMCs=-1, filters=filters)
# %%
labels=["POWHEG", "MINLO", "amcatnlo" ]
fig ,ax = plt.subplots(1, 1, )
for idx, (df, label) in enumerate(zip(dfs, labels)):
    ax.hist(df.dijet_pT_asymmetry, bins=np.linspace(-1, 1, 41), histtype='step', label=label, density=False, weights=df.flat_weight/genw[idx])
    print(label, df.flat_weight[abs(df.dijet_pT_asymmetry)<0.45].sum() / df.flat_weight.sum())
ax.legend()
ax.set_title("60 < dijet p$_T$ < 120 GeV")
ax.set_xlabel("dijet p$_T$ asymmetry")
# %%

dfs, genw = loadMultiParquet_v2(paths=[0,37,46], returnNumEventsTotal=True,nMCs=-1, filters=None)
# %%
fig ,ax = plt.subplots(1, 1, )
for idx, (df, label) in enumerate(zip(dfs, labels)):
    ax.hist(df.dijet_pt, bins=np.linspace(0, 300, 41), histtype='step', label=label, density=False, weights=df.flat_weight/genw[idx])
ax.legend()
#ax.set_yscale("log")
ax.set_xlabel("dijet p$_T$ [GeV]")
# %%
