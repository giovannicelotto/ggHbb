# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import mplhep as hep
hep.style.use("CMS")
# %%
flatCommonPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
dfs = []
massHypo = [50, 70, 100, 125, 200, 300]
for m in massHypo:
    if m==125:
        fileNames = glob.glob(flatCommonPath+"/GluGluHToBB/**/*.parquet")[:50]
    else:
        fileNames = glob.glob(flatCommonPath+"/GluGluH_M%d_ToBB/*.parquet"%m)
    print(m, len(fileNames))
    df = pd.read_parquet(fileNames)
    dfs.append(df)
# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 500, 101)
for df,m in zip(dfs,massHypo):
    c = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    c = c/np.sum(c)
    ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label="M = %d GeV"%m)
ax.legend()
hep.cms.label()
ax.set_ylabel("Normalized Events")
ax.set_xlabel("Dijet Mass [GeV]")
# %%
