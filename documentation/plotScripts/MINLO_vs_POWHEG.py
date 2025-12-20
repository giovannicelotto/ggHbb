# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet_v2, getDfProcesses_v2, cut
from hist import Hist
# %%
dfProcesses = getDfProcesses_v2()[0]
# %%
dfs, sumw = loadMultiParquet_v2(paths=[0,37], nMCs=[-1, -1], returnNumEventsTotal=True, filters=None)
# %%
for df, s in zip(dfs, sumw):
    df['weight'] = df.genWeight /s * 1000 * 41.6 * dfProcesses.xsection.iloc[0]



# %%
bins=np.linspace(0, 250, 51)
h_powheg = Hist.new.Var(bins, name="$p_T$ [GeV]").Weight()
h_minlo  = Hist.new.Var(bins, name="$p_T$ [GeV]").Weight()

h_powheg.fill(dfs[0].dijet_pt, weight=dfs[0].weight)
h_minlo.fill(dfs[1].dijet_pt, weight=dfs[1].weight)

fig, ax = plt.subplots(1, 1)

h_powheg.plot(ax=ax, label='POWHEG')
h_minlo.plot(ax=ax, label='MINLO')
#ax.set_yscale('log')
#ax.hist(dfs[0].dijet_pt, bins=bins, weights=dfs[0].weight, histtype='step', label='POWHEG')
#ax.hist(dfs[1].dijet_pt, bins=bins, weights=dfs[1].weight, histtype='step', label="MINLO")
ax.legend()
ax.set_ylabel("Events / %d GeV" % (bins[1]-bins[0]))
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/MINLO_vs_POWHEG_Higgs/Higgs_Pt.png")

# %%
