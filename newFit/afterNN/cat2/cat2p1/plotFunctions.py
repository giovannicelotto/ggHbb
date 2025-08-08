
# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
f0 = pd.read_parquet("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/df_0.parquet")
f1 = pd.read_parquet("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/df_1.parquet")
f16 = pd.read_parquet("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/df_16.parquet")

mean_values = (f0.y + f1.y + f16.y)/3
fig, ax = plt.subplots(1, 1)
ax.plot(f0.x, f0.y-mean_values, color='C0', label='F0' )
ax.plot(f0.x, f1.y-mean_values, color='C1', label='F1')
ax.plot(f16.x, f16.y-mean_values, color='C2', label='F16' )
ax.plot(f0.x, mean_values-mean_values, label='Mean', color='black', linewidth=0.5)
ax.fill_between(f0.x, f0.y - mean_values - f0.yerr, f0.y - mean_values + f0.yerr, facecolor="C0", alpha=0.2)
ax.fill_between(f1.x, f1.y - mean_values - f1.yerr, f1.y - mean_values + f1.yerr, facecolor="C1", alpha=0.2)
ax.fill_between(f16.x, f16.y - mean_values - f16.yerr, f16.y - mean_values + f16.yerr, facecolor="C2", alpha=0.2)

import uproot
histSignal_values = uproot.open("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/hists/counts_cat2p1.root")["Higgs_nominal"].values()
ax.plot(f0.x, histSignal_values, color='red', label='Signal')
ax.legend()
#ax.set_xlim(70, 220)
ax.set_ylim(-100, 100)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Fit Function - Mean ")
fig.savefig("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/function_bkg.png")
# %%
