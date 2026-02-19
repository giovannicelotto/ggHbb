# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import mplhep as hep
hep.style.use("CMS")
# %%
df_ggH = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/GluGluHTo2B_Run3/GluGlu*.parquet"))
df_data = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/DataRun3/Data_2025/*.parquet"))
df_ggH = df_ggH[df_ggH.dijet_pt>100]
df_data = df_data[df_data.dijet_pt>100]
# %%
fig, ax = plt.subplots(1, 1)
bins_NN = np.linspace(0, 1, 31)
ax.hist(df_ggH.PNN, bins=bins_NN, histtype='step', label="ggH", density=True)
ax.hist(df_data.PNN, bins=bins_NN, histtype='step', label="Data", density=True)
ax.text(0.95, 0.95, s=r"Dijet p$_T$ > 100 GeV", transform=ax.transAxes, ha='right', va='top')
ax.set_xlabel("NN output")
ax.legend()
# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Scores
scores_sig = df_ggH.PNN.values
scores_bkg = df_data.PNN.values

y_true = np.concatenate([
    np.ones(len(scores_sig)),
    np.zeros(len(scores_bkg))
])
y_score = np.concatenate([scores_sig, scores_bkg])
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Plot
fig, ax = plt.subplots(1, 1)
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle='--')  # random classifier line
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(True)


fig, ax = plt.subplots(1, 1)
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle='--')  # random classifier line
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 0.02)
ax.set_ylim(0, 0.1)
ax.legend()
ax.grid(True)

# %%
fig, ax = plt.subplots(1, 1)
for nn_t in [0, 0.5, 0.7]:
    ax.hist(df_data.dijet_mass[df_data.PNN>nn_t], bins=np.linspace(50, 300, 51), histtype='step', label=f"Data NN > {nn_t:1}", density=False)
ax.legend()
# %%
#import sys
#sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/")
#from plotFeatures import plotNormalizedFeatures
#plotNormalizedFeatures([df_ggH, df_data], outFile="/t3home/gcelotto/ggHbb/documentation/run3/ggH_vs_Data_run3.png", legendLabels=["ggH", "Data"], colors=["red", "blue"], figsize=(30,90))

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df_ggH.jet1_pt, bins=np.linspace(0, 200, 51), histtype='step', label="ggH", density=True)
ax.hist(df_data.jet1_pt, bins=np.linspace(0, 200, 51), histtype='step', label="data", density=True)
ax.set_xlabel("jet1_pt")
ax.legend()
# %%
