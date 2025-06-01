# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, getDfProcesses_v2, getCommonFilters
import mplhep as hep
hep.style.use("CMS")
# %%
dfProcessMC = getDfProcesses_v2()[0]
isMCList = [25,26,27,28,
            29,30,31,32,33,34]
dfsMC, sumw = loadMultiParquet_v2(paths=isMCList, returnNumEventsTotal=True, nMCs=-1, filters=getCommonFilters(btagTight=True))
# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['weight']=df.genWeight * df.btag_central * df.PU_SF * df.sf * dfProcessMC.xsection.iloc[isMCList].values[idx]
# %%
dfMC  = pd.concat(dfsMC)
fig, ax = plt.subplots(1, 1)
bins = np.arange(-1,7,1)
c1 = np.histogram(dfMC.jet1_genHadronFlavour, bins=bins, weights=dfMC.weight)[0]
c1 = c1 / np.sum(c1)
ax.hist(bins[:-1], bins=bins, weights=c1, histtype='step', label='MuonJet', linewidth=3)

# Jet2
c2 = np.histogram(dfMC.jet2_genHadronFlavour, bins=bins, weights=dfMC.weight)[0]
c2 = c2 / np.sum(c2)
ax.hist(bins[:-1], bins=bins, weights=c2, histtype='step', label='Jet2', linewidth=3)

# Jet3
c3 = np.histogram(dfMC.jet3_genHadronFlavour, bins=bins, weights=dfMC.weight)[0]
c3 = c3 / np.sum(c3)
ax.hist(bins[:-1], bins=bins, weights=c3, histtype='step', label='Jet3', linewidth=3)

# Get indices for flavours 0, 4, 5 in bins
flavours = [0, 4, 5]
indices = [np.where(bins[:-1] == f)[0][0] for f in flavours]

# Compose text
text = (
    f"Flavour Probabilities:\n"
    f"MuonJet - udsg: {c1[indices[0]]:.3f}, c: {c1[indices[1]]:.3f}, b: {c1[indices[2]]:.3f}\n"
    f"Jet2    - udsg: {c2[indices[0]]:.3f}, c: {c2[indices[1]]:.3f}, b: {c2[indices[2]]:.3f}\n"
    f"Jet3    - udsg: {c3[indices[0]]:.3f}, c: {c3[indices[1]]:.3f}, b: {c3[indices[2]]:.3f}"
)

# Add text to plot
ax.text(
    0.4, 0.7, text,
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=20, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

ax.set_title("QCD_MuEnriched")
ax.set_xlabel('genHadronFlavour')
ax.set_ylabel('Probability')
ax.legend()
tick_positions = [-0.5, 0.5, 4.5, 5.5]
tick_labels = []

# Define label mapping
label_map = {-0.5: 'No Jet3', 0.5: 'udsg', 4.5: 'c', 5.5: 'b'}

for pos in tick_positions:
    tick_labels.append(label_map.get(pos, ''))  # empty label if not in label_map

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)
ax.tick_params(labelsize=40)
# %%




















# ggHbb
isMCList = [0]
dfsMC, sumw = loadMultiParquet_v2(paths=isMCList, returnNumEventsTotal=True, nMCs=-1, filters=getCommonFilters(btagTight=True))

for idx, df in enumerate(dfsMC):
    dfsMC[idx]['weight']=df.genWeight * df.btag_central * df.PU_SF * df.sf * dfProcessMC.xsection.iloc[isMCList].values[idx]

dfMC_Higgs  = pd.concat(dfsMC)
# %%
fig, ax = plt.subplots(1, 1)
bins = np.arange(-1.,7,1)
c1 = np.histogram(dfMC_Higgs.jet1_genHadronFlavour, bins=bins, weights=dfMC_Higgs.weight)[0]
c1 = c1 / np.sum(c1)
ax.hist(bins[:-1], bins=bins, weights=c1, histtype='step', label='MuonJet', linewidth=3)

# Jet2
c2 = np.histogram(dfMC_Higgs.jet2_genHadronFlavour, bins=bins, weights=dfMC_Higgs.weight)[0]
c2 = c2 / np.sum(c2)
ax.hist(bins[:-1], bins=bins, weights=c2, histtype='step', label='2ndJet', linewidth=3)

# Jet3
c3 = np.histogram(dfMC_Higgs.jet3_genHadronFlavour, bins=bins, weights=dfMC_Higgs.weight)[0]
c3 = c3 / np.sum(c3)
ax.hist(bins[:-1], bins=bins, weights=c3, histtype='step', label='3rdJet', linewidth=3)

# Get indices for flavours 0, 4, 5 in bins
flavours = [0, 4, 5]
indices = [np.where(bins[:-1] == f)[0][0] for f in flavours]

# Compose text
text = (
    f"Flavour Probabilities:\n"
    f"MuonJet - udsg: {c1[indices[0]]:.3f}, c: {c1[indices[1]]:.3f}, b: {c1[indices[2]]:.3f}\n"
    f"Jet2    - udsg: {c2[indices[0]]:.3f}, c: {c2[indices[1]]:.3f}, b: {c2[indices[2]]:.3f}\n"
    f"Jet3    - udsg: {c3[indices[0]]:.3f}, c: {c3[indices[1]]:.3f}, b: {c3[indices[2]]:.3f}"
)


# Add text to plot
ax.text(
    0.4, 0.7, text,
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=20, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)
ax.set_title("ggHbb")
ax.set_xlabel('genHadronFlavour')
ax.set_ylabel('Probability')
ax.legend()
tick_positions = [-0.5, 0.5, 4.5, 5.5]
tick_labels = []

# Define label mapping
label_map = {-0.5: 'No Jet3', 0.5: 'udsg', 4.5: 'c', 5.5: 'b'}

for pos in tick_positions:
    tick_labels.append(label_map.get(pos, ''))  # empty label if not in label_map

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)
ax.tick_params(labelsize=40)
# %%
# %%



fig, ax = plt.subplots(1, 1)
ax.hist(dfMC.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 100), density=True, histtype='step', label='QCD')
ax.hist(dfMC_Higgs.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 100), density=True, histtype='step', label='Higgs')
ax.legend()

fig, ax = plt.subplots(1, 1)
ax.hist(dfMC.jet2_btagDeepFlavB, bins=np.linspace(.99, 1, 100), density=True, histtype='step', label='QCD')
ax.hist(dfMC_Higgs.jet2_btagDeepFlavB, bins=np.linspace(.99, 1, 100), density=True, histtype='step', label='Higgs')
ax.legend()
ax.tick_params(labelsize=20)
# %%
