# %%
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import mplhep as hep
hep.style.use("CMS")
# %%
pN = 4
dfs, sumw = loadMultiParquet_v2(paths=[pN], nMCs=-1, returnNumEventsTotal=True, filters=getCommonFilters(btagTight=True))
dfProcesses = getDfProcesses_v2()[0].iloc[pN]
df_MC = dfs[0][dfs[0].dijet_pt>100]
# %%
dfs, lumi = loadMultiParquet_Data_new(dataTaking=[17], nReals=-1, columns=None, training=True, filters=getCommonFilters(btagTight=True))
df_Data=dfs[0][dfs[0].dijet_pt>100]

# %%
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten to simplify indexing

# Common settings
bins = np.linspace(0, 100, 51)
weights = df_MC.sf * df_MC.PU_SF * df_MC.btag_central

# Plot 1: Dimuon pT
axes[0].hist(df_Data.dimuonZZ_pt, bins=bins, density=True, label='Data', histtype='step')
axes[0].hist(df_MC.dimuonZZ_pt, bins=bins, density=True, label='MC', weights=weights, histtype='step')
axes[0].set_xlabel("Dimuon pT [GeV]")
axes[0].legend()

# Plot 2: Muon1 pT
axes[1].hist(df_Data.muonZ1_pt, bins=bins, density=True, label='Data', histtype='step')
axes[1].hist(df_MC.muonZ1_pt, bins=bins, density=True, label='MC', weights=weights, histtype='step')
axes[1].set_xlabel("Muon1 pT [GeV]")
axes[1].legend()

# Plot 3: Muon2 pT
axes[2].hist(df_Data.muonZ2_pt, bins=bins, density=True, label='Data', histtype='step')
axes[2].hist(df_MC.muonZ2_pt, bins=bins, density=True, label='MC', weights=weights, histtype='step')
axes[2].set_xlabel("Muon2 pT [GeV]")
axes[2].legend()

# Plot 4: Dimuon mass
axes[3].hist(df_Data.dimuonZZ_mass, bins=bins, density=True, label='Data', histtype='step')
axes[3].hist(df_MC.dimuonZZ_mass, bins=bins, density=True, label='MC', weights=weights, histtype='step')
axes[3].set_xlabel("Dimuon mass [GeV]")
axes[3].legend()

# Adjust layout
plt.tight_layout()
plt.show()























# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
min_mumu, max_mumu = 1, 100

bins_ll = np.linspace(0, 200, 51)
bins_jj = np.linspace(50, 200, 51)

ax[0].hist(df_MC.dimuonZZ_mass, bins=bins_ll)
ax[0].set_yscale('log')
ax[0].set_ylim(ax[0].get_ylim())
ax[0].vlines(x=[min_mumu, max_mumu], ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1], color='C1')
ax[0].set_xlabel("Dimuon mass [GeV]")
c=ax[1].hist(df_MC.dijet_mass, bins=bins_jj, histtype='step', weights=(df_MC.genWeight*df_MC.sf*df_MC.PU_SF) *1000 * dfProcesses.xsection *41.6/sumw[0], label='Inclusive')[0]
mZ_mumu = (df_MC.dimuonZZ_mass>min_mumu) & (df_MC.dimuonZZ_mass<max_mumu)
ax[1].hist(df_MC.dijet_mass[mZ_mumu], bins=bins_jj, histtype='step', weights=(df_MC.genWeight*df_MC.sf*df_MC.PU_SF)[mZ_mumu] *1000 * dfProcesses.xsection *41.6/sumw[0], label='%d < $m_{\mu\mu}$ < %d'%(min_mumu, max_mumu), color='C1')
ax[1].set_ylabel("Counts per %.1f GeV"%(bins_jj[1]-bins_jj[0]))
#ax[1].set_yscale('log')
#ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*20)
ax[1].legend()
ax[1].set_xlabel("Dijet mass [GeV]")
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
min_ee, max_ee = 1, 100

ax[0].hist(df_MC.diele_mass, bins=bins_ll)
ax[0].set_yscale('log')
ax[0].set_ylim(ax[0].get_ylim())
ax[0].vlines(x=[min_ee, max_ee], ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1], color='C1')
ax[0].set_xlabel("Dielectron mass [GeV]")
ax[1].hist(df_MC.dijet_mass, bins=bins_jj, histtype='step', weights=(df_MC.genWeight*df_MC.sf*df_MC.PU_SF) *1000 * dfProcesses.xsection *41.6/sumw[0], label='Inclusive')
ax[1].hist(df_MC.dijet_mass[(df_MC.diele_mass>min_ee) & (df_MC.diele_mass<max_ee)], bins=bins_jj, histtype='step', weights=df_MC.genWeight[(df_MC.diele_mass>min_ee) & (df_MC.diele_mass<max_ee)] *1000 * dfProcesses.xsection *41.6/sumw[0], label='%d < $m_{ee}$ < %d'%(min_ee, max_ee), color='C1')
#ax[1].set_yscale('log')
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1])
ax[1].set_ylabel("Counts per %.1f GeV"%(bins_jj[1]-bins_jj[0]))
ax[1].legend()
ax[1].set_xlabel("Dijet mass [GeV]")
# %%
