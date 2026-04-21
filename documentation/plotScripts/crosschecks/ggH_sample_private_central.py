# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import pandas as pd
from functions import  *
# %%

def plot_with_ratio(ax_top, ax_ratio, x_c, x_v1, x_v2, bins, range_, xlabel):

    # --- histograms ---
    h_c, edges = np.histogram(x_c, bins=bins, range=range_, density=True)
    h_v1, _ = np.histogram(x_v1, bins=bins, range=range_, density=True)
    h_v2, _ = np.histogram(x_v2, bins=bins, range=range_, density=True)

    centers = 0.5 * (edges[1:] + edges[:-1])

    # --- top panel ---
    ax_top.step(centers, h_c, where='mid', label=f"Central ({len(x_c)})")
    ax_top.step(centers, h_v1, where='mid', label=f"Private V1 ({len(x_v1)})", color="C1")
    ax_top.step(centers, h_v2, where='mid', label=f"Private V2 ({len(x_v2)})", color="C2")

    ax_top.set_ylabel("Density")
    ax_top.legend()

    # --- ratio (avoid division by zero) ---
    mask = h_c > 0
    ratio_v1 = np.zeros_like(h_c)
    ratio_v2 = np.zeros_like(h_c)

    ratio_v1[mask] = h_v1[mask] / h_c[mask]
    ratio_v2[mask] = h_v2[mask] / h_c[mask]

    # --- ratio panel ---
    ax_ratio.step(centers, ratio_v1, where='mid', label="V1/Central", color="C1")
    ax_ratio.step(centers, ratio_v2, where='mid', label="V2/Central", color="C2")

    ax_ratio.axhline(1.0, linestyle='--')
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylim(0.5, 1.5)


from functions import getCommonFilters

df_central = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/others", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=9))
# %%
df_v1 = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/v2", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=9))
df_v2 = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/v9992", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=9))
# %%

from scipy.stats import ks_2samp
def ks_test(x, y):
    stat, pval = ks_2samp(x, y)
    return stat, pval

# %%
import numpy as np


fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex='col',
                       gridspec_kw={'height_ratios': [3, 1]})

plot_with_ratio(
    ax[0,0], ax[1,0],
    df_central["dijet_pt"], df_v1["dijet_pt"], df_v2["dijet_pt"],
    bins=50, range_=(60,120),
    xlabel="Dijet pT [GeV]"
)

plot_with_ratio(
    ax[0,1], ax[1,1],
    df_central["jet1_muon_pt"], df_v1["jet1_muon_pt"], df_v2["jet1_muon_pt"],
    bins=50, range_=(9,40),
    xlabel="Muon in Jet p$_T$ [GeV]"
)

plot_with_ratio(
    ax[0,2], ax[1,2],
    df_central["dijet_mass"], df_v1["dijet_mass"], df_v2["dijet_mass"],
    bins=50, range_=(70,190),
    xlabel="Dijet Mass [GeV]"
)

fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex='col',
                       gridspec_kw={'height_ratios': [3, 1]})

plot_with_ratio(
    ax[0,0], ax[1,0],
    df_central["jet1_pt"], df_v1["jet1_pt"], df_v2["jet1_pt"],
    bins=50, range_=(20,120),
    xlabel="Jet1 pT [GeV]"
)

plot_with_ratio(
    ax[0,1], ax[1,1],
    df_central["jet2_pt"], df_v1["jet2_pt"], df_v2["jet2_pt"],
    bins=50, range_=(20,100),
    xlabel="Jet2 pT [GeV]"
)

plot_with_ratio(
    ax[0,2], ax[1,2],
    df_central["dijet_dPhi"], df_v1["dijet_dPhi"], df_v2["dijet_dPhi"],
    bins=50, range_=(0,3.14),
    xlabel="Dijet dPhi]"
)
# %%










# %%

df_central = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/others", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=12))

df_v1 = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/v2", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=12))
df_v2 = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/v9992", 
                             filters=getCommonFilters(btagWP="M", cutDijet=False, boosted=12))
# %%
fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex='col',
                       gridspec_kw={'height_ratios': [3, 1]})

plot_with_ratio(
    ax[0,0], ax[1,0],
    df_central["dijet_pt"], df_v1["dijet_pt"], df_v2["dijet_pt"],
    bins=50, range_=(120,300),
    xlabel="Dijet pT [GeV]"
)

plot_with_ratio(
    ax[0,1], ax[1,1],
    df_central["jet1_muon_pt"], df_v1["jet1_muon_pt"], df_v2["jet1_muon_pt"],
    bins=50, range_=(9,40),
    xlabel="Muon in Jet p$_T$ [GeV]"
)

plot_with_ratio(
    ax[0,2], ax[1,2],
    df_central["dijet_mass"], df_v1["dijet_mass"], df_v2["dijet_mass"],
    bins=50, range_=(70,190),
    xlabel="Dijet Mass [GeV]"
)

fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex='col',
                       gridspec_kw={'height_ratios': [3, 1]})

plot_with_ratio(
    ax[0,0], ax[1,0],
    df_central["jet1_pt"], df_v1["jet1_pt"], df_v2["jet1_pt"],
    bins=50, range_=(20,120),
    xlabel="Jet1 pT [GeV]"
)

plot_with_ratio(
    ax[0,1], ax[1,1],
    df_central["jet2_pt"], df_v1["jet2_pt"], df_v2["jet2_pt"],
    bins=50, range_=(20,100),
    xlabel="Jet2 pT [GeV]"
)

plot_with_ratio(
    ax[0,2], ax[1,2],
    df_central["dijet_dPhi"], df_v1["dijet_dPhi"], df_v2["dijet_dPhi"],
    bins=50, range_=(0,3.14),
    xlabel="Dijet dPhi"
)
# %%
