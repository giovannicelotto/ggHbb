# %%
# This needs to be done before trigger
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

from hist import Hist
from functions import loadMultiParquet_v2, getDfProcesses_v2

# %%
dfProcesses = getDfProcesses_v2()[0]

# %%
dfs, sumw = loadMultiParquet_v2(
    paths=[0, 37],
    nMCs=[-1, -1],
    returnNumEventsTotal=True,
    filters=None
)

# %%
# Apply weights
for df, s in zip(dfs, sumw):
    df["weight"] = (
        df.genWeight / s
        * 1000
        * 41.6
        * dfProcesses.xsection.iloc[0]
    )

# %%
# Observable configuration (ordered)
observables = [
    {
        "name": "dijet_pt",
        "label": r"$p_T$ [GeV]",
        "bins": np.linspace(0, 250, 51),
        "filename": "Higgs_Pt.png",
    },
    {
        "name": "dijet_eta",
        "label": r"$\eta$",
        "bins": np.linspace(-5, 5, 51),
        "filename": "Higgs_Eta.png",
    },
    {
        "name": "dijet_phi",
        "label": r"$\phi$",
        "bins": np.linspace(-np.pi, np.pi, 51),
        "filename": "Higgs_Phi.png",
    },
    {
        "name": "dijet_mass",
        "label": r"$m_{jj}$ [GeV]",
        "bins": np.linspace(0, 300, 61),
        "filename": "Higgs_Mass.png",
    },
]

# %%
outdir = "/t3home/gcelotto/ggHbb/documentation/plots/"

for obs in observables:

    h_powheg = Hist.new.Var(obs["bins"], name=obs["label"]).Weight()
    h_minlo  = Hist.new.Var(obs["bins"], name=obs["label"]).Weight()

    h_powheg.fill(
        dfs[0][obs["name"]],
        weight=dfs[0]["weight"],
    )
    h_minlo.fill(
        dfs[1][obs["name"]],
        weight=dfs[1]["weight"],
    )

    fig, ax = plt.subplots(1, 1)

    h_powheg.plot(ax=ax, label="POWHEG")
    h_minlo.plot(ax=ax, label="MINLO")

    ax.set_ylabel(f"Events / {obs['bins'][1] - obs['bins'][0]:.1f}")
    ax.legend()

    fig.savefig(f"{outdir}/{obs['filename']}")
    plt.close(fig)

# %%
