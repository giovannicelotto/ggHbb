# %%
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
# %%
hep.style.use("CMS")
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"
modelName = "Jan21_3_50p0"
df = pd.read_parquet(folder + modelName + "/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
df_VBF = pd.read_parquet(folder + modelName + "/df_VBFHToBB_Jan21_3_50p0.parquet")
# %%
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(1, 1)
bins_nn, bins_dijet_pt = np.linspace(0, 1, 101), np.linspace(80, 500, 101)
ax.hist2d(df.PNN, df.dijet_pt, bins=(bins_nn, bins_dijet_pt), cmap="viridis", norm=LogNorm())
ax.set_xlabel("NN")
ax.set_ylabel("dijet pt")

ax_top = inset_axes(
    ax,
    width="100%",
    height="40%",
    loc="upper center",
    bbox_to_anchor=(0, 0.85, 1, 0.25),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# right marginal
ax_right = inset_axes(
    ax,
    width="40%",
    height="100%",
    loc="center right",
    bbox_to_anchor=(0.85, 0, 0.25, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# MC marginals
ax_top.hist(df.PNN, bins=bins_nn, density=True, color='red', alpha=0.4)
ax_top.hist(df.PNN, bins=bins_nn, density=True, color='red', linewidth=3, histtype='step')



ax_right.hist(df.dijet_pt, bins=bins_dijet_pt, density=True, color='red', alpha=0.4, orientation="horizontal")
ax_right.hist(df.dijet_pt, bins=bins_dijet_pt, density=True, color='red', linewidth=3, histtype='step', orientation="horizontal")

# Data marginals
#ax_top.set_yscale('log')
#ax_right.set_xscale('log')

ax_right.set_ylim(ax.get_ylim())
ax_top.set_xlim(ax.get_xlim())
#ax_right.set_xlim(ax.get_xlim())
#ax_top.axis("off")
ax_right.set_xticks([])
ax_right.set_yticks([])
ax_top.set_xticks([])
ax_top.set_yticks([])
#ax_right.axis("off")

# %%
dfs = []
import yaml
for cat in [0,1,2,3,10,11]:
    yaml_file=f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{cat}.yml"
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

        df_=df.copy().query(cfg["cuts_string"])
        print("Appended")
        dfs.append(df_)


dfs_VBF = []
import yaml
for cat in [0,1,2,3,10]:
    yaml_file=f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{cat}.yml"
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

        df_=df_VBF.copy().query(cfg["cuts_string"])
        print("Appended")
        dfs_VBF.append(df_)
bins_dijet_pt = np.linspace(80, 800, 101)
fig, ax  = plt.subplots(3, 2)
for i, cat in enumerate([0,1,2,3,10]):
    ax[i//2][i%2].hist(np.clip(dfs[i].dijet_pt, bins_dijet_pt[0], bins_dijet_pt[-1]), bins=bins_dijet_pt, weights=dfs[i].weight, label="ggF")
    ax[i//2][i%2].set_xlabel("dijet pt")
    ax[i//2][i%2].set_ylabel("Events")
    ax[i//2][i%2].text(x=0.95, y=0.9, s=f"cat {cat}", transform=ax[i//2][i%2].transAxes, ha="right", va="center")
    ax[i//2][i%2].set_xlim(bins_dijet_pt[0],bins_dijet_pt[-1])

    ax[i//2][i%2].hist(np.clip(dfs_VBF[i].dijet_pt, bins_dijet_pt[0], bins_dijet_pt[-1]), bins=bins_dijet_pt, weights=dfs_VBF[i].weight, label='VBF', histtype='step')
    ax[i//2][i%2].legend()

# %%
dfs[0][['jet1_eta', 'jet2_eta', 'jet1_phi', 'jet2_phi']]
dfs[0]['dPhi'] = np.abs(dfs[0]['jet1_phi'] - dfs[0]['jet2_phi'])
dfs[0]['dEta'] = np.abs(dfs[0]['jet1_eta'] - dfs[0]['jet2_eta'])
dfs[0]['dR'] = np.sqrt(dfs[0]['dPhi']**2 + dfs[0]['dEta']**2)
dfs[0][['jet1_eta', 'jet2_eta', 'jet1_phi', 'jet2_phi', 'dR']]

fig, ax = plt.subplots(1, 1)
ax.hist(dfs[0].dR, bins=np.linspace(0, 1, 101), weights=dfs[0].weight)
ax.set_xlabel("dR")
# %%
from matplotlib.patches import Circle

fig, ax = plt.subplots(1, 1)

ev = 0

eta1 = dfs[0]['jet1_eta'].iloc[ev]
phi1 = dfs[0]['jet1_phi'].iloc[ev]

eta2 = dfs[0]['jet2_eta'].iloc[ev]
phi2 = dfs[0]['jet2_phi'].iloc[ev]

ax.scatter(eta1, phi1, s=50, label='Jet 1')
ax.scatter(eta2, phi2, s=50, label='Jet 2')

R = 0.4

circle1 = Circle((eta1, phi1), R, fill=False)
circle2 = Circle((eta2, phi2), R, fill=False)

ax.add_patch(circle1)
ax.add_patch(circle2)

ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\phi$')
ax.legend()

plt.show()

# %%
