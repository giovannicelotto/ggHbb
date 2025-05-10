# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from functions import getDfProcesses_v2
import pickle
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D  # 
import matplotlib.gridspec as gridspec
import seaborn as sns
# %%
dataFrames_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/Apr17_100p0"
dfProcess = getDfProcesses_v2()[0]
Z_processes = [1, 3, 4, 19, 20, 21, 22, 35]
H_processes = [0,37]

Z_names = []
for processName in dfProcess.process.iloc[Z_processes]:
    Z_names.append(dataFrames_folder+"/df_dd_%s_Apr17_100p0.parquet"%(processName))

df_Z = pd.read_parquet(Z_names)

H_names = []
for processName in dfProcess.process.iloc[H_processes]:
    H_names.append(dataFrames_folder+"/df_dd_%s_Apr17_100p0.parquet"%(processName))

df_H = pd.read_parquet(H_names)


df_data = pd.read_parquet(dataFrames_folder+"/dataframes_dd_Data1A_Apr17_100p0.parquet")
# %%
x1 = 'PNN1'
x2 = 'PNN2'
t11 = 0.56
t12 = 0.56
t21 =0.58
t22 = 0.58
xx = 'dijet_mass'
# %%
def computeSig(df_data, df_H, t11, t22):
    t12 = t11
    t21 = t22
    mA_data      = (df_data[x1]<t11 ) & (df_data[x2]>t22 ) 
    mB_data      = (df_data[x1]>t12 ) & (df_data[x2]>t22 ) 
    mC_data      = (df_data[x1]<t11 ) & (df_data[x2]<t21 ) 
    mD_data      = (df_data[x1]>t12 ) & (df_data[x2]<t21 ) 

    N_A = np.sum(df_data.weight[mA_data])/df_data.weight.sum()
    N_B = np.sum(df_data.weight[mB_data])/df_data.weight.sum()
    N_C = np.sum(df_data.weight[mC_data])/df_data.weight.sum()
    N_D = np.sum(df_data.weight[mD_data])/df_data.weight.sum()

    mA_H      = (df_H[x1]<t11 ) & (df_H[x2]>t22 ) 
    mB_H      = (df_H[x1]>t12 ) & (df_H[x2]>t22 ) 
    mC_H      = (df_H[x1]<t11 ) & (df_H[x2]<t21 ) 
    mD_H      = (df_H[x1]>t12 ) & (df_H[x2]<t21 ) 

    effSig = np.sum(df_H.weight[mB_H])/df_H.weight.sum()
    errorQCD = N_A * N_D / (N_C+1e-12) * np.sqrt(1/(N_A+1e-12) + 1/(N_C+1e-12) + 1/(N_D+1e-12))

    if N_C == 0:
        return 0, 0, 0

    return effSig,  errorQCD/(N_B+1e-12), effSig/(errorQCD+1e-12)


t1_space = np.linspace(0.1, 0.9, 20)
t2_space = np.linspace(0.1, 0.9, 20)
sig_results = np.empty((len(t1_space), len(t2_space)))
effH_results = np.empty((len(t1_space), len(t2_space)))
errData_results = np.empty((len(t1_space), len(t2_space)))

for idx, t1 in enumerate(t1_space):
    for jdx, t2 in enumerate(t2_space):
        print(t1, t2)
        effH_results[idx, jdx], errData_results[idx, jdx], sig_results[idx, jdx] = computeSig(df_data=df_data, df_H=df_H, t11=t1 ,t22=t2)
        print(effH_results[idx, jdx], errData_results[idx, jdx], sig_results[idx, jdx])
# %%

from matplotlib.colors import LogNorm
# Assuming sig_results, t1_space, t2_space already exist from your loop

fig, ax = plt.subplots(figsize=(8, 6))
sig_results = np.where(sig_results > 0, sig_results, 1e-10)
sig_results = np.where(errData_results>1e-10, sig_results, 1e-10)
# Create the heatmap
sns.heatmap(sig_results, 
            xticklabels=np.round(t2_space, 2), 
            yticklabels=np.round(t1_space, 2), 
            cmap="viridis", 
            norm=LogNorm(vmin=sig_results.min(), vmax=sig_results.max()),
            cbar_kws={'label': 'Sig Value'},
            ax=ax)

# Set labels and title
ax.set_xlabel('t2')
ax.set_ylabel('t1')
ax.set_title('Sig Results over t1/t2 space')

# Rotate ticks for clarity
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig.tight_layout()
plt.show()


# %%
mA_data      = (df_data[x1]<t11 ) & (df_data[x2]>t22 ) 
mB_data      = (df_data[x1]>t12 ) & (df_data[x2]>t22 ) 
mC_data      = (df_data[x1]<t11 ) & (df_data[x2]<t21 ) 
mD_data      = (df_data[x1]>t12 ) & (df_data[x2]<t21 ) 

mA_H      = (df_H[x1]<t11 ) & (df_H[x2]>t22 ) 
mB_H      = (df_H[x1]>t12 ) & (df_H[x2]>t22 ) 
mC_H      = (df_H[x1]<t11 ) & (df_H[x2]<t21 ) 
mD_H      = (df_H[x1]>t12 ) & (df_H[x2]<t21 ) 


print("Efficiency   | Data | Higgs  |")
print("Region A     | %.2f | %.2f  |"%(np.sum(df_data.weight[mA_data])/df_data.weight.sum() * 100, np.sum(df_H.weight[mA_H])/df_H.weight.sum()*100))
print("Region B     | %.2f | %.2f  |"%(np.sum(df_data.weight[mB_data])/df_data.weight.sum() * 100, np.sum(df_H.weight[mB_H])/df_H.weight.sum()*100))
print("Region C     | %.2f | %.2f  |"%(np.sum(df_data.weight[mC_data])/df_data.weight.sum() * 100, np.sum(df_H.weight[mC_H])/df_H.weight.sum()*100))
print("Region D     | %.2f | %.2f  |"%(np.sum(df_data.weight[mD_data])/df_data.weight.sum() * 100, np.sum(df_H.weight[mD_H])/df_H.weight.sum()*100))

# %%

def compute_kde(df, grid_size=100):
    x = df[x1].values  # Column 'x1' for x-axis
    y = df[x2].values  # Column 'x2' for y-axis
    weights = df['weight'].values

    # Create a grid over which to evaluate the KDE
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Perform KDE (gaussian_kde estimates the density)
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kde = gaussian_kde(values, weights=weights)
    Z = np.reshape(kde(positions).T, X.shape)  # Evaluate KDE on grid and reshape
    return X, Y, Z, x, y

def compute_2d_hist(df, grid_size=50):
    x = df[x1].values
    y = df[x2].values
    weights=df.weight.values

    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=grid_size, weights=weights)

    # Get the grid centers
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
    return X, Y, H
# %%

# Contour for Higgs


fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='cyan', lw=2, label='H')
]

df = df_H
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Blues')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='cyan', alpha=0.7)
ax_top.set_ylabel('Counts')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='cyan', alpha=0.7)
ax_left.set_xlabel('Counts')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0.2783, 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/Higgs.png", bbox_inches='tight')


# %%

# Contour for Z
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Z',alpha=0.5),
]


df = df_Z
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Blues')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='blue', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='blue', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0.2783, 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/Z.png", bbox_inches='tight')

# %%

# Contour for Data

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='black', lw=2, label='Data'),
]


X, Y, Z, x, y  = compute_kde(df_data.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Grays')  # Plot contour for df_data4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df_data['weight'].head(10000), color='black', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df_data['weight'].head(10000), orientation='horizontal', color='black', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0.2783, 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/Data.png", bbox_inches='tight')
# %%
