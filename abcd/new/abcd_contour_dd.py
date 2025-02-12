# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses, cut, getDfProcesses_v2
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
import pickle
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from helpersABCD.abcd_maker_v2 import ABCD
import argparse
# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    parser.add_argument("-dd", "--doubleDisco", type=bool, help="Single Disco (False) or double Disco (True). If false use jet1btag as variable", default=False)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
    dd = args.doubleDisco
except:
    print("Interactive mode")
    modelName = "Jan19_900p0"
    dd = True
outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")

# %%
dfs = []
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
# %%
dfsMC = []
isMCList = [0,
            1, 
            2,3, 4,
            5,6,7,8, 9,10,
            11,12,13,
            14,15,16,17,18,
            19,20,21, 22, 35]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsMC.append(df)
# %%
dfsData = []
isDataList = [
            0,
            1, 
            #2
            ]

lumis = []
for idx, p in enumerate(dfProcessesData.process):
    if idx not in isDataList:
        continue
    df = pd.read_parquet(df_folder+"/dataframes%s%s_%s.parquet"%("_dd_" if dd else "", p, modelName))
    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi%s%s_%s.npy"%("_dd_", p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight =dfsMC[idx].weight*lumi


# %%
dfZ = []
for idx,df in enumerate(dfsMC):
    print(isMCList[idx])
    if (isMCList[idx] == 1) | (isMCList[idx] == 19) | (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 35):
        dfZ.append(df)
dfZ=pd.concat(dfZ)


# %%

# do hist or contour in the abcd regions
dfW=[]
for idx,df in enumerate(dfsMC):
    if (isMCList[idx] >= 14) & (isMCList[idx] < 19):
        dfW.append(df)
dfW=pd.concat(dfW)


df_tt=[]
for idx,df in enumerate(dfsMC):
    if (isMCList[idx] >= 11) & (isMCList[idx] < 14):
        df_tt.append(df)
df_tt=pd.concat(df_tt)


df_st=[]
for idx,df in enumerate(dfsMC):
    if (isMCList[idx] >= 5) & (isMCList[idx] < 11):
        df_st.append(df)
df_st=pd.concat(df_st)

df_VV=[]
for idx,df in enumerate(dfsMC):
    if (isMCList[idx] >= 2) & (isMCList[idx] <= 4 ):
        df_VV.append(df)
df_VV=pd.concat(df_VV)

# %%
x1='PNN1'
x2='PNN2'
t11=0.5
t12=0.5
t22=0.5
t21=0.5
# %%
df = dfsData[0].copy()
mA      = (df[x1]<t11 ) & (df[x2]>t22 ) 
mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 
mC      = (df[x1]<t11 ) & (df[x2]<t21 ) 
mD      = (df[x1]>t12 ) & (df[x2]<t21 ) 



print("Region A : ", np.sum(df.weight[mA])/df.weight.sum())
print("Region B : ", np.sum(df.weight[mB])/df.weight.sum())
print("Region C : ", np.sum(df.weight[mC])/df.weight.sum())
print("Region D : ", np.sum(df.weight[mD])/df.weight.sum())

# %%
from scipy.stats import gaussian_kde
def compute_kde(df, grid_size=100):
    x = df[x1].values  # Column 'x1' for x-axis
    y = df[x2].values  # Column 'x2' for y-axis
    weights = np.where(df['weight'].values>0, df['weight'].values, 0)

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

from matplotlib.lines import Line2D  # 
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='cyan', lw=2, label='H')
]

df = dfsMC[0]
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


ax_main.set_xlim(0., 1)
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


df = dfZ
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


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/Z.png", bbox_inches='tight')

# %%

# Contour for W

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='red', lw=2, label='W'),

]

df = dfW
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Reds')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='red', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='red', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/W.png", bbox_inches='tight')
# %%

# Contour for tt

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='green', lw=2, label='tt'),
]


df = df_tt
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Greens')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='green', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='green', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/ttbar.png", bbox_inches='tight')
# %%

# Contour for Data

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='black', lw=2, label='Data'),
]


df = dfsData[0]
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Grays')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='black', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='black', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/Data.png", bbox_inches='tight')
# %%


# Contour for VV

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='orange', lw=2, label='VV')
]


df = df_VV
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='copper')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='orange', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='orange', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/VV.png", bbox_inches='tight')

# %%


# Contour for ST

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='orange', lw=2, label='ST')
]


df = df_st
X, Y, Z, x, y  = compute_kde(df.head(10000))
contour = ax_main.contour(X, Y, Z, levels=10, cmap='Purples')  # Plot contour for df4
ax_main.clabel(contour, inline=True, fontsize=8)


ax_top = fig.add_subplot(gs[0, 1:4], sharex=ax_main)
ax_top.hist(x, bins=30, weights=df['weight'].head(10000), color='purple', alpha=0.7)
ax_top.set_ylabel('Count')
ax_top.tick_params(axis="x", labelbottom=False)

ax_left = fig.add_subplot(gs[1:4, 4], sharey=ax_main)
ax_left.hist(y, bins=30, weights=df['weight'].head(10000), orientation='horizontal', color='purple', alpha=0.7)
ax_left.set_xlabel('Count')
ax_left.tick_params(axis="y", labelleft=False)


ax_main.set_xlim(0., 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/ST.png", bbox_inches='tight')





# %%
