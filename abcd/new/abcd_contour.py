# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
# %%

nReal, nMC = 30, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            #39    # Data2A
]

dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    print(p)
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/others/*.parquet"%p))


# %%
predictionsFileNumbers = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l = []
    for fileName in predictionsFileNames[idx]:
        print
        fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
        l.append(int(fn))

    predictionsFileNumbers.append(l)
# %%
paths = list(dfProcesses.flatPath[isMCList])
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                      columns=['sf', 'dijet_mass',# 'dijet_pt', 'jet1_pt',
                                                               #'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                               #'jet2_eta', 'dijet_dR',
                                                                #'jet3_mass',
                                                                #'Pileup_nTrueInt',
                                                               'jet2_btagDeepFlavB', #'dijet_cs', 'nJets_20GeV',
                                                               'jet1_btagDeepFlavB', 'PU_SF'],
                                                               returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                               returnFileNumberList=True)
if isMCList[-1]==39:
    nReal = nReal *2
    print("Duplicating nReal")
# %%
preds = []
predictionsFileNamesNew = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l =[]
    for fileName in predictionsFileNames[idx]:
        print(fileName)
        fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
        if fn in fileNumberList[idx]:
            l.append(fileName)
    predictionsFileNamesNew.append(l)
    
    print(len(predictionsFileNamesNew[idx]), " files for process")
    df = pd.read_parquet(predictionsFileNamesNew[idx])
    preds.append(df)


# given the fn load the data


# preprocess 
dfs = preprocessMultiClass(dfs=dfs)
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])

# %%
for idx, df in enumerate(dfs):
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]

# make uinque data columns
if isMCList[-1]==39:
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
    dfs = dfs[:-1]       # remove the last element (data2a)
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))
# %%
#for idx, df in enumerate(dfs):
#    dfs[idx]['dijet_cs_abs'] = 1-abs(df.dijet_cs)
# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
t11 = 0.7100
t12 = 0.7100
t21 =0.4
t22 = 0.4
xx = 'dijet_mass'
# further preprocess
from functions import cut
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.2783, max=None)
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.2783, max=None)


# %%
#fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)


# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 2) | (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)


# %%

# do hist or contour in the abcd regions
dfW=[]
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 15) & (isMCList[idx] < 20):
        dfW.append(df)
dfW=pd.concat(dfW)


df_tt=[]
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 12) & (isMCList[idx] < 15):
        df_tt.append(df)
df_tt=pd.concat(df_tt)


df_st=[]
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 6) & (isMCList[idx] < 12):
        df_st.append(df)
df_st=pd.concat(df_st)

df_VV=[]
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 3) & (isMCList[idx] <= 5 ):
        df_VV.append(df)
df_VV=pd.concat(df_VV)

# %%
# %%
df = dfs[0].copy()
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

from matplotlib.lines import Line2D  # 
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 5, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[1:4, 1:4])
legend_elements = [

    Line2D([0], [0], color='cyan', lw=2, label='H')
]

df = dfs[1]
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


ax_main.set_xlim(0.2783, 1)
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


ax_main.set_xlim(0.2783, 1)
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


ax_main.set_xlim(0.2783, 1)
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


df = dfs[0]
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


ax_main.set_xlim(0.2783, 1)
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


ax_main.set_xlim(0.2783, 1)
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


ax_main.set_xlim(0.2783, 1)
ax_main.set_ylim(0,   1)
ax_main.vlines(x=t11, ymin=0, ymax=1, linestyles='dotted', color='black')
ax_main.hlines(y=t22, xmin=0, xmax=1, linestyles='dotted', color='black')
ax_main.legend(handles=legend_elements)
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/contours/ST.png", bbox_inches='tight')





# %%
import dcor
m = dcor.distance_correlation(dfs[0][x1], dfs[0][x2])
# %%
