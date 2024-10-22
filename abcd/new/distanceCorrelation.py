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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dcor
# %%

nReal, nMC = 100, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_v3b_prova"
isMCList = [0, 1,
            #2,
            #3, 4, 5,
            #6,7,8,9,10,11,
            #12,13,14,
            #15,16,17,18,19,
            #20, 21, 22, 23, 36,
            #24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
            #39
            ]
if isMCList[-1]==39:
    nReal = nReal *2
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
                                                      columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt',
                                                               'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                               'jet2_eta', 'dijet_dR',
                                                               'dijet_dPhi', 'jet3_mass', 'Pileup_nTrueInt',
                                                               'jet2_btagPNetB', 'normalized_dijet_pt', 'dijet_cs',
                                                               'jet1_btagPNetB',
                                                                'PU_SF'],
                                                               returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                               returnFileNumberList=True)

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
dfs_precut = dfs.copy()
# %%
# Add weights predictions dijet_cs_abs
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]


# %%
# set to 1 weights of data
if (isMCList[-1]==39) & (len(dfs)==len(isMCList)):
    print("Removing last element from dfs")
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
    dfs = dfs[:-1]
dfs[0]['weight'] = np.ones(len(dfs[0]))


# %%
# Define the ABCD regions

x1 = 'dijet_cs_abs'
x2 = 'PNN'
t1  = 0.7
t21 = 0.4
t22 = 0.4
xx  = 'dijet_mass'
# further preprocess
from functions import cut

dfs = cut (data=dfs, feature='jet1_btagPNetB', min=0.2, max=None)
dfs = cut (data=dfs, feature='jet2_btagPNetB', min=0.2, max=None)
#dfs = cut (data=dfs, feature='dijet_pt', min=50, max=None)

# %%
mA      = (dfs[0][x1]<t1 ) & (dfs[0][x2]>t22 ) 
mB      = (dfs[0][x1]>t1 ) & (dfs[0][x2]>t22 ) 
mC      = (dfs[0][x1]<t1 ) & (dfs[0][x2]<t21 ) 
mD      = (dfs[0][x1]>t1 ) & (dfs[0][x2]<t21 ) 

fig, ax = plt.subplots(2, 2)
bins = np.linspace(0, 1, 11)
cA= ax[0,0].hist(dfs[0][x1][mA], histtype='step', bins=bins, density=True)[0]
cB= ax[0,0].hist(dfs[0][x1][mC], histtype='step', bins=bins, density=True)[0]
cC= ax[0,0].hist(dfs[0][x1][mB], histtype='step', bins=bins, density=True)[0]
cD= ax[0,0].hist(dfs[0][x1][mD], histtype='step', bins=bins, density=True)[0]



ax[1,0].hist(bins[:-1], bins=bins, weights= cA/(cB+1e-6), histtype='step')
ax[1,0].hist(bins[:-1], bins=bins, weights= cC/(cD+1e-6), histtype='step')
ax[1,0].set_ylim(0.95, 1.05)

cA = ax[0,1].hist(dfs[0][x2][mA], histtype='step', bins=np.linspace(0, 1, 11), density=True)[0]
cB = ax[0,1].hist(dfs[0][x2][mB], histtype='step', bins=np.linspace(0, 1, 11), density=True)[0]
cC = ax[0,1].hist(dfs[0][x2][mC], histtype='step', bins=np.linspace(0, 1, 11), density=True)[0]
cD = ax[0,1].hist(dfs[0][x2][mD], histtype='step', bins=np.linspace(0, 1, 11), density=True)[0]

ax[1,1].hist(bins[:-1], bins=bins, weights= cA/(cB + 1e-6), histtype='step')
ax[1,1].hist(bins[:-1], bins=bins, weights= cC/(cD + 1e-6), histtype='step')
ax[1,1].set_ylim(0.95, 1.05)

# %%
# df QCD
dfQCD = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 24) & (isMCList[idx] <= 35):
        dfQCD.append(df)
dfQCD=pd.concat(dfQCD)


# %%
bins = np.linspace(40, 300, 9)
df = dfs[0]
df = df.sample(frac=1).reset_index(drop=True)

xLog = [1e4, 2e4, 3e4, 4e4, 6e4, 8e4, 1e5, 2e5]
fig, ax = plt.subplots(1,1 )
indexColor = 0

for b in range(len(bins)-1):
    insideBin = (df.dijet_mass>bins[b]) & (df.dijet_mass<bins[b+1])
    lengths_bins = []
    corrList =[]
    pearsonList =[]
    for i in xLog:
        i=int(i)


        print("Correlation bewteen %s and %s in %d events for bin = %d"%(x1, x2, i, b))
        dcorCoef = dcor.distance_correlation(df[x1][insideBin].iloc[:i], df[x2][insideBin].iloc[:i])
        pearson = df[x1][insideBin].iloc[:i].corr(df[x2][insideBin].iloc[:i])

        #if int(i) == int(xLog[0]):
        #    fig2, ax2 = plt.subplots(1, 1)
        #    ax2.scatter(df[x1][insideBin].iloc[:i], df[x2][insideBin].iloc[:i], label='low stat %.2f'%dcorCoef)
        #elif int(i) == int(xLog[-1]):
        #    ax2.scatter(df[x1][insideBin].iloc[:i], df[x2][insideBin].iloc[:i], s=1.5, label='high stat %.2f'%dcorCoef, alpha=0.2)
        #    ax2.legend()
        #    ax2.set_title("Bin %d"%b)
        corrList.append(dcorCoef)
        pearsonList.append(pearson)
        lengths_bins.append(len(df[insideBin].iloc[:i]))
        print(" correlation %.5f"%dcorCoef)
    kwargs = {'color':'C%d'%indexColor,
              'linestyle':'solid' if indexColor<5 else 'dotted'}
    ax.plot(lengths_bins, corrList, label='Dcor bin %d'%b, marker='o', **kwargs)
    #ax.plot(lengths_bins, pearsonList, label='Pearson bin %d'%b, linestyle='dotted', marker='o', color='C%d'%indexColor)
    indexColor = indexColor+1
ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_ylim(1e-3, 1)
ax.set_xlabel("Sample Size")
ax.set_ylabel("Correlation")
ax.legend(bbox_to_anchor=(1,1), ncols=2)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/correlation_vs_sampleSize.png", bbox_inches='tight')
# %%
bins = np.linspace(40, 300, 9)
bin_pnn = np.linspace(0, 1, 11)
dec = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
for b in range(len(bins)-1):
    fig, ax = plt.subplots(1, 1)
    insideBin = (df.dijet_mass>bins[b]) & (df.dijet_mass<bins[b+1])
    for j in range(len(dec)-1):
        ax.hist(df[x2][insideBin][(df[x1]<dec[j]) & (df[x1][insideBin]<dec[j+1])], histtype='step', density=True, label='Bin %d : %.1f<%s<%.1f'%(b, dec[j], x1, dec[j+1]), bins=bin_pnn)
    ax.set_xlabel(x2)
    ax.legend(bbox_to_anchor=(1, 1))
#ax.hist(df[x1][insideBin][df[x2]<0.5], histtype='step', density=True)
#ax.hist(df[x1][insideBin][df[x2]>0.5], histtype='step', density=True)


# %%
#x = np.linspace(-2, 2, 100000)
#y = x**2 + np.random.normal(loc=0, scale=0.1, size=len(x))
#fig, ax  =  plt.subplots(1, 1)
#ax.plot(x, y, marker='.', linestyle='none')
#from scipy.stats import pearsonr
#ax.text(x=-1, y=3, s="Pearson R : %.3f"%pearsonr(x, y)[0], ha='left')
#ax.text(x=-1, y=2.5, s="DCor : %.3f"%dcor.distance_correlation(x, y), ha='left')

# %%



df = dfs[0].iloc[:10000]

print("Correlation bewteen %s and %s"%(x1, x2))
#m = dcor.distance_correlation(df[x1], df[x2])
#correlation = df[x1].corr(df[x2])
#print("Total correlation %.2f"%correlation)
for i in range(len(bins)-1):
    insideBin = (df.dijet_mass>bins[i]) & (df.dijet_mass<bins[i+1])

    m = dcor.distance_correlation(df[x1][insideBin], df[x2][insideBin])
    correlation = df[x1][insideBin].corr(df[x2][insideBin])
    print("Bin %d"%i)
    print("Distance correlation : %.3f"%(m))
    print("Linear correaltion : %.3f"%(correlation))
    print("\n")
# %%
import seaborn as sns
df = dfs[0]
cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Initialize empty matrices to store the distance correlations and linear correlations
dist_corr_matrix = np.zeros((len(cols), len(cols)))
linear_corr_matrix = np.zeros((len(cols), len(cols)))

# Compute both distance and linear correlations
for i, col1 in enumerate(cols):
    for j, col2 in enumerate(cols):
        # Distance correlation
        dist_corr_matrix[i, j] = dcor.distance_correlation(df[col1], df[col2])
        
        # Pearson correlation
        linear_corr_matrix[i, j] = df[col1].corr(df[col2])

# Convert to DataFrames for easy plotting
dist_corr_df = pd.DataFrame(dist_corr_matrix, index=cols, columns=cols)
linear_corr_df = pd.DataFrame(linear_corr_matrix, index=cols, columns=cols)

# Plot the Distance Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dist_corr_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Distance Correlation Matrix")
plt.show()

# Plot the Linear Correlation (Pearson) Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(linear_corr_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Linear Correlation Matrix (Pearson)")
plt.show()
# %%

fig, ax_main = plt.subplots(figsize=(8, 8))
divider = make_axes_locatable(ax_main)
ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

# Plot the 2D histogram in the main axes
x_bins, y_bins = np.linspace(0.3, 1, 31), np.linspace(0, 1, 31)
hist, x_edges, y_edges = np.histogram2d(x=dfs[0][x1], y=dfs[0][x2], bins=[x_bins, y_bins])
ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='coolwarm')
ax_main.set_xlabel(x1)
ax_main.set_ylabel(x2)

# Plot the marginalized histogram on top
ax_top.hist(dfs[0][x1], bins=x_bins, color='lightblue', edgecolor='black')
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_yticks([])
ax_top.xaxis.tick_top()

# Plot the marginalized histogram on the right
ax_right.hist(dfs[0][x2], bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
ax_right.set_ylim(ax_main.get_ylim())
ax_right.set_xticks([])
ax_right.yaxis.tick_right()

# %%
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/")
from plotFeatures import plotNormalizedFeatures
# %%
#dfZ = []
#for idx,df in enumerate(dfs):
#    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
#        dfZ.append(df)
#dfZ=pd.concat(dfZ)

dfQCD = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] >= 24) & (isMCList[idx] < 36):
        dfQCD.append(df)
dfQCD=pd.concat(dfQCD)

# %%
plotNormalizedFeatures([dfQCD, dfZ], outFile="/t3home/gcelotto/ggHbb/abcd/z_vs_qcdMu.png",
                       legendLabels=['QCDMuEnr', 'Z'], colors=['gray', 'green'],
                       histtypes=['step', 'step'], weights=[dfQCD.weight, dfZ.weight],
                       figsize=(15,60))
# %%
