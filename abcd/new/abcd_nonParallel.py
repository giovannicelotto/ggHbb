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
# %%

nReal, nMC = 790, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            39]
if isMCList[-1]==39:
    nReal = nReal *2
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    print(p)
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


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
                                                               'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR',
                                                               'dijet_dPhi', 'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt',
                                                               'jet2_btagDeepFlavB',
                                                               'jet1_btagDeepFlavB',
                                                               'dijet_dEta'],
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
    dfs[idx]['weight'] = df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
dfs[0]=pd.concat([dfs[0], dfs[-1]])
dfs = dfs[:-1]
dfs[0]['weight'] = np.ones(len(dfs[0]))
#dfs[-1]['weight'] = np.ones(len(dfs[-1]))
# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
t11=0.3
t12=0.5
t2 =0.3
xx = 'dijet_mass'

# further preprocess
from functions import cut
dfs = cut (data=dfs, feature=x1, min=t11, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)


# %%
# Data MC Control plot for dijet mass 
fig, ax =plt.subplots(1, 1)
bins_mass = np.linspace(40, 300, 101)
c = np.histogram(dfs[0].dijet_mass, bins=bins_mass)[0]
x = (bins_mass[:-1] + bins_mass[1:])/2
ax.errorbar(x, c, yerr=np.sqrt(c), linestyle='none', color='black', marker='o')
countsDict = {
        'Data':np.zeros(len(bins_mass)-1),
        'H':np.zeros(len(bins_mass)-1),
        'VV':np.zeros(len(bins_mass)-1),
        'ST':np.zeros(len(bins_mass)-1),
        'ttbar':np.zeros(len(bins_mass)-1),
        'Z+Jets':np.zeros(len(bins_mass)-1),
        'W+Jets':np.zeros(len(bins_mass)-1),
        'QCD':np.zeros(len(bins_mass)-1),
    }
cTot = np.zeros(len(bins_mass)-1)
for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    print(idx, process, isMC)
    c = np.histogram(df.dijet_mass, bins=bins_mass,weights=df.weight)[0]
    if 'Data' in process:
        continue
    elif 'GluGluHToBB' in process:
        print(process, isMC, " for Higgs")
        countsDict['H'] = countsDict['H'] + c
    elif 'ST' in process:
        countsDict['ST'] = countsDict['ST'] + c
    elif 'TTTo' in process:
        countsDict['ttbar'] = countsDict['ttbar'] + c
    elif 'QCD' in process:
        countsDict['QCD'] = countsDict['QCD'] + c
    elif 'ZJets' in process:
        #print(process, c)
        countsDict['Z+Jets'] = countsDict['Z+Jets'] + c
    elif 'WJets' in process:
        countsDict['W+Jets'] = countsDict['W+Jets'] + c
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] = countsDict['VV'] + c

    #c = ax.hist(df.dijet_mass, bins=bins_mass, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    print(key, np.sum(countsDict[key]))
    if np.sum(countsDict[key])==0:
        continue
    ax.hist(bins_mass[:-1], bins=bins_mass, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax.legend()
ax.set_yscale('log')
ax.set_ylim(10, ax.get_ylim()[1])



# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)
# %%

#sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
#from plotFeatures import plotNormalizedFeatures
#plotNormalizedFeatures(data=[dfZ, dfs[0]],
#                       outFile="/t3home/gcelotto/ggHbb/abcd/new/features.png", legendLabels=['Z', 'Data'],
#                       colors=['blue', 'red'], histtypes=[u'step', u'step'],
#                       alphas=[1, 1, 0.4, 0.4], figsize=(10,30), autobins=False,
#                       weights=[dfZ.weight, dfs[0].weight], error=True)


# %%

# %%
mA      = (dfZ[x1]<t12 ) & (dfZ[x2]>t2 ) 
mB      = (dfZ[x1]>t12 ) & (dfZ[x2]>t2 ) 
mC      = (dfZ[x1]<t12 ) & (dfZ[x2]<t2 ) 
mD      = (dfZ[x1]>t12 ) & (dfZ[x2]<t2 ) 



print("Region A : ", np.sum(dfZ.weight[mA])/dfZ.weight.sum())
print("Region B : ", np.sum(dfZ.weight[mB])/dfZ.weight.sum())
print("Region C : ", np.sum(dfZ.weight[mC])/dfZ.weight.sum())
print("Region D : ", np.sum(dfZ.weight[mD])/dfZ.weight.sum())

# %%
bins = np.linspace(40, 300, 16)
regions = {
    'A':np.zeros(len(bins)-1),
    'B':np.zeros(len(bins)-1),
    'D':np.zeros(len(bins)-1),
    'C':np.zeros(len(bins)-1),
}


# add data
mA      = (dfs[0][x1]<t12 ) & (dfs[0][x2]>t2 ) 
mB      = (dfs[0][x1]>t12 ) & (dfs[0][x2]>t2 ) 
mC      = (dfs[0][x1]<t12 ) & (dfs[0][x2]<t2 ) 
mD      = (dfs[0][x1]>t12 ) & (dfs[0][x2]<t2 ) 
regions['A'] = regions['A'] + np.histogram(dfs[0][mA][xx], bins=bins)[0]
regions['B'] = regions['B'] + np.histogram(dfs[0][mB][xx], bins=bins)[0]
regions['C'] = regions['C'] + np.histogram(dfs[0][mC][xx], bins=bins)[0]
regions['D'] = regions['D'] + np.histogram(dfs[0][mD][xx], bins=bins)[0]
print("Region A : ", regions["A"].sum())
print("Region B : ", regions["B"].sum())
print("Region C : ", regions["C"].sum())
print("Region D : ", regions["D"].sum())

# remove MC simulations
for idx, df in enumerate(dfs[1:]):
    mA      = (df[x1]<t12 ) & (df[x2]>t2 ) 
    mB      = (df[x1]>t12 ) & (df[x2]>t2 ) 
    mC      = (df[x1]<t12 ) & (df[x2]<t2 ) 
    mD      = (df[x1]>t12 ) & (df[x2]<t2 ) 
    regions['A'] = regions['A'] - np.histogram(df[mA][xx], bins=bins, weights=df[mA].weight)[0]
    #regions['B'] = regions['B'] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['C'] = regions['C'] - np.histogram(df[mC][xx], bins=bins, weights=df[mC].weight)[0]
    regions['D'] = regions['D'] - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight)[0]
print("Region A : ", regions["A"].sum())
print("Region B : ", regions["B"].sum())
print("Region C : ", regions["C"].sum())
print("Region D : ", regions["D"].sum())
# %%

fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
x=(bins[1:]+bins[:-1])/2
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"], histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')

ax[0,1].hist(bins[:-1], bins=bins, weights=regions['A']*regions['D']/(regions['C']+1e-6), histtype=u'step', label=r'$A\times D / C$ ')
#ax[0,2].hist(bins[:-1], bins=bins, weights=regions["B"]*regions['C']/(regions['A']*regions['D']), histtype=u'step', label=r'B$\times$C/A$\times$D')


ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t12, x2, t2), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t12, x2, t2), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t12, x2, t2), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t12, x2, t2), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
#fig.savefig("", bbox_inches='tight')
#print("Saving in ", "/t3home/gcelotto/ggHbb/abcd/output/abcd.png")

# %%
x = (bins[1:] + bins[:-1])/2
fig, ax = plt.subplots(1, 1)
b_err = np.sqrt(regions['B'])
adc_err = regions['A']*regions['D']/regions['C']*np.sqrt(1/regions['A'] + 1/regions['D'] + 1/regions['C'])
ax.errorbar(x, regions['B']-regions['A']*regions['D']/regions['C'], yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)
countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
    }

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    #print(isMC, process)
    c = np.histogram(df.dijet_mass, bins=bins,weights=df.weight)[0]
    if 'Data' in process:
        countsDict['Data'] = countsDict['Data'] + c
        print("adding data with", process)
    elif 'GluGluHToBB' in process:
        
        countsDict['H'] = countsDict['H'] + c
    elif 'ST' in process:
        countsDict['ST'] = countsDict['ST'] + c
    elif 'TTTo' in process:
        countsDict['ttbar'] = countsDict['ttbar'] + c
    elif 'QCD' in process:
        countsDict['QCD'] = countsDict['QCD'] + c
    elif 'ZJets' in process:
        countsDict['Z+Jets'] = countsDict['Z+Jets'] + c
    elif 'WJets' in process:
        countsDict['W+Jets'] = countsDict['W+Jets'] + c
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] = countsDict['VV'] + c

    #c = ax.hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax.hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax.legend()
#ax.set_yscale('log')
# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))
# %%
    


import seaborn as sns

# Create a 2D histogram to visualize correlation density
plt.figure(figsize=(10, 8))
plt.hist2d(dfs[0][x1], dfs[0][x2], bins=300, cmap='viridis')
plt.colorbar(label='Counts')
plt.title(f"2D Histogram of {x1} vs {x2}", fontsize=16)
plt.xlabel(x1, fontsize=14)
plt.ylabel(x2, fontsize=14)
plt.grid(True)
plt.show()

# Calculate the Pearson correlation coefficient
x3='jet1_btagDeepFlavB'
features = [x1, x2, x3]
correlation = dfs[0][features].corr().iloc[0, 1]
print(f"Pearson correlation coefficient between {x1} and {x2}: {correlation:.4f}")

# Create a correlation heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(dfs[0][features].corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# %%
