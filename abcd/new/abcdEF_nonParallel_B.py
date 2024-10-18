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

nReal, nMC = 100, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_v3b"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            39    
            # Data2A
]

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
                                                                'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt',
                                                               'jet2_btagDeepFlavB', 'dijet_cs',
                                                               'jet1_btagDeepFlavB'],
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
# Define regions and cuts
x1 = 'dijet_cs_abs'
x2 = 'PNN'
t1=0.3
t21 =0.3
t22 = 0.8
xx = 'dijet_mass'
# further preprocess
from functions import cut
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.6, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.6, max=None)


# %%
fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/dataMC_stacked.png")

# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

# %%
mA      = (dfZ[x1]<t1 ) & (dfZ[x2]>t22 ) 
mB      = (dfZ[x1]>t1 ) & (dfZ[x2]>t22 ) 
mAprime = (dfZ[x1]<t1 ) & (dfZ[x2]>t21 ) & (dfZ[x2]<t22 ) 
mBprime = (dfZ[x1]>t1 ) & (dfZ[x2]>t21 ) & (dfZ[x2]<t22 ) 
mC      = (dfZ[x1]<t1 ) & (dfZ[x2]<t21 ) 
mD      = (dfZ[x1]>t1 ) & (dfZ[x2]<t21 ) 



print("Region A : ", np.sum(dfZ.weight[mA])/dfZ.weight.sum())
print("Region B : ", np.sum(dfZ.weight[mB])/dfZ.weight.sum())
print("Region A' : ", np.sum(dfZ.weight[mAprime])/dfZ.weight.sum())
print("Region B' : ", np.sum(dfZ.weight[mBprime])/dfZ.weight.sum())
print("Region C : ", np.sum(dfZ.weight[mC])/dfZ.weight.sum())
print("Region D : ", np.sum(dfZ.weight[mD])/dfZ.weight.sum())

# %%
# Fill regions with data
bins = np.linspace(40, 300, 9)
x=(bins[1:]+bins[:-1])/2
regions = {
    'A':np.zeros(len(bins)-1),
    'B':np.zeros(len(bins)-1),
    'Aprime':np.zeros(len(bins)-1),
    'Bprime':np.zeros(len(bins)-1),
    'D':np.zeros(len(bins)-1),
    'C':np.zeros(len(bins)-1),
}


mA      = (dfs[0][x1]<t1 ) & (dfs[0][x2]>t22 ) 
mB      = (dfs[0][x1]>t1 ) & (dfs[0][x2]>t22 ) 
mAprime = (dfs[0][x1]<t1 ) & (dfs[0][x2]>t21 ) & (dfs[0][x2]<t22 ) 
mBprime = (dfs[0][x1]>t1 ) & (dfs[0][x2]>t21 ) & (dfs[0][x2]<t22 ) 
mC      = (dfs[0][x1]<t1 ) & (dfs[0][x2]<t21 ) 
mD      = (dfs[0][x1]>t1 ) & (dfs[0][x2]<t21 ) 
regions['A'] = regions['A'] + np.histogram(dfs[0][mA][xx], bins=bins)[0]
regions['B'] = regions['B'] + np.histogram(dfs[0][mB][xx], bins=bins)[0]
regions['Aprime'] = regions['Aprime'] + np.histogram(dfs[0][mAprime][xx], bins=bins)[0]
regions['Bprime'] = regions['Bprime'] + np.histogram(dfs[0][mBprime][xx], bins=bins)[0]
regions['C'] = regions['C'] + np.histogram(dfs[0][mC][xx], bins=bins)[0]
regions['D'] = regions['D'] + np.histogram(dfs[0][mD][xx], bins=bins)[0]

print("Data counts in ABCD regions")
print("Region A : ", regions["A"].sum())
print("Region B : ", regions["B"].sum())
print("Region Aprime : ", regions["Aprime"].sum())
print("Region Bprime : ", regions["Bprime"].sum())
print("Region C : ", regions["C"].sum())
print("Region D : ", regions["D"].sum())

regions['BprimeDepleted'] = regions["Bprime"]
regions['Bdepleted'] = regions["B"]
# remove MC simulations from a, b, c
for idx, df in enumerate(dfs[1:]):
    print(idx, df.dijet_mass.mean())
    mA      = (df[x1]<t1 ) & (df[x2]>t22 ) 
    mB      = (df[x1]>t1 ) & (df[x2]>t22 ) 
    mAprime = (df[x1]<t1 ) & (df[x2]>t21 ) & (df[x2]<t22 ) 
    mBprime = (df[x1]>t1 ) & (df[x2]>t21 ) & (df[x2]<t22 ) 
    mC      = (df[x1]<t1 ) & (df[x2]<t21 ) 
    mD      = (df[x1]>t1 ) & (df[x2]<t21 ) 
    regions['A'] = regions['A'] - np.histogram(df[mA][xx], bins=bins, weights=df[mA].weight)[0]
    regions['Bdepleted'] = regions['Bdepleted'] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['Aprime'] = regions['Aprime'] - np.histogram(df[mAprime][xx], bins=bins, weights=df[mAprime].weight)[0]
    regions['BprimeDepleted'] = regions['BprimeDepleted'] - np.histogram(df[mBprime][xx], bins=bins, weights=df[mBprime].weight)[0]
    regions['C'] = regions['C'] - np.histogram(df[mC][xx], bins=bins, weights=df[mC].weight)[0]
    regions['D'] = regions['D'] - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight)[0]

# %%
fig, ax = plt.subplots(1, 1)
ax.set_title("Trasnfer factor dependence")
ax.errorbar(x, regions["Aprime"]/regions["C"],              yerr = regions["Aprime"]/regions["C"]* np.sqrt(1/regions["Aprime"] + 1/regions["C"]) , marker='o', label='Aprime/C')
ax.errorbar(x, regions["BprimeDepleted"]/regions["D"],      yerr = regions["BprimeDepleted"]/regions["D"]* np.sqrt(1/regions["BprimeDepleted"] + 1/regions["D"]) , marker='o', label='Bprime/D')
ax.errorbar(x, regions["Bdepleted"]/regions["D"],           yerr = regions["Bdepleted"]/regions["D"]* np.sqrt(1/regions["Bdepleted"] + 1/regions["D"]) , marker='o', label='B/D')
ax.errorbar(x, regions["A"]/regions["C"],                   yerr = regions["A"]/regions["C"]* np.sqrt(1/regions["A"] + 1/regions["C"]) , marker='o', label='A/C')
ax.errorbar(x, regions["B"]/regions["Bprime"], yerr = regions["B"]/regions["Bprime"]* np.sqrt(1/regions["B"] + 1/regions["Bprime"]), marker='o', label='B/Bprime')
ax.errorbar(x, regions["A"]/regions["Aprime"], yerr = regions["B"]/regions["Bprime"]* np.sqrt(1/regions["A"] + 1/regions["Aprime"]), marker='o', label='A/Aprime')
ax.legend(bbox_to_anchor=(1, 1))


# %%

fig, ax = plt.subplots(3, 2, constrained_layout=True, figsize=(15, 10))
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"], histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions["Aprime"], histtype=u'step', label='Region A\'')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions["Bprime"], histtype=u'step', label='Region B\'')
ax[2,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[2,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')

ax[1,1].hist(bins[:-1], bins=bins, weights=np.sum(regions['Aprime'])*regions['D']/np.sum(regions['C']+1e-6), histtype=u'step', label=r'$A^\prime \times D / C$ ')
ax[1,1].errorbar(x, regions["Bprime"], yerr=np.sqrt(regions["Bprime"]), linestyle='none', color='black', marker='o')


ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t1, x2, t22), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t1, x2, t22), fontsize=14)
ax[1,0].set_title("%s < %.1f, %.1f < %s < %.1f"%(x1, t1, t21, x2, t22), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %.1f < %s < %.1f"%(x1, t1, t21, x2, t22), fontsize=14)
ax[2,0].set_title("%s < %.1f, %s < %.1f"%(x1, t1, x2, t21), fontsize=14)
ax[2,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t1, x2, t21), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/controlregions.png")

# %%
x = (bins[1:] + bins[:-1])/2


fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
b_err = np.sqrt(regions['Bprime'])
bprime_abcd = regions['Aprime']*regions['D']/regions['C']
adc_err = bprime_abcd*np.sqrt(1/regions['Aprime']+ 1/regions['D'] + 1/regions['C'])
ax[0].errorbar(x, regions['Bprime']-bprime_abcd, yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)
countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
    }

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mBprime      = (df[x1]>t1 ) & (df[x2]>t21 ) & (df[x2]<t22 ) 
    c = np.histogram(df.dijet_mass[mBprime], bins=bins,weights=df.weight[mBprime])[0]
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

    #c = ax[0].hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax[0].legend()

ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
data = regions['Bprime']-bprime_abcd
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[1].set_ylim(0., 3)
ax[1].errorbar(x, data/mc, yerr=np.sqrt(b_err**2 + adc_err**2)/mc , marker='o', color='black', linestyle='none')

fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/Ztry.png")
#ax.set_yscale('log')

# %%

correctionF = np.sum(data)/np.sum(mc)



fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
b_err = np.sqrt(regions['B'])
b_abcd = regions['A']*regions['D']/regions['C']
adc_err = b_abcd*np.sqrt(1/regions['A'] + 1/regions['D'] + 1/regions['C'])
ax[0].errorbar(x, regions['B']-b_abcd, yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)
countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
    }

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mB      = (df[x1]>t1 ) & (df[x2]>t22 ) 
    c = np.histogram(df.dijet_mass[mB], bins=bins,weights=df.weight[mB])[0]
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

    #c = ax[0].hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax[0].legend()

ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
data = regions['B']-b_abcd
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[1].set_ylim(0.5, 1.5)
ax[1].errorbar(x, data/mc, yerr=np.sqrt(b_err**2 + adc_err**2)/mc , marker='o', color='black', linestyle='none')
ax[1].errorbar(x, data/(mc*correctionF), yerr=np.sqrt(b_err**2 + adc_err**2)/mc , marker='o', color='gray', linestyle='none')

fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/Z_corrected.png")


# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))
# %%




qcd_mc = regions['B'] - countsDict['H'] - countsDict['ttbar'] - countsDict['ST'] - countsDict['VV'] - countsDict['VV'] - countsDict['W+Jets'] - countsDict['Z+Jets']
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


ax[0].hist(bins[:-1], bins=bins, weights=(regions['A']*regions['D']/regions['C']), label='QCD = ABCD estimation', histtype='step')
ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc, label='QCD = B - MC estimation[B]', histtype='step')
ax[1].errorbar(x, regions['A']*regions['D']/regions['C']/qcd_mc, yerr=adc_err/qcd_mc,linestyle='none', marker='o', color='black')
ax[1].set_ylim(0.95, 1.05)
ax[0].legend()
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/qcd_control.png")



# %%
