# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet, cut
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
# %%

nReal, nMC = 100, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            #24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
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
                                                      columns=['sf', 'dijet_mass', 'jet1_pt',
                                                               'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                               'jet2_eta', #'dijet_dR',
                                                               'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB','nJets_20GeV',
                                                                'jet3_mass', 'Pileup_nTrueInt','PU_SF',
                                                               #'dimuon_mass', 'muon_pt', 'muon_dxySig',
                                                               #'muon2_pt', 'muon2_dxySig'
                                                               ],
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
# remove the last element (data2a)
    dfs = dfs[:-1]
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))

# %%
# save dfs before applying cuts
dfs_precut = dfs.copy()

# %%
# Restore the cuts and apply new cuts
dfs  = dfs_precut.copy()
x1 = 'dijet_mass'
x2 = 'PNN'

t11=60
t12=110
t13=150
t21 =0.75
t22 = 0.75
xx = 'dijet_mass'
# further preprocess


dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.95, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.95, max=None)

# %%
fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=11, log=True)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/dataMC_stacked.png")


# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

# %%
mA      = ((dfZ[x1]<t11 ) | (dfZ[x1]>t13 )) & (dfZ[x2]>t21)
mB      = ((dfZ[x1]>t11 ) & (dfZ[x1]<t12 )) & (dfZ[x2]>t21)
mC      = ((dfZ[x1]<t11 ) | (dfZ[x1]>t13 )) & (dfZ[x2]<t21)
mD      = ((dfZ[x1]>t11 ) & (dfZ[x1]<t12 )) & (dfZ[x2]<t21)



print("Region A : ", np.sum(dfZ.weight[mA])/dfZ.weight.sum())
print("Region B : ", np.sum(dfZ.weight[mB])/dfZ.weight.sum())
print("Region C : ", np.sum(dfZ.weight[mC])/dfZ.weight.sum())
print("Region D : ", np.sum(dfZ.weight[mD])/dfZ.weight.sum())

# %%
bins = np.linspace(40, 300, 21)
regions = {
    'A':[np.zeros(1), np.zeros(1)], #tuple for values and errors
    'B':[np.zeros(len(bins)-1), np.zeros(len(bins)-1)], #tuple for values and errors
    'D':[np.zeros(len(bins)-1), np.zeros(len(bins)-1)], #tuple for values and errors
    'C':[np.zeros(1), np.zeros(1)], #tuple for values and errors
}


# Fill regions with data
mA      = ((dfs[0][x1]<t11 ) | (dfs[0][x1]>t13 )) & (dfs[0][x2]>t21)
mB      = ((dfs[0][x1]>t11 ) & (dfs[0][x1]<t12 )) & (dfs[0][x2]>t21)
mC      = ((dfs[0][x1]<t11 ) | (dfs[0][x1]>t13 )) & (dfs[0][x2]<t21)
mD      = ((dfs[0][x1]>t11 ) & (dfs[0][x1]<t12 )) & (dfs[0][x2]<t21)
regions['A'][0] = regions['A'][0] + np.sum(dfs[0][mA][xx])
regions['B'][0] = regions['B'][0] + np.histogram(dfs[0][mB][xx], bins=bins)[0]
regions['C'][0] = regions['C'][0] + np.sum(dfs[0][mC][xx])
regions['D'][0] = regions['D'][0] + np.histogram(dfs[0][mD][xx], bins=bins)[0]

regions['A'][1] = regions['A'][1] + np.sqrt(np.sum(dfs[0][mA][xx]))
regions['B'][1] = regions['B'][1] + np.sqrt(np.histogram(dfs[0][mB][xx], bins=bins)[0])
regions['C'][1] = regions['C'][1] + np.sqrt(np.sum(dfs[0][mC][xx]))
regions['D'][1] = regions['D'][1] + np.sqrt(np.histogram(dfs[0][mD][xx], bins=bins)[0])
print("Data counts in ABCD regions")
print("Region A : ", regions["A"][0].sum())
print("Region B : ", regions["B"][0].sum())
print("Region C : ", regions["C"][0].sum())
print("Region D : ", regions["D"][0].sum())

# remove MC simulations from a, b, c
for idx, df in enumerate(dfs[1:]):
    print(idx, df.dijet_mass.mean())
    mA      = ((df[x1]<t11 ) | (df[x1]>t13 )) & (df[x2]>t21)
    mB      = ((df[x1]>t11 ) & (df[x1]<t12 )) & (df[x2]>t21)
    mC      = ((df[x1]<t11 ) | (df[x1]>t13 )) & (df[x2]<t21)
    mD      = ((df[x1]>t11 ) & (df[x1]<t12 )) & (df[x2]<t21)
    regions['A'][0] = regions['A'][0] - np.sum(df[mA].weight)
    #regions['B'][0] = regions['B'][0] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['C'][0] = regions['C'][0] - np.sum(df[mC].weight)
    regions['D'][0] = regions['D'][0] - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight)[0]

    regions['A'][1] = np.sqrt(regions['A'][1]**2 + np.sum(df[mA].weight**2))
    #regions['B'][1] = regions['B'][1] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['C'][1] = np.sqrt(regions['C'][1]**2 - np.sum(df[mC].weight**2))
    regions['D'][1] = np.sqrt(regions['D'][1]**2 - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight**2)[0])




# %%

fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
x=(bins[1:]+bins[:-1])/2
#ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"][0], histtype=u'step', label='Region B')
#ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'][0], histtype=u'step', label='Region D')

ax[0,1].hist(bins[:-1], bins=bins, weights=regions['A'][0]*regions['D'][0]/(regions['C'][0]+1e-6), histtype=u'step', label=r'$A\times D / C$ ')
ax[0,1].errorbar(x, regions["B"][0], yerr=regions["B"][1], linestyle='none', color='black', marker='o')


ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t11, x2, t22), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t12, x2, t22), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t11, x2, t21), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t12, x2, t21), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
#fig.savefig("", bbox_inches='tight')
#print("Saving in ", "/t3home/gcelotto/ggHbb/abcd/output/abcd.png")

# %%
x = (bins[1:] + bins[:-1])/2


fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)

adc_err = regions['A'][0]*regions['D'][0]/regions['C'][0]*np.sqrt((regions["A"][1]/regions['A'][0])**2 + (regions["D"][1]/regions['D'][0])**2 + (regions["C"][1]/regions['C'][0])**2)
ax[0].errorbar(x, regions['B'][0]-regions['A'][0]*regions['D'][0]/regions['C'][0], yerr=np.sqrt(regions['B'][1]**2 + adc_err**2) , marker='o', color='black', linestyle='none')
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
    mB      = ((df[x1]>t11 ) & (df[x1]<t12 )) & (df[x2]>t21)
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
data = regions['B'][0]-regions['A'][0]*regions['D'][0]/regions['C'][0]
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[1].set_ylim(0., 3)
ax[1].errorbar(x, data/mc, yerr=np.sqrt(regions['B'][1]**2 + adc_err**2)/mc , marker='o', color='black', linestyle='none')

fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/Ztry.png")
#ax.set_yscale('log')
# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))
# %%




qcd_mc = regions['B'][0] - countsDict['H'] - countsDict['ttbar'] - countsDict['ST'] - countsDict['VV'] - countsDict['VV'] - countsDict['W+Jets'] - countsDict['Z+Jets']
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


ax[0].hist(bins[:-1], bins=bins, weights=(regions['A'][0]*regions['D'][0]/regions['C'][0]), label='QCD = ABCD estimation', histtype='step')
ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc, label='QCD = B - MC estimation[B]', histtype='step')
ax[1].errorbar(x, regions['A'][0]*regions['D'][0]/regions['C'][0]/qcd_mc, yerr=adc_err/qcd_mc,linestyle='none', marker='o', color='black')
ax[1].set_ylim(0.8, 1.2)
ax[0].legend()



# %%
mA      = ((dfs[0][x1]<t11 ) | (dfs[0][x1]>t13 )) & (dfs[0][x2]>t21)
mB      = ((dfs[0][x1]>t11 ) & (dfs[0][x1]<t12 )) & (dfs[0][x2]>t21)
mC      = ((dfs[0][x1]<t11 ) | (dfs[0][x1]>t13 )) & (dfs[0][x2]<t21)
mD      = ((dfs[0][x1]>t11 ) & (dfs[0][x1]<t12 )) & (dfs[0][x2]<t21)
fig, ax = plt.subplots(1, 1)
bins=np.arange(40, 300, 10)
ax.hist(dfs[0].dijet_mass,bins=bins, histtype='step', density=False, label='mA', color='black')
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Dijet Mass [GeV]")
ax.hist(dfs[0].dijet_mass[(mA) | (mC)],bins=bins, density=False, label='Fail')
ax.hist(dfs[0].dijet_mass[(mB) | (mD)],bins=bins, density=False, label='Pass')
#ax.hist(dfs[0].dijet_mass[mC],bins=bins, density=False, label='mC')
#ax.hist(dfs[0].dijet_mass[mD],bins=bins, density=False, label='mD')
ax.legend()
# %%





# From now on, new approach dated 23 October
# Build a histogram of dijet mass from data1A at low PNN score
# Subtract to the shape the counts of SM nonQCD contributions
# The result will be a binned dijet mass distributions of QCD for low PNN score
# Get the dijet mass histogram for high PNN score
# subtract the SM non QCD contributions
# The result will be a binned dijet mass distributions of QCD for high PNN score
# Find the Transfer Factor (or Scale Factor) that minimizes the residuals between highPNN_data_SMsubtracted and lowPNN_data_SMsubtracted
# The Scale Factor can be fitted in the CR of dijet mass (mjj<60 | mjj>150)
# Use the QCD_lowNN_SMsubtracted x SF to estimate the QCD amount in data at high PNN
# Subtract these histograms:
# highNN (non SM subtracted) - QCD_lowNN_SMsubtracted x SF  = highNN_SM
# You should get the SM non QCD events at high NN

dfs = cut(dfs, 'jet1_btagDeepFlavB', min=0.3, max=None)
dfs = cut(dfs, 'jet2_btagDeepFlavB', min=0.3, max=None)



# Get the shape from dijet mass of data at low PNN score
pnn_threshold = 0.5
bins = np.array([40, 60, 110, 130, 150, 180, 200, 220, 240, 260, 280, 300])
#bins = np.arange(40, 300,20)
x = (bins[:-1]+bins[1:])/2
lowPNN_data = np.histogram(dfs[0].dijet_mass[dfs[0].PNN<pnn_threshold], bins=bins)[0]



# Subtract to the shape the counts of SM nonQCD contributions
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
    mlowPNN      = df.PNN<pnn_threshold
    c = np.histogram(df.dijet_mass[mlowPNN], bins=bins,weights=df.weight[mlowPNN])[0]
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

cTot_lowNN = np.zeros(len(bins)-1)
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    cTot_lowNN = cTot_lowNN + countsDict[key]
lowPNN_data_SMsubtracted = lowPNN_data - cTot_lowNN
# The result will be a binned dijet mass distributions of QCD for low PNN score

# Get the dijet mass histogram for high PNN score
highPNN_data = np.histogram(dfs[0].dijet_mass[dfs[0].PNN>pnn_threshold], bins=bins)[0]
# subtract the SM non QCD contributions
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
    mlowPNN      = df.PNN>pnn_threshold
    c = np.histogram(df.dijet_mass[mlowPNN], bins=bins,weights=df.weight[mlowPNN])[0]
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
cTot = np.zeros(len(bins)-1)
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    cTot = cTot + countsDict[key]

highPNN_data_SMsubtracted = highPNN_data - cTot
# Result is the histogram binned of dijet mass for qcd at high pnn
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins=bins, weights=lowPNN_data, histtype='step', label='Data PNN < %.1f'%pnn_threshold)
ax.hist(bins[:-1], bins=bins, weights=lowPNN_data_SMsubtracted, histtype='step', label='Data - SMnonQCD PNN < %.1f'%pnn_threshold)
ax.hist(bins[:-1], bins=bins, weights=highPNN_data, histtype='step', label='Data PNN > %.1f'%pnn_threshold)
ax.hist(bins[:-1], bins=bins, weights=highPNN_data_SMsubtracted, histtype='step', label='Data - SMnonQCD PNN > %.1f'%pnn_threshold)
ax.set_yscale('log')
ax.legend()
# Find the scale factor that minimizes the residuals between highPNN_data_SMsubtracted and lowPNN_data_SMsubtracted
# Get bin centers from bin edges
# The scale factor can be fitted in the CR of dijet mass
bin_centers = (bins[:-1] + bins[1:]) / 2
filter_indices = np.where((bin_centers < 60) | (bin_centers > 150))
highPNN_filtered = highPNN_data_SMsubtracted[filter_indices]
lowPNN_filtered = lowPNN_data_SMsubtracted[filter_indices]
alpha = np.sum(highPNN_filtered * lowPNN_filtered) / np.sum(lowPNN_filtered ** 2)
print("Optimal scale factor:", alpha)

# Use the QCD low PNN times scale factor to estimate the QCD amount in data at high PNN
lowPNN_scaled = alpha * lowPNN_data_SMsubtracted
# Subtract the QCD at high PNN with this method.
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
data = highPNN_data - lowPNN_scaled
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[0].errorbar(bin_centers, highPNN_data - lowPNN_scaled, yerr=np.sqrt(highPNN_data+alpha**2*lowPNN_data), marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax[0].legend()
ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
ax[1].set_ylim(0., 2)
ax[1].errorbar(x, data/mc, yerr=np.sqrt(highPNN_data+alpha**2*lowPNN_data)/mc , marker='o', color='black', linestyle='none')

fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/newmethod.png")
# You should get the SM non QCD events
#dfs = dfs_precut.copy()
#sys.exit()
# %%






















# Proposed update using SF derived from a 2D distributions
# Create 2d distribution for dijet mass and jet1 pt at low and high PNN score

mass_bins = np.linspace(40, 300, 14)   # Dijet mass binning
pt_bins = np.linspace(0, 200, 5)     # Jet pt binning
lowPNN_data = np.histogram2d(dfs[0].dijet_mass[dfs[0].PNN < pnn_threshold], 
                              np.clip(dfs[0].jet1_pt[dfs[0].PNN < pnn_threshold], pt_bins[0], pt_bins[-1]),
                              bins=[mass_bins, pt_bins])[0]
highPNN_data = np.histogram2d(dfs[0].dijet_mass[dfs[0].PNN > pnn_threshold], 
                               np.clip(dfs[0].jet1_pt[dfs[0].PNN > pnn_threshold], pt_bins[0], pt_bins[-1]), 
                               bins=[mass_bins, pt_bins])[0]
from plots.plot2d import plot
fig = plot(dfs[0][dfs[0].PNN<pnn_threshold], x1='dijet_mass', x2='jet1_pt', x_bins=mass_bins, y_bins=pt_bins)

fig = plot(dfs[0][dfs[0].PNN>pnn_threshold], x1='dijet_mass', x2='jet1_pt', x_bins=mass_bins, y_bins=pt_bins)
fig = plot(dfs[0], x1='dijet_mass', x2='jet1_pt', x_bins=mass_bins, y_bins=pt_bins)


# %%
# Get the counts of SM non QCD for low PNN score and high PNN score
countsDict = {
    'Data': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'H': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'VV': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'ST': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'ttbar': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'W+Jets': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'QCD': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
    'Z+Jets': np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)),
}

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx + 1]
    process = dfProcesses.process[isMC]
    mlowPNN = df.PNN < pnn_threshold  # Low PNN condition

    # 2D histogram for this background process
    c = np.histogram2d(df.dijet_mass[mlowPNN], 
                       np.clip(df.jet1_pt[mlowPNN], pt_bins[0], pt_bins[-1]), 
                       bins=[mass_bins, pt_bins], 
                       weights=df.weight[mlowPNN])[0]

    if 'Data' in process:
        countsDict['Data'] += c
        print("adding data with", process)
    elif 'GluGluHToBB' in process:
        countsDict['H'] += c
    elif 'ST' in process:
        countsDict['ST'] += c
    elif 'TTTo' in process:
        countsDict['ttbar'] += c
    elif 'QCD' in process:
        countsDict['QCD'] += c
    elif 'ZJets' in process:
        countsDict['Z+Jets'] += c
    elif 'WJets' in process:
        countsDict['W+Jets'] += c
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] += c

# Total background count in low PNN region
cTot_lowPNN = np.zeros((len(mass_bins) - 1, len(pt_bins) - 1))
for key in countsDict.keys():
    if np.sum(countsDict[key]) == 0:
        continue
    print(key, np.sum(countsDict[key]))
    cTot_lowPNN += countsDict[key]

# Subtract total background from low PNN data (2d operation)
lowPNN_data_SMsubtracted = lowPNN_data - cTot_lowPNN


# Do the same for high PNN
countsDict = {key: np.zeros((len(mass_bins) - 1, len(pt_bins) - 1)) for key in countsDict.keys()}

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx + 1]
    process = dfProcesses.process[isMC]
    mhighPNN = df.PNN > pnn_threshold  # High PNN condition

    # 2D histogram for this background process
    c = np.histogram2d(df.dijet_mass[mhighPNN], 
                       np.clip(df.jet1_pt[mhighPNN], pt_bins[0], pt_bins[-1]),
                       bins=[mass_bins, pt_bins], 
                       weights=df.weight[mhighPNN])[0]

    if 'Data' in process:
        countsDict['Data'] += c
        print("adding data with", process)
    elif 'GluGluHToBB' in process:
        countsDict['H'] += c
    elif 'ST' in process:
        countsDict['ST'] += c
    elif 'TTTo' in process:
        countsDict['ttbar'] += c
    elif 'QCD' in process:
        countsDict['QCD'] += c
    elif 'ZJets' in process:
        countsDict['Z+Jets'] += c
    elif 'WJets' in process:
        countsDict['W+Jets'] += c
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] += c

# Total background count in high PNN region
cTot_highPNN = np.zeros((len(mass_bins) - 1, len(pt_bins) - 1))
for key in countsDict.keys():
    if np.sum(countsDict[key]) == 0:
        continue
    print(key, np.sum(countsDict[key]))
    cTot_highPNN += countsDict[key]

# Subtract total background from high PNN data
highPNN_data_SMsubtracted = highPNN_data - cTot_highPNN


# Initialize the SF array
#SF = np.ones((len(mass_bins) - 1, len(pt_bins) - 1))  # SF for each (mass, pt) bin
SF = np.ones(len(pt_bins) - 1)
# Loop over the mass and pt bins to compute the SF in the CR
#for mass_bin in range(len(mass_bins) - 1):
for pt_bin in range(len(pt_bins) - 1):
    # marginalize in dijet mass, get a binned pt distribution
    highPNN_cr =np.sum( highPNN_data_SMsubtracted[:, pt_bin], axis=0)  
    lowPNN_cr = np.sum(lowPNN_data_SMsubtracted[:, pt_bin], axis=0)   
    print(highPNN_cr)
            
    # Calculate transfer factor for this (mjj, pt) slice
    if lowPNN_cr > 0:  # Avoid division by zero
        SF[pt_bin] = highPNN_cr / lowPNN_cr
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(dfs[0].jet1_pt[dfs[0].PNN<pnn_threshold], bins=pt_bins, histtype='step', density=False, label='PNN<%.1f'%pnn_threshold)
chigh=  ax.hist(np.clip(dfs[0].jet1_pt[dfs[0].PNN>pnn_threshold], pt_bins[0], pt_bins[-1]), bins=pt_bins, histtype='step', density=False, label='PNN>%.1f'%pnn_threshold)[0]

bin_indices = np.digitize(dfs[0]['jet1_pt'], pt_bins) - 1
bin_indices = np.clip(bin_indices, 0, len(SF)-1)
dfs[0]['pt_SF'] = SF[bin_indices]
clow_cor = ax.hist(dfs[0].jet1_pt[dfs[0].PNN<pnn_threshold], bins=pt_bins, weights=dfs[0].pt_SF[dfs[0].PNN<pnn_threshold], histtype='step', density=False, label='PNN<%.1f corrected'%pnn_threshold, linestyle='dotted')[0]
ax.set_yscale('log')
ax.legend()
# these plots do not match perfectly because you need to subtract the contribution from SM non qcd
# These SF correct the lowPNN distribution of QCD into highPNN distribution of QCD which can be subtracted to the highPNN distributions to see the SMnonQCD residuals
# %%
df = dfs[0]

# 2d of qcd
mass_final_counts = np.zeros(len(mass_bins)-1)
lowPNN_data_SMsubtracted_scaled = lowPNN_data_SMsubtracted.copy()
#for i in range(len(mass_final_counts)):
for j in range(len(pt_bins)-1):
    lowPNN_data_SMsubtracted_scaled[:,j] = lowPNN_data_SMsubtracted[:,j]*SF[j]
    mass_final_counts = np.sum(highPNN_data, axis=1) - np.sum(lowPNN_data_SMsubtracted_scaled, axis=1)

# %%
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
cTot = np.zeros(len(mass_bins)-1)
x_mass = (mass_bins[:-1] + mass_bins[1:])/2
ax[0].errorbar(x_mass, mass_final_counts, yerr=np.sqrt(np.sum(highPNN_data, axis=1)), color='black', marker='o', linestyle='none')
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax[0].hist(mass_bins[:-1], bins=mass_bins, weights=np.sum(countsDict[key], axis=1), bottom=cTot, label=key)
    cTot = cTot + np.sum(countsDict[key], axis=1)
ax[0].legend()

ax[1].errorbar(x=x_mass, y=mass_final_counts/cTot, yerr=np.sqrt(np.sum(highPNN_data, axis=1))/cTot, linestyle='none', color='black', marker='o')
ax[1].hlines(y=1, xmin=mass_bins[0], xmax=mass_bins[-1], color='black')
ax[0].set_xlim(mass_bins[0], mass_bins[-1])




# %%
