# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet, getDfProcesses, sortPredictions
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from hist import Hist
import hist
# %%
# Define number of Data Files, MC files per process, predictionsPath, list of MC processes
nReal = 308
nMC = -1
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

# Take the DataFrame with processes, path, xsection. Filter the needed rows (processes)
dfProcesses = getDfProcesses()
processes = dfProcesses.process[isMCList].values

# Put all predictions used for training in the proper folder. They will not be used here
sortPredictions()

# Get predictions names path for all the datasets
predictionsFileNames = []
for p in processes:
    print(p)
    tempFileNames = glob.glob(predictionsPath+"/%s/others/*.parquet"%p)
    sortedFileNames = sorted(tempFileNames, key=lambda x: int(''.join(filter(str.isdigit, x))))
    predictionsFileNames.append(sortedFileNames)
    if len(predictionsFileNames)==0:
        print("*"*10)
        print("No Files found for process ", p)
        print("*"*10)

# For each fileNumber extract the fileNumber
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
# Load flattuple for fileNumbers matching
paths = list(dfProcesses.flatPath[isMCList])
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                      columns=[ 'sf',        'dijet_mass',   'dijet_pt',             'jet1_pt',
                                                                'jet2_pt',   'jet1_mass',    'jet2_mass',            'jet1_eta',
                                                                'jet2_eta',  'dijet_dR',     'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
                                                                'jet3_mass', 'Pileup_nTrueInt', 'leptonClass',
                                                                'dijet_cs',  'PU_SF'],
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
for idx, df in enumerate(dfs):
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)
dfs_precut = dfs.copy()

# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
t11=0.6
t12=0.6
t21 =0.4
t22 = 0.4
xx = 'dijet_mass'
# further preprocess
from functions import cut
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.3, max=None)#dfs = cut (data=dfs, feature='PNN', min=0.6, max=None)


# %%
fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=101, log=True)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/dataMC_stacked.png")


# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 2) | (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

# %%
mA      = (dfZ[x1]<t11 ) & (dfZ[x2]>t22 ) 
mB      = (dfZ[x1]>t12 ) & (dfZ[x2]>t22 ) 
mC      = (dfZ[x1]<t11 ) & (dfZ[x2]<t21 ) 
mD      = (dfZ[x1]>t12 ) & (dfZ[x2]<t21 ) 


print("Region A : ", np.sum(dfZ.weight[mA])/dfZ.weight.sum())
print("Region B : ", np.sum(dfZ.weight[mB])/dfZ.weight.sum())
print("Region C : ", np.sum(dfZ.weight[mC])/dfZ.weight.sum())
print("Region D : ", np.sum(dfZ.weight[mD])/dfZ.weight.sum())
# %%

fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 1, 51)
ax.hist(dfs[0].dijet_cs_abs, bins=bins, histtype='step', label='Data', density=True)
ax.hist(dfZ.dijet_cs_abs, bins=bins, weights=dfZ.weight, histtype='step', label='ZJets', density=True)
ax.hist(dfs[1].dijet_cs_abs, bins=bins, weights=dfs[1].weight, histtype='step', label='Higgs', density=True)
ax.set_xlabel("Dijet cs")
ax.legend()

# %%
fig, ax = plt.subplots(1, 1)
bins=np.arange(1, 5)
cData = ax.hist(dfs[0].leptonClass, bins=bins, histtype='step', label='Data', density=True, color='C0')[0]
cZ = ax.hist(dfZ.leptonClass, bins=bins, weights=dfZ.weight, histtype='step', label='ZJets', density=True, color='C1')[0]
cH = ax.hist(dfs[1].leptonClass, bins=bins, weights=dfs[1].weight, histtype='step', label='Higgs', density=True, color='C2')[0]

for i in range(len(cData)):
    ax.text((bins[i] + bins[i + 1]) / 2, 0.125, f'{cData[i]*100:.1f}%', ha='center', va='bottom', color='C0')
    ax.text((bins[i] + bins[i + 1]) / 2, 0.1, f'{cZ[i]*100:.1f}%', ha='center', va='bottom', color='C1')
    ax.text((bins[i] + bins[i + 1]) / 2, 0.08, f'{cH[i]*100:.1f}%', ha='center', va='bottom', color='C2')

ax.set_xticks(bins[:-1] + 0.5)  # Center labels within the bins
ax.set_xticklabels(["Trig Muon", "Muon", "No Muon"])
ax.set_xlabel("Muon In Jet")
ax.set_yscale('log')
ax.legend()




# %%
# ABCD Start here
# Define 4 histograms for mjj, one for each region

bins = np.linspace(40, 300, 7)
hA = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
hB = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
hC = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
hD = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
regions = {
    'A' : hA,
    'B' : hB,
    'C' : hC,
    'D' : hD,
}


# Fill regions with data
mA      = (dfs[0][x1]<t11 ) & (dfs[0][x2]>t22 ) 
mB      = (dfs[0][x1]>t12 ) & (dfs[0][x2]>t22 ) 
mC      = (dfs[0][x1]<t11 ) & (dfs[0][x2]<t21 ) 
mD      = (dfs[0][x1]>t12 ) & (dfs[0][x2]<t21 ) 
regions['A'].fill(dfs[0][mA][xx])
regions['B'].fill(dfs[0][mB][xx])
regions['C'].fill(dfs[0][mC][xx])
regions['D'].fill(dfs[0][mD][xx])

print("Data counts in ABCD regions")
print("Region A : ", regions["A"].sum())
print("Region B : ", regions["B"].sum())
print("Region C : ", regions["C"].sum())
print("Region D : ", regions["D"].sum())
# %%
# remove MC from non QCD processes simulations from A, C, D
for idx, df in enumerate(dfs[1:]):
    print(idx, df.dijet_mass.mean())
    mA      = (df[x1]<t11 ) & (df[x2]>t22 ) 
    mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 
    mC      = (df[x1]<t11 ) & (df[x2]<t21 ) 
    mD      = (df[x1]>t12 ) & (df[x2]<t21 ) 
    # Subtract the events by filling with opposite weights
    regions['A'].fill(df[mA][xx], weight=-df[mA].weight)  
    regions['C'].fill(df[mC][xx], weight=-df[mC].weight)  
    regions['D'].fill(df[mD][xx], weight=-df[mD].weight)  
    # In B don't do it, we want to see the excess from Data - QCD = MCnonQCD
    #regions['B'] = regions['B'] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]


# %%
# Plot Data and Data and B=A*D/C estimation in SR
fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
x=(bins[1:]+bins[:-1])/2
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"].values(), histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"].values(), histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'].values(), histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'].values(), histtype=u'step', label='Region D')

hB_ADC_values = regions['A'].values()*regions['D'].values()/(regions['C'].values()+1e-6)
ADC_err = regions['A'].values()*regions['D'].values()/regions['C'].values()*np.sqrt(1/regions['A'].values() + 1/regions['D'].values() + 1/regions['C'].values())
hB_ADC = Hist.new.Reg(len(bins) - 1, bins[0], bins[-1], name='mjj').Weight()
hB_ADC.values()[:] = hB_ADC_values
hB_ADC.variances()[:] = ADC_err**2 

ax[0,1].hist(bins[:-1], bins=bins, weights=hB_ADC.values(), histtype=u'step', label=r'$A\times D / C$ ')
ax[0,1].errorbar(x, regions["B"].values(), yerr=np.sqrt(regions["B"].variances()), linestyle='none', color='black', marker='o')

ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t11, x2, t22), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t12, x2, t22), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t11, x2, t21), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t12, x2, t21), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/CR_SR_check.png", bbox_inches='tight')


# %%
x = (bins[1:] + bins[:-1])/2

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
hExcess = regions["B"].copy()
hExcess.values()[:] = regions["B"].values()-hB_ADC.values()
hExcess.variances()[:] = regions["B"].variances() + hB_ADC.variances()

ax[0].errorbar(x, hExcess.values(), yerr=np.sqrt(hExcess.variances()) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)


h = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
countsDict = {
        'Data'   : h.copy() ,
        'H'      : h.copy() ,
        'VV'     : h.copy() ,
        'ST'     : h.copy() ,
        'ttbar'  : h.copy() ,
        'W+Jets' : h.copy() ,
        'QCD'    : h.copy() ,
        'Z+Jets' : h.copy() ,
    }

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 

    h = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
    h.fill(df.dijet_mass[mB], weight=df.weight[mB])

    if 'Data' in process:
        countsDict['Data'] = countsDict['Data'] + h
        print("adding data with", process)
    elif 'GluGluHToBB' in process:
        countsDict['H'] = countsDict['H'] + h
    elif 'ST' in process:
        countsDict['ST'] = countsDict['ST'] + h
    elif 'TTTo' in process:
        countsDict['ttbar'] = countsDict['ttbar'] + h
    elif 'QCD' in process:
        countsDict['QCD'] = countsDict['QCD'] + h
    elif 'ZJets' in process:
        countsDict['Z+Jets'] = countsDict['Z+Jets'] + h
    elif 'WJets' in process:
        countsDict['W+Jets'] = countsDict['W+Jets'] + h
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] = countsDict['VV'] + h

    #c = ax[0].hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    if countsDict[key].values().sum()==0:
        continue
    print(key, countsDict[key].values().sum())
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key].values(), bottom=cTot, label=key)
    cTot = cTot + countsDict[key].values()
ax[0].legend()

ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')

mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[0].bar(x, 2*np.sqrt(mc.variances()), width=np.diff(bins), bottom=mc.values() - np.sqrt(mc.variances()), 
       color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")

ax[1].bar(x, 2*np.sqrt(mc.variances())/mc.values(), width=np.diff(bins), bottom=1 - np.sqrt(mc.variances())/mc.values(), 
       color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")


ax[1].set_ylim(0., 2)
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[1].errorbar(x, hExcess.values()/mc.values(), yerr=np.sqrt(hExcess.variances())/mc.values() , marker='o', color='black', linestyle='none')
hep.cms.label(lumi=np.round(nReal*0.774/1017, 3), ax=ax[0])
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/SMnonQCD_closure.png")
#ax.set_yscale('log')
# %%
for letter in ['A', 'B', 'C', 'D']:
    print(regions[letter].sum())
# %%

# put negative values to countsDict
for key in countsDict:
    countsDict[key].values()[:] = -countsDict[key].values()[:]

qcd_mc = regions['B'] + countsDict['H'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['VV'] + countsDict['VV'] + countsDict['W+Jets'] + countsDict['Z+Jets']
# Restore positive histograms
for key in countsDict:
    countsDict[key].values()[:] = -countsDict[key].values()[:]

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
ax[0].hist(bins[:-1], bins=bins, weights=(hB_ADC.values()), label='QCD = ABCD estimation', histtype='step')
ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc.values(), label='QCD = B - MC estimation[B]', histtype='step')
ax[1].errorbar(x, hB_ADC.values()/qcd_mc.values(), yerr=np.sqrt(hB_ADC.variances())/qcd_mc.values(),linestyle='none', marker='o', color='black')
ax[1].set_ylim(0.95, 1.05)
ax[1].set_xlim(bins[0], bins[-1])
ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].legend()
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/QCD_closure.png", bbox_inches='tight')


# %%

x = (bins[1:] + bins[:-1])/2

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)

#ADC_err = regions['A']*regions['D']/regions['C']*np.sqrt(1/regions['A'] + 1/regions['D'] + 1/regions['C'])
ax[0].errorbar(x, regions['B'].values(), yerr=np.sqrt(regions['B'].variances()) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)

    
cQCD = ax[0].hist(bins[:-1], bins=bins, weights=hB_ADC.values(), bottom=cTot, label='QCD')[0]
cTot = cTot + cQCD
for key in countsDict.keys():
    if np.sum(countsDict[key].values())==0:
        continue
    print(key, countsDict[key].sum())
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key].values(), bottom=cTot, label=key)
    cTot = cTot + countsDict[key].values()
ax[0].legend()
ax[0].set_yscale('log')

ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')

mcPlusQCD = mc.copy()
mcPlusQCD= mcPlusQCD + hB_ADC
ax[1].set_ylim(0.95, 1.05)
ax[1].errorbar(x, regions["B"].values()/mcPlusQCD.values(), yerr=np.sqrt(regions["B"].variances() + hB_ADC.variances())/mcPlusQCD.values() , marker='o', color='black', linestyle='none')
ax[1].set_xlabel("Dijet Mass [GeV]")
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/ZQCDplusSM.png")
# %%

# %%
import ROOT

# Create a ROOT file to store histograms
root_file = ROOT.TFile("/t3home/gcelotto/ggHbb/abcd/combineTry/counts.root", "RECREATE")

# Create histograms for each process
# use same bins
processes = ['H', 'VV', 'ST', 'ttbar', 'W+Jets', 'Z+Jets', 'QCD']
hists = {
    'H':      countsDict["H"]  ,
    'VV':     countsDict["VV"]  ,
    'ST':     countsDict["ST"]  ,
    'ttbar':  countsDict["ttbar"]      ,
    'W+Jets': countsDict["W+Jets"]      ,
    'Z+Jets': countsDict["Z+Jets"]      ,
    'QCD':    hB_ADC,
    'data_obs':    regions["B"]
}

# Create histograms for each process
for proc, hist in hists.items():
    rootHist = ROOT.TH1F(proc, proc, len(bins)-1, bins)
    for i, (value, error) in enumerate(zip(hist.values(), np.sqrt(hist.variances()))):
        rootHist.SetBinContent(i+1, value)
        rootHist.SetBinError(i+1, error)
    rootHist.Write()

# Close the file
root_file.Close()

# %%
assert False
import numpy as np
from scipy.optimize import minimize

# Function to compute the negative log-likelihood (NLL)

def nll(mu_z):
    lambda_exp = (
        countsDict['H'] +
        countsDict['VV'] +
        countsDict['ST'] +
        countsDict['ttbar'] +
        countsDict['W+Jets'] +
        mu_z * countsDict['Z+Jets'] +  # Signal contribution scaled by mu_z
        countsDict['QCD']
    )
    return np.sum(lambda_exp - regions["B"] * np.log(lambda_exp))

# Minimize the NLL to get the best-fit mu_Z
result = minimize(nll, x0=1.0, bounds=[(0, None)])  # mu_Z >= 0
mu_z_best = result.x[0]

# Profile the likelihood for a range of mu_z values around the best-fit value
mu_z_values = np.linspace(0.5, 1.5, 50)  # Range of mu_z to scan
nll_values = [nll(mu_z) for mu_z in mu_z_values]

# Calculate the difference between the minimum NLL and the profile NLL
nll_min = min(nll_values)
delta_nll = np.array([n - nll_min for n in nll_values])

# Define the confidence level for a 68% interval (Delta NLL = 1)
delta_nll_68 = 1

# Find the interval where delta NLL is less than or equal to 1
lower_limit = mu_z_values[np.min(np.where(delta_nll <= delta_nll_68))]
upper_limit = mu_z_values[np.max(np.where(delta_nll <= delta_nll_68))]

print(f"Best-fit signal strength: {mu_z_best:.3f}")
print(f"68% confidence interval: ({lower_limit:.3f}, {upper_limit:.3f})")

# %%



dfs=dfs_precut.copy()

# Compute histogram of highPNN in dijet mass
# Subtract histogram of MC non QCD
pnn_threshold = 0.5
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.3, max=None)
countsHighPNN = np.histogram(dfs[0][dfs[0].PNN>pnn_threshold], bins=bins)[0]
cSMHighPNN = np.zeros(len(bins)-1)

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mHigh = df.PNN > pnn_threshold
    c = np.histogram(df.dijet_mass[mHigh], bins=bins, weights=df.weight[mHigh])[0]
    cSMHighPNN = cSMHighPNN + c
countsHighPNN_SMsub = countsHighPNN - cSMHighPNN
# DO the same for LowpNN
countsLowPNN = np.histogram(dfs[0][dfs[0].PNN<pnn_threshold], bins=bins)[0]
cSMLowPNN = np.zeros(len(bins)-1)

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mHigh = df.PNN < pnn_threshold
    c = np.histogram(df.dijet_mass[mHigh], bins=bins, weights=df.weight[mHigh])[0]
    cSMLowPNN = cSMLowPNN + c
countsLowPNN_SMsub = countsLowPNN - cSMLowPNN
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
ax[0].hist(bins[:-1], bins=bins, weights=countsHighPNN_SMsub/np.sum(countsHighPNN_SMsub), histtype='step', density=True, label='Data - MC(SMnonQCD) PNN>%.1f'%pnn_threshold)
ax[0].hist(bins[:-1], bins=bins, weights=countsLowPNN_SMsub/np.sum(countsLowPNN_SMsub), histtype='step', density=True,label='Data - MC(SMnonQCD) PNN<%.1f'%pnn_threshold)
ax[0].legend()
ax[0].text(x=0.9, y=0.5, s="Jet1_btagDeepFlavB > 0.3", ha='right',transform=ax[0].transAxes)
ax[0].text(x=0.9, y=0.45, s="Jet2_btagDeepFlavB > 0.3",ha='right', transform=ax[0].transAxes)
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[1].errorbar(x = (bins[1:]+bins[:-1])/2, y=countsHighPNN_SMsub/countsLowPNN_SMsub*np.sum(countsLowPNN_SMsub)/np.sum(countsHighPNN_SMsub), yerr=countsHighPNN_SMsub/countsHighPNN_SMsub*np.sum(countsLowPNN_SMsub)/np.sum(countsHighPNN_SMsub)*np.sqrt(1/countsHighPNN_SMsub + 1/countsLowNN_SMsub), linestyle='none', color='black', marker='o')
ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
ax[0].set_xlim(bins[0], bins[-1])
ax[1].set_ylim(.9, 1.1)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/High_Low_PNN_score.png", bbox_inches='tight')