# %%
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from plotDfs import plotDfs
from functions import getDfProcesses_v2, cut
import numpy as np
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
    modelName = "Feb17_900p2"
    dd = False
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
outFolder = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName 
#mass_bins = np.load(outFolder+"/mass_bins.npy")
# %%
dfs = []
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
dfsMC = []
# %%
isMCList = [0,
            1, 
            #2,3, 4,
            #5,6,7,8, 9,10,
            #11,12,13,
            #14,15,16,17,18,
            19,20,21, #22,
            #35,
            36
            ]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s_%s.parquet"%(p, modelName))
    dfsMC.append(df)
# %%
dfsData = []
isDataList = [
            #0,
            #1, 
            2,
            #3
            ]

lumis = []
for idx, p in enumerate(dfProcessesData.process):
    if idx not in isDataList:
        continue
    df = pd.read_parquet(df_folder+"/dataframes_%s_%s.parquet"%(p, modelName))
    print("/dataframes_%s_%s.parquet"%(p, modelName))

    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi_%s_%s.npy"%(p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight =dfsMC[idx].weight*lumi
print("Lumi total is %.2f fb-1"%lumi)
# %%
#dfsMC = cut(dfsMC, 'dijet_pt', 140, None)
#dfsData = cut(dfsData, 'dijet_pt', 140, None)
# %%
fig = plotDfs(dfsData=dfsData, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=lumi, log=True, blindPar=(False, 125, 20))
# %%
df=pd.concat(dfsData)

import matplotlib.pyplot as plt
from hist import Hist
bins = np.linspace(100, 150, 201)
h_low = Hist.new.Var(bins, name="mjj").Weight()
h_high = Hist.new.Var(bins, name="mjj").Weight()

t = 0.

h_low.fill(df.dijet_mass[df.PNN<t])
h_high.fill(df.dijet_mass[df.PNN>t])

for idx, df in enumerate(dfsMC):
    print(dfProcessesMC.process[isMCList[idx]])
    m = df.PNN>t
    #h_low.fill(df.dijet_mass[~m], weight=-df.weight[~m])
    #h_high.fill(df.dijet_mass[m], weight=-df.weight[m])

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])

x =(bins[1:] + bins[:-1])/2
ax[0].errorbar(x, h_low.values()/np.sum(h_low.values()), yerr=np.sqrt(h_low.variances())/np.sum(h_low.values()), label='NN < %.1f'%t, color='blue', linestyle='none', marker='o')
ax[0].errorbar(x, h_high.values()/np.sum(h_high.values()), yerr=np.sqrt(h_high.variances())/np.sum(h_high.values()), label='NN > %.1f'%t, color='red', linestyle='none', marker='o')[0]
c_low=h_low.values()/np.sum(h_low.values())
c_high=h_high.values()/np.sum(h_high.values())
err_c_low = np.sqrt(h_low.variances())/np.sum(h_low.values())
err_c_high = np.sqrt(h_high.variances())/np.sum(h_high.values())

#ax[1].errorbar(x, c_high/c_low, yerr = c_high/c_low * np.sqrt((err_c_low/c_low)**2 + (err_c_high/c_high)**2) , color='red', linestyle='none', marker='o')
ax[1].errorbar(x, h_high.values() - h_low.values()*np.sum(h_high.values())/np.sum(h_low.values()), yerr = np.sqrt(h_high.variances() + h_low.variances()) , color='red', linestyle='none', marker='o')

#ax[1].hist(bins[:-1], bins=bins, weights=cz1-cz2)
ax[1].set_ylim(-600, None)
ax[1].set_ylabel("Ratio")
ax[0].legend()
ax[0].set_xlim(bins[0], bins[-1])
ax[1].hlines(xmin=bins[0], xmax=bins[-1], y=1, color='black')
# %%
dfZ = pd.concat(dfsMC[1:-1])
dfH = dfsMC[-1]
cz1 = np.histogram(dfZ.dijet_mass[dfZ.PNN>t], bins=bins, weights=dfZ.weight[dfZ.PNN>t])[0]
cH = np.histogram(dfH.dijet_mass[dfH.PNN>t], bins=bins, weights=dfH.weight[dfH.PNN>t])[0]
fig, ax = plt.subplots(1, 1)
ax.errorbar(x, h_high.values(), yerr=np.sqrt(h_high.variances()), marker='o', color='black', linestyle='none')
ax.errorbar(x, cH, marker='o', color='black', linestyle='none')
#ax.fill_between(x, h_high.values()-cz1, h_high.values(),  color='blue')
ax.set_yscale('log')
# %%
for idx, p in enumerate(dfProcessesMC.process):
    print(p)
    
# %%
fig,ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 51)
cDataTot =np.zeros(len(bins)-1)
for df in dfsData:
    c = np.histogram(df.PNN, bins=bins)[0]
    print(c)
    cDataTot = cDataTot + c
x = (bins[:-1] + bins[1:])/2
#ax.errorbar(x, cDataTot, yerr=np.sqrt(cDataTot), linestyle='none', color='black', marker='o')

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
cTot = np.zeros(len(bins)-1)
for idx, df in enumerate(dfsMC):
    isMC = isMCList[idx]
    process = dfProcessesMC.process[isMC]
    print(idx, process, isMC)
    c = np.histogram(df.PNN, bins=bins,weights=df.weight)[0]
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


for key in countsDict.keys():
    print(key, np.sum(countsDict[key]))
    if np.sum(countsDict[key])==0:
        continue
    ax.hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key, histtype='step')
    print(cTot)
    cTot = cTot + countsDict[key]

ax.legend(loc='upper right')
#ax.set_yscale('log')
#ax.set_ylim(0, 500)
ax.set_ylabel("Counts")
ax.set_xlabel("PNN")
import mplhep as hep
hep.style.use("CMS")
hep.cms.label(lumi=round(lumi,2))

# %%
dfMC = pd.concat(dfsMC)
dfdata = pd.concat(dfsData)

# %%
fig, ax  =plt.subplots(1, 1)
pnn_t = 0.7
ax.hist(dfMC.dijet_mass[dfMC.PNN>pnn_t], bins=mass_bins, weights=dfMC.weight[dfMC.PNN>pnn_t], histtype='step', color='black', label='MC')
ax.hist(dfdata.dijet_mass[dfdata.PNN>pnn_t], bins=mass_bins, weights=dfdata.weight[dfdata.PNN>pnn_t], histtype='step', label='Data')
#ax.set_yscale('log')
ax.legend()
# %%
