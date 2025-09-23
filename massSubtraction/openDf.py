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
    modelName = "Jul15_3_20p0"
    dd = False
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
outFolder = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName 
#mass_bins = np.load(outFolder+"/mass_bins.npy")
# %%
dfs = []
dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
dfsMC = []
# %%
isMCList = [#0,
            1, 
            #2,
            3, 4,
            #5,6,7,8, 9,10,
            #11,12,13,
            #14,
    #15,
    #        16,
    #        17,18,
            19,20,21,22,
            #35,
            36,
            37,
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
            #2,
            #3,4,5,
            #6,7,8,9,
            #10,
            #11,12,13,14,15,
            #16,
            17,
            #18,
            #19,20,21
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
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
dfsData = cut(dfsData, 'dijet_pt', 100, None)
#dfsMC = cut(dfsMC, 'PNN', 0.4,None)
#dfsData = cut(dfsData, 'PNN', 0.4,None)
#dfsData = cut(dfsData, 'dijet_pt', 140, None)
# %%
fig = plotDfs(dfsData=dfsData, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=lumi, log=True, blindPar=(False, 125, 20))
# %%
df=pd.concat(dfsData)
dfZ = pd.concat(dfsMC[0:-1])
dfH = dfsMC[-1]
#%%
import matplotlib.pyplot as plt
from hist import Hist
bins = np.linspace(80, 160, 41)
h_low = Hist.new.Var(bins, name="mjj").Weight()
h_high = Hist.new.Var(bins, name="mjj").Weight()
h_low_Z = Hist.new.Var(bins, name="mjj").Weight()
h_high_Z = Hist.new.Var(bins, name="mjj").Weight()
h_low_H = Hist.new.Var(bins, name="mjj").Weight()
h_high_H = Hist.new.Var(bins, name="mjj").Weight()

t = 0.6
t_min = 0.5
t_max = 0.7

h_low.fill(df.dijet_mass[(df.PNN<t) & (df.PNN>t_min) & (df.PNN<t_max)])
h_high.fill(df.dijet_mass[(df.PNN>t) & (df.PNN>t_min) & (df.PNN<t_max)])

h_low_Z.fill(dfZ.dijet_mass[(dfZ.PNN<t) & (dfZ.PNN>t_min) & (dfZ.PNN<t_max)])
h_high_Z.fill(dfZ.dijet_mass[(dfZ.PNN>t) & (dfZ.PNN>t_min) & (dfZ.PNN<t_max)])

h_low_H.fill(dfH.dijet_mass[(dfH.PNN<t) & (dfH.PNN>t_min) & (dfH.PNN<t_max)])
h_high_H.fill(dfH.dijet_mass[(dfH.PNN>t) & (dfH.PNN>t_min) & (dfH.PNN<t_max)])

#for idx, df_ in enumerate(dfsMC):
#    print(dfProcessesMC.process[isMCList[idx]])
#    m = (df_.PNN>t)  & (df_.PNN<0.7)
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
ax[1].errorbar(x, h_high.values() - h_low.values()*np.sum(h_high.values())/np.sum(h_low.values()), yerr = np.sqrt(h_high.variances() + h_low.variances()*np.sum(h_high.values())/np.sum(h_low.values())) , color='red', linestyle='none', marker='o')
ax[1].errorbar(x, h_high_Z.values() - h_low_Z.values()*np.sum(h_high.values())/np.sum(h_low.values()), yerr = np.sqrt(h_high_Z.variances() + h_low_Z.variances()*np.sum(h_high.values())/np.sum(h_low.values())) , color='green', linestyle='none', marker='o')
ax[1].errorbar(x, h_high_H.values() - h_low_H.values()*np.sum(h_high.values())/np.sum(h_low.values()), yerr = np.sqrt(h_high_H.variances() + h_low_H.variances()*np.sum(h_high.values())/np.sum(h_low.values())) , color='black', linestyle='none', marker='o')


#ax[1].hist(bins[:-1], bins=bins, weights=cz1-cz2)
#$ax[1].set_ylim(-600, None)
ax[1].set_ylabel("High - Low")
ax[0].legend()
ax[0].set_xlim(bins[0], bins[-1])
ax[1].hlines(xmin=bins[0], xmax=bins[-1], y=1, color='black')





# %%
dfZ = pd.concat(dfsMC[0:-1])
dfH = dfsMC[-1]
fig, ax = plt.subplots(1, 1)
ax.hist(df.PNN, bins=np.linspace(0, 1, 51), histtype='step', color='black', density=True, label='Data', weights=df.weight)
ax.hist(dfH.PNN, bins=np.linspace(0, 1, 51), histtype='step', color='red', density=True, label='Higgs', weights=dfH.weight)
ax.hist(dfZ.PNN, bins=np.linspace(0, 1, 51), histtype='step', color='green', density=True, label='Z', weights=dfZ.weight)
ax.set_xlabel("NN score")
ax.set_ylabel("Density")
ax.legend()
# %%
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
        dfsMC[idx]['process']="Higgs"
    elif 'ST' in process:
        countsDict['ST'] = countsDict['ST'] + c
    elif 'TTTo' in process:
        countsDict['ttbar'] = countsDict['ttbar'] + c
    elif 'QCD' in process:
        countsDict['QCD'] = countsDict['QCD'] + c
    elif 'ZJets' in process:
        #print(process, c)
        countsDict['Z+Jets'] = countsDict['Z+Jets'] + c
        dfsMC[idx]['process']="Z"
    elif 'WJets' in process:
        print("Here")
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
#fig, ax  =plt.subplots(1, 1)
#pnn_t = 0.7
#ax.hist(dfMC.dijet_mass[dfMC.PNN>pnn_t], bins=mass_bins, weights=dfMC.weight[dfMC.PNN>pnn_t], histtype='step', color='black', label='MC')
#ax.hist(dfdata.dijet_mass[dfdata.PNN>pnn_t], bins=mass_bins, weights=dfdata.weight[dfdata.PNN>pnn_t], histtype='step', label='Data')
##ax.set_yscale('log')
#ax.legend()
# %%

dfdata = dfdata[(dfdata.jet1_btagDeepFlavB>0.71) & (dfdata.jet2_btagDeepFlavB>0.71)]

import hist
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
thresholds = np.array([0, 0.3, 0.5, 0.7, 1])
bins=np.linspace(50, 300, 51)
bin1 = bins[np.abs(bins - 105).argmin()]
bin2 = bins[np.abs(bins - 140).argmin()]
inclusive_hist = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
inclusive_hist.fill(dfdata.dijet_mass[((dfdata.dijet_mass < bin1) | (dfdata.dijet_mass > bin2)) & (dfdata.dijet_mass>bins[0])])
inclusive_hist.plot(density=True, ax=ax[0], label="Inclusive", color='black')


for t_b, t_a in zip(thresholds[:-1], thresholds[1:]):
    
    mask = (dfdata.PNN > t_b) & (dfdata.PNN < t_a) & ((dfdata.dijet_mass < bin1) | (dfdata.dijet_mass > bin2)) & (dfdata.dijet_mass>bins[0])
    h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
    h.fill(dfdata.dijet_mass[mask].values)
    
    # Plot step hist for the upper plot
    h.plot(density=True, ax=ax[0], label=f"{t_b:.2f} < NN < {t_a:.2f}")

    normH = (h/h.integrate(0).value)/((inclusive_hist.values()/inclusive_hist.integrate(0).value))
    normH.plot(ax=ax[1])
    
    # Ratio wrt inclusive
    #ratio = np.divide(h.view(), inclusive_hist.view(), out=np.zeros_like(h.view()), where=inclusive_hist.view()!=0)
    #ax[1].step(h.axes[0].centers, ratio, where='mid', label=f"{t_b:.2f} < NN < {t_a:.2f}")

# Inclusive in upper plot
#ax[0].step(inclusive_hist.axes[0].centers, inclusive_hist.view()/inclusive_hist.view().sum(), color='black', label='Inclusive')

# Axes labels and legend
ax[0].set_xlim(bins[0], bins[-1])
ax[0].legend()
ax[0].set_xlabel("")
ax[0].set_ylabel("Density [a.u.]")
ax[1].axhline(1, color='black', linestyle='--')
ax[1].set_ylim(0.8, 1.2)
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[1].set_ylabel("Ratio / Inclusive")


# %%
NN_t, factorHiggs, factorZ = 0.7, 300,11


fig, ax = plt.subplots(1, 1)
thresholds = np.array([0, 0.3, 0.5, 0.7, 1])
bins=np.linspace(50, 300, 101)
x = (bins[:-1]+bins[1:])/2
inclusive_hist = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
maskData = ((dfdata.dijet_mass < bin1) | (dfdata.dijet_mass > bin2)) & (dfdata.dijet_mass>bins[0]) & (dfdata.PNN>NN_t)
inclusive_hist.fill(dfdata.dijet_mass[maskData])
#inclusive_hist.plot(density=False, ax=ax, label="Inclusive", color='black')
ax.errorbar(x, inclusive_hist.values(), yerr=np.sqrt(inclusive_hist.values()), marker='o', linestyle='none', color='black', markersize=2, label='Data')
h_Higgs = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
mask = (dfMC.process=='Higgs') & (dfMC.PNN > NN_t)
h_Higgs.fill(dfMC.dijet_mass[mask].values, weight=dfMC.weight[mask]*factorHiggs)
h_Higgs.plot(ax=ax, label='ggF Higgs x %d'%factorHiggs)

h_Z = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
mask = (dfMC.process=='Z') & (dfMC.PNN > NN_t)
h_Z.fill(dfMC.dijet_mass[mask].values, weight=dfMC.weight[mask]*factorZ)
h_Z.plot(ax=ax, label='Z x %d'%factorZ)
ax.legend()
if NN_t>0:
    ax.text(x=0.95, y=0.65, s="NN score > %.1f"%NN_t, transform=ax.transAxes, ha='right')
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Density [a.u.]")
hep.cms.label("Private Work", exp='CMS', data=True,  lumi = np.round(lumi, 2))

# %%




dfdata_mass = dfdata [(dfdata.dijet_mass>105) & (dfdata.dijet_mass<140)]
dfHiggs_mass = dfMC [(dfMC.dijet_mass>105) & (dfMC.dijet_mass<140) & (dfMC.process=="Higgs")]


QCD_gain = len(dfdata_mass[dfdata_mass.PNN>0.7]) / len(dfdata_mass)
Higgs_gain = ((dfHiggs_mass[dfHiggs_mass.PNN>0.7].weight.sum()) / dfHiggs_mass.weight.sum())
print(Higgs_gain/QCD_gain)
# %%
