import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplhep as hep
hep.style.use("CMS")

def plotDfs(dfs, isMCList, dfProcesses, nbin, nReal, log=True):
    fig, ax =plt.subplots(1, 1)
    bins_mass = np.linspace(40, 300, nbin)
    c = np.histogram(dfs[0].dijet_mass, bins=bins_mass)[0]
    x = (bins_mass[:-1] + bins_mass[1:])/2
    ax.errorbar(x, c, yerr=np.sqrt(c), linestyle='none', color='black', marker='o')
    countsDict = {
            'Data':np.zeros(len(bins_mass)-1),
            'H':np.zeros(len(bins_mass)-1),
            'VV':np.zeros(len(bins_mass)-1),
            'ST':np.zeros(len(bins_mass)-1),
            'ttbar':np.zeros(len(bins_mass)-1),
            'W+Jets':np.zeros(len(bins_mass)-1),
            'QCD':np.zeros(len(bins_mass)-1),
            'Z+Jets':np.zeros(len(bins_mass)-1),
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
    if log:
        ax.set_yscale('log')
        ax.set_ylim(10, ax.get_ylim()[1])
    ax.set_ylabel("Counts")
    ax.set_xlabel("Dijet Mass [GeV]")
    hep.cms.label(lumi=round(nReal*0.774/1017,2))


    return fig
