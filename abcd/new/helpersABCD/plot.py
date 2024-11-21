import numpy as np
import matplotlib.pyplot as plt
from hist import Hist
import mplhep as hep
hep.style.use("CMS")
def plotQCDhists_SR_CR(regions, bins, x1, x2, t1, t2, suffix):
    
    hB_ADC_values = regions['A'].values()*regions['D'].values()/(regions['C'].values()+1e-6)
    ADC_err = regions['A'].values()*regions['D'].values()/regions['C'].values()*np.sqrt(1/regions['A'].values() + 1/regions['D'].values() + 1/regions['C'].values())
    hB_ADC = Hist.new.Reg(len(bins) - 1, bins[0], bins[-1], name='mjj').Weight()
    hB_ADC.values()[:] = hB_ADC_values
    hB_ADC.variances()[:] = ADC_err**2 


    # Plot Data and Data and B=A*D/C estimation in SR
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
    x=(bins[1:]+bins[:-1])/2
    ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"].values(), histtype=u'step', label='Region A')
    ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"].values(), histtype=u'step', label='Region B')
    ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'].values(), histtype=u'step', label='Region C')
    ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'].values(), histtype=u'step', label='Region D')


    ax[0,1].hist(bins[:-1], bins=bins, weights=hB_ADC.values(), histtype=u'step', label=r'$A\times D / C$ ')
    ax[0,1].errorbar(x, regions["B"].values(), yerr=np.sqrt(regions["B"].variances()), linestyle='none', color='black', marker='o')
    
    ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t1, x2, t2), fontsize=14)
    ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t1, x2, t2), fontsize=14)
    ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t1, x2, t2), fontsize=14)
    ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t1, x2, t2), fontsize=14)
    for idx, axx in enumerate(ax.ravel()):
        axx.set_xlim(bins[0], bins[-1])
        axx.set_xlabel("Dijet Mass [GeV]")
        axx.legend(fontsize=18, loc='upper right')
    fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/QCDhists_SR_CR_%s.png"%suffix, bbox_inches='tight')


    return hB_ADC



def controlPlotABCD(regions, hB_ADC, bins, dfs, isMCList, dfProcesses, x1, t1, x2, t2, nReal, suffix):
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
        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 

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

    # Define a new histogram for SM contributions
    countsDict['mc'] = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
    ax[0].bar(x, 2*np.sqrt(countsDict['mc'].variances()), width=np.diff(bins), bottom=countsDict['mc'].values() - np.sqrt(countsDict['mc'].variances()), 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")

    ax[1].bar(x, 2*np.sqrt(countsDict['mc'].variances())/countsDict['mc'].values(), width=np.diff(bins), bottom=1 - np.sqrt(countsDict['mc'].variances())/countsDict['mc'].values(), 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")


    ax[1].set_ylim(0., 2)
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].errorbar(x, hExcess.values()/countsDict['mc'].values(), yerr=np.sqrt(hExcess.variances())/countsDict['mc'].values() , marker='o', color='black', linestyle='none')
    hep.cms.label(lumi=np.round(nReal*0.774/1017, 3), ax=ax[0])
    fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/SMnonQCD_closure_%s.png"%suffix)
    return countsDict




def plotQCDClosure(bins, hB_ADC, qcd_mc, suffix):
    x = (bins[1:] + bins[:-1])/2
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    ax[0].hist(bins[:-1], bins=bins, weights=(hB_ADC.values()), label='QCD = ABCD estimation', histtype='step')
    ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc.values(), label='QCD = B - MC estimation[B]', histtype='step')
    ax[1].errorbar(x, hB_ADC.values()/qcd_mc.values(), yerr=np.sqrt(hB_ADC.variances())/qcd_mc.values(),linestyle='none', marker='o', color='black')
    ax[1].set_ylim(0.95, 1.05)
    ax[1].set_xlim(bins[0], bins[-1])
    ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1])
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[0].legend()
    fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/QCD_closure_%s.png"%suffix, bbox_inches='tight')


def plotQCDPlusSM(bins, regions, countsDict, hB_ADC, suffix):
    x = (bins[1:] + bins[:-1])/2

    labels = ['Data', 'H', 'VV', 'ST', 'ttbar', 'W+Jets', 'QCD', 'Z+Jets']

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    ax[0].errorbar(x, regions['B'].values(), yerr=np.sqrt(regions['B'].variances()) , marker='o', color='black', linestyle='none')
    cTot = np.zeros(len(bins)-1)
    cQCD = ax[0].hist(bins[:-1], bins=bins, weights=hB_ADC.values(), bottom=cTot, label='QCD')[0]
    cTot = cTot + cQCD
    for key in labels:
        if np.sum(countsDict[key].values())==0:
            continue
        print(key, countsDict[key].sum())
        ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key].values(), bottom=cTot, label=key)
        cTot = cTot + countsDict[key].values()
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[1].set_xlim(ax[1].get_xlim())    
    ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
    mcPlusQCD = countsDict["mc"].copy()
    mcPlusQCD= mcPlusQCD + hB_ADC
    ax[1].set_ylim(0.95, 1.05)
    ax[1].errorbar(x, regions["B"].values()/mcPlusQCD.values(), yerr=np.sqrt(regions["B"].variances())/mcPlusQCD.values() , marker='o', color='black', linestyle='none')
    ax[1].set_xlabel("Dijet Mass [GeV]")
    fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/ZQCDplusSM_%s.png"%suffix)