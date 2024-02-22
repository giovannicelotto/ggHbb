import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import matplotlib.patches as patches
import mplhep as hep
hep.style.use("CMS")

def main():
    countsDic, errorsDic = {}, {}
    print("Start")
    df = pd.read_csv("/t3home/gcelotto/ggHbb/bkgEstimation/output/processes.csv")
    currentLumi = np.load('/t3home/gcelotto/ggHbb/bkgEstimation/output/currentLumi.npy')
    bins = np.load('/t3home/gcelotto/ggHbb/bkgEstimation/output/binsForHT.npy')
    allCounts = np.zeros(len(bins)-1)
    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    for (process, path, xsection) in zip(df.iloc[:,0], df.path, df.xsection):
        
        mini = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/output/mini_%s.npy"%process)
        counts = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/output/counts_%s.npy"%process)
        #print(process, mini, counts, xsection, currentLumi)
        if process != 'Data':
            errorsDic[process] = np.sqrt(counts)
            counts = counts * xsection*1000*currentLumi/mini
            errorsDic[process] = errorsDic[process]* xsection*1000*currentLumi/mini
            countsDic[process] = counts
        else :
            dataCounts = counts
            dataErr = np.sqrt(dataCounts)
            ax1.errorbar((bins[1:]+bins[:-1])/2, counts, xerr=np.diff(bins)/2, yerr=dataErr, marker='o', color='black', linestyle='none', label='Data')
            

    #
    compactDict = {}
    compactDict['VV'] = countsDic['WW'] + countsDic['WZ'] + countsDic['ZZ']
    qcd_sum, singleBoson, singleTop=0, 0, 0
    for key, value in countsDic.items():
        if 'QCD' in key:
            qcd_sum += value
        elif 'WJets' in key or 'ZJets' in key:
            singleBoson+=value
        elif 'ST_' in key:
            singleTop+=value
    compactDict['ST']= singleTop
    compactDict['tt'] = countsDic['TTTo2L2Nu'] +     countsDic['TTToHadronic'] + countsDic['TTToSemiLeptonic']
    compactDict['W/Z+Jets']= singleBoson
    compactDict['QCD']= qcd_sum
    # plot the MC processes compacted in categories
    for key in compactDict.keys():
        ax1.bar((bins[1:]+bins[:-1])/2, compactDict[key] , align='center', width=np.diff(bins), label=key, alpha = 1, bottom=allCounts)
        allCounts = allCounts + compactDict[key]
    # compute the errors total for each bin by summing in quadrature
    totalErr = np.zeros(len(bins)-1)
    for key in errorsDic.keys():
        totalErr = np.sqrt(totalErr**2 + errorsDic[key]**2)
    for i in range(len(bins)-1):
        if i ==0:
            rect = patches.Rectangle((bins[i], allCounts[i] - totalErr[i]),
                    bins[i+1]-bins[i], 2 *  totalErr[i],
                    linewidth=0, edgecolor='black', facecolor='none', hatch='///', label='Uncertainty')
            
            rect2 = patches.Rectangle((bins[i], (allCounts[i] - totalErr[i])/allCounts[i]),
                    bins[i+1]-bins[i], (2 *  totalErr[i])/allCounts[i],
                    linewidth=0, edgecolor='black', facecolor='none', hatch='///', label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], allCounts[i] - totalErr[i]),
                    bins[i+1]-bins[i], 2 *  totalErr[i],
                    linewidth=0, edgecolor='black', facecolor='none', hatch='///')
            
            rect2 = patches.Rectangle((bins[i], (allCounts[i] - totalErr[i])/allCounts[i]),
                    bins[i+1]-bins[i], (2 *  totalErr[i])/allCounts[i],
                    linewidth=0, edgecolor='black', facecolor='none', hatch='///')
        ax1.add_patch(rect)
        ax2.add_patch(rect2)
    
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    ax1.set_yscale('log')
    ax1.set_xlim(bins[0], bins[-1])
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_ylim(0, 2)
    ax1.set_ylim(10**0, ax1.get_ylim()[1])
    ax2.hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
    ax2.set_xlabel("$\mathrm{H_{T}}$ [GeV]")
    ax1.set_ylabel("Events")
    ax2.set_ylabel("Data/MC", fontsize=28)
    ax2.errorbar((bins[1:]+bins[:-1])/2, (dataCounts/allCounts), marker='o', color='black', linewidth=1, linestyle='', label='MC', alpha=1)
    ax1.legend(bbox_to_anchor=(1 ,1), ncols=1)
    #ax1.text(s="%.2f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax1.transAxes, fontsize=16)
    outName = "/t3home/gcelotto/ggHbb/bkgEstimation/output/closure_parallel.png"
    print("Savin in ", outName)
    fig.savefig(outName, bbox_inches='tight')


    print("MC/Data", allCounts/dataCounts)
        



if __name__ == "__main__":
    main()