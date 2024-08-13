import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re, sys
import mplhep as hep
hep.style.use("CMS")


def main(nData, nMC):
    miniDf = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_June.csv")
    datasets = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")



    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 12))
    bins= np.linspace(0, 300, 101)
    allCounts = np.zeros(len(bins)-1)

    countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
    }

    lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")


    for (process, path, xsection) in zip(datasets.process, datasets.flatPath, datasets.xsection):
        nFiles = nData if process=='Data' else nMC
        fileNames = glob.glob(path+"/**/*.parquet", recursive=True)[:nFiles]
        if len(fileNames)==0:
            sys.exit("No file found for process %s"%process)
        df = pd.read_parquet(fileNames, columns=['dijet_mass', 'sf', 'dijet_pt', 'jet2_btagDeepFlavB'])
        if process=='Data':
            currentLumi=lumiPerEvent*len(df)
            print(currentLumi)
        else:
            mini=0
            for fileName in fileNames:
                fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))    
                mini = mini + miniDf[(miniDf.process==process) & (miniDf.fileNumber == fileNumber)].numEventsPassed.values[0]

            xsection = datasets[datasets.process==process].xsection.values[0]*1000
        

        df=df[(df.dijet_pt>100) & (df.jet2_btagDeepFlavB>0.8)]
        counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
        if process=='Data':
            pass
        else:
            counts=counts*xsection / mini*currentLumi
        if 'Data' in process:
            countsDict['Data'] = countsDict['Data'] +counts
        elif 'GluGluHToBB' in process:
            
            countsDict['H'] = countsDict['H'] +counts
        elif 'ST' in process:
            countsDict['ST'] = countsDict['ST'] +counts
        elif 'TTTo' in process:
            countsDict['ttbar'] = countsDict['ttbar'] +counts
        elif 'QCD' in process:
            countsDict['QCD'] = countsDict['QCD'] +counts
        elif 'ZJets' in process:
            countsDict['Z+Jets'] = countsDict['Z+Jets'] +counts
        elif 'WJets' in process:
            countsDict['W+Jets'] = countsDict['W+Jets'] +counts
        elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
            countsDict['VV'] = countsDict['VV'] +counts

    for key in countsDict.keys():
        if key=='Data':
            ax1.errorbar((bins[1:]+bins[:-1])/2, countsDict[key], xerr=np.diff(bins)/2, yerr=np.sqrt(countsDict[key]), color='black', linestyle='none')[0]
        elif key=='H':
            ax1.errorbar((bins[1:]+bins[:-1])/2, countsDict[key], xerr=np.diff(bins)/2, color='darkgreen', linestyle='none')
        else:
            print(key, np.sum(countsDict[key]))
            ax1.hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=allCounts, label=key)
            allCounts = allCounts + countsDict[key]
        

    ax1.set_yscale('log')
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    ax1.set_ylim(1, ax1.get_ylim()[1]*1.3)
    ax1.set_xlim(bins[0], bins[-1])
    ax2.set_xlim(bins[0], bins[-1])
    
    ax1.legend()


    
    ax2.errorbar((bins[:-1]+bins[1:])/2, countsDict['Data']/allCounts, xerr=np.diff(bins)/2, color='black', linestyle='none')
    ax2.set_ylim(0.5, 1.5)
    ax2.hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
    fig.savefig("/t3home/gcelotto/ggHbb/bkgEstimation/output/dijet_mass.png", bbox_inches='tight')

if __name__=="__main__":
    nData, nMC = int(sys.argv[1]), int(sys.argv[2])
    main(nData, nMC)