import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak

import mplhep as hep
hep.style.use("CMS")

def main():
    print("Start")
    df = pd.read_csv("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/processes.csv")
    currentLumi = np.load('/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/currentLumi.npy')
    bins = np.load('/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/binsForHT.npy')
    allCounts = np.zeros(len(bins)-1)
    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    for (process, path, xsection) in zip(df.iloc[:,0], df.path, df.xsection):
        
        mini = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/mini_%s.npy"%process)
        counts = np.load("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/counts_%s.npy"%process)
        #print(process, mini, counts, xsection, currentLumi)
        if process != 'Data':
            counts = counts * xsection*1000*currentLumi/mini
        else :
            dataCounts = counts
            
        
        print(process, "\n", counts, "\n\n\n")
        
        if process == 'Data':
            ax1.errorbar((bins[1:]+bins[:-1])/2-0.5, counts, xerr=np.diff(bins)/2, marker='o', color='black', linestyle='none', label='Data')
        else:
            ax1.bar((bins[1:]+bins[:-1])/2-0.5, counts , align='center', width=np.diff(bins), label=process, alpha = 1, bottom=allCounts)
            allCounts=np.array(allCounts) + np.array(counts)

    ax2.set_xlabel("$\mathrm{H_{T}}$")
    ax1.set_ylabel("Events")
    ax1.set_yscale('log')
    ax2.set_ylabel("Data/MC", fontsize=28)
    ax2.errorbar((bins[1:]+bins[:-1])/2-0.5, (allCounts/(dataCounts)), marker='o', color='black', linewidth=1, linestyle='', label='MC', alpha=1)
    ax1.legend(bbox_to_anchor=(1 ,1))
    outName = "/t3home/gcelotto/ggHbb/outputs/bkgEstimation/closure_parallel.png"
    print("Savin in ", outName)
    fig.savefig(outName, bbox_inches='tight')


    print("MC/Data", allCounts/dataCounts)
        



if __name__ == "__main__":
    main()