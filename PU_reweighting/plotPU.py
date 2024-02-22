import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import uproot
import glob
import sys
import mplhep as hep
import subprocess
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from utilsForPlot import getXSectionBR


def main():
    df                  = pd.read_csv("/t3home/gcelotto/ggHbb/PU_reweighting/output/processes.csv")
    currentLumi         = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/currentLumi.npy")
    bins                = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/bins.npy")
    
    allCounts = np.zeros(len(bins)-1)
    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    
    for (process, path, xsection) in zip(df.iloc[:,0], df.path, df.xsection):
        counts = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/counts_%s.npy" % process)
        
        if process !='Data':
            mini = np.load("/t3home/gcelotto/ggHbb/PU_reweighting/output/numEventsTotal_%s.npy"%process)
            counts = counts * currentLumi*xsection *1000/ mini
            allCounts=allCounts+counts
        else:
            dataCounts = counts
    allCounts = allCounts/np.sum(allCounts)
    dataCounts = dataCounts/np.sum(dataCounts)

    print(dataCounts)
    print(allCounts)
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Number of PV")
    ax1.errorbar((bins[1:]+bins[:-1])/2, dataCounts, xerr=np.diff(bins)/2, marker='o', color='black', linestyle='none', label='Data')
    ax1.errorbar((bins[1:]+bins[:-1])/2, allCounts, xerr=np.diff(bins)/2, marker='o', color='red', linestyle='none', label='MC')
    ax2.errorbar((bins[1:]+bins[:-1])/2, (dataCounts/allCounts), marker='o', color='black', linewidth=1, linestyle='', label='MC', alpha=1)
    ax2.set_ylabel("Data/MC", fontsize=28)
    ax2.set_ylim(0, 2)
    ax1.legend()
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    outName = "/t3home/gcelotto/ggHbb/PU_reweighting/PU_ggH_vs_data.png"
    print("Saving in %s"%outName)
    fig.savefig(outName, bbox_inches='tight')

if __name__=="__main__":
    main()