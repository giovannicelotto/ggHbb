import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import uproot
import glob
import sys
import mplhep as hep
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from utilsForPlot import getXSectionBR

def plot():
    print("start")
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB_20UL18"
    bkgPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*"
    nFilesData = 10
    nFilesSignal = 100
    currentLumi = nFilesData * 0.774 / 1017

    signalFileNames = glob.glob(signalPath+"/*.root")[:nFilesSignal]
    bkgFileNames = glob.glob(bkgPath+"/*.root")[:nFilesData]

    bins = np.arange(100)
    np.save("/t3home/gcelotto/ggHbb/PU_reweighting/bins.npy", bins)
    
    counts_ggH = np.zeros(len(bins)-1)
    totalNorm = 0

    
    for fileName in signalFileNames:
        print("%d / %d ..."%(signalFileNames.index(fileName), len(signalFileNames)))
        f =  uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        PV_npvs         = branches["PV_npvs"]
        PV_npvsGood     = branches["PV_npvsGood"]
        
        lumiBlocks = f['LuminosityBlocks']
        numEventsTotal = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])
        totalNorm = totalNorm + numEventsTotal
        
        counts_ggH = counts_ggH + np.histogram(PV_npvs, bins=bins)[0]




    counts_bkg = np.zeros(len(bins)-1)
    for fileName in bkgFileNames:
        print("%d / %d ..."%(bkgFileNames.index(fileName), len(bkgFileNames)))
        f =  uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        PV_npvs         = branches["PV_npvs"]
        PV_npvsGood     = branches["PV_npvsGood"]
        
        counts_bkg = counts_bkg + np.histogram(PV_npvs, bins=bins)[0]

    counts_ggH = counts_ggH/totalNorm  * getXSectionBR() * currentLumi * 1000
    counts_ggH = counts_ggH/np.sum(counts_ggH)
    counts_bkg = counts_bkg/np.sum(counts_bkg)
    
    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    
    ax1.errorbar((bins[1:]+bins[:-1])/2, counts_ggH, xerr=np.diff(bins)/2, marker='o', color='black', linestyle='none', label='ggH')
    ax1.errorbar((bins[1:]+bins[:-1])/2, counts_bkg, xerr=np.diff(bins)/2, marker='o', color='red', linestyle='none', label='Data')
    ax2.errorbar((bins[1:]+bins[:-1])/2, (counts_bkg/counts_ggH), marker='o', color='black', linewidth=1, linestyle='', label='MC', alpha=1)
    ax2.set_ylabel("Data/MC", fontsize=28)
    ax1.legend()
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    fig.savefig("/t3home/gcelotto/ggHbb/PU_reweighting/PU_ggH_vs_data.png", bbox_inches='tight')
    




    return

if __name__ == "__main__":
    plot()