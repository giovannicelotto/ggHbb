import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from utilsForPlot import loadData
# Loading files
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withoutArea"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData"

signal, realData = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=100, nRealDataFiles=100)

# Correction factors and counters
N_mini = np.load("/t3home/gcelotto/bbar_analysis/output/N_mini.npy")
totalNanoEntries = 190153970
correctionData = totalNanoEntries/len(realData)
totalSignalCounts = 0                               # total counts for the signal after rescaled
totalData = 0                   

# Plot
fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharey=False)
#fig.subplots_adjust(wspace=0.01)

# invariant mass bins and # Classes of pT
x1, x2, nbin = 0, 300, 51
bins = np.linspace(x1, x2, nbin)
pTBins = [0, 40, 100, 230, 9999999]

#bins for showing pT distribution
pTDiffBins = np.linspace(0, 500, 101)

for idx, ax in enumerate(axes[1,:]):
    pTSignalCounts = np.histogram (np.clip(signal[:,0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]/N_mini*30.52*0.67*1000
    pTrealDataCounts = np.histogram (np.clip(realData[:,0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, histtype=u'step', color='blue')
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, histtype=u'step', color='red')
    pTSignalCounts = np.histogram (np.clip(signal[(signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1]),0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]/N_mini*30.52*0.67*1000
    pTrealDataCounts = np.histogram (np.clip(realData[(realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1]),0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, alpha=0.4, color='blue')
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, alpha=0.4, color='red')

    ax.set_xlabel("Dijet Pt [GeV]", fontsize=16)
    ax.set_yscale('log')
    ax.set_xlim(0, 500)
axes[1,0].set_ylabel("Events / %.1f GeV"%(pTDiffBins[1]-pTDiffBins[0]), fontsize=16)
for idx, ax in enumerate(axes[0,:]):
    maskSignal = (signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1]) #& (signal[:,2]>30)  & (signal[:,3]>150) & (signal[:,4]>0.9)
    maskData = (realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1]) #& (realData[:,2]>30) & (realData[:,3]>150) & (realData[:,4]>0.9)
    signalCounts = np.histogram (np.clip(signal[maskSignal,1], bins[0], bins[-1]), bins=bins)[0]
    realDataCounts = np.histogram(np.clip(realData[maskData, 1], bins[0], bins[-1]), bins=bins)[0]
    
#realLumi = 0.67*np.sum(realDataCounts)/totalNanoEntries
    
    realDataCountsErr= np.sqrt(realDataCounts)*correctionData
    signalCountsErr= np.sqrt(signalCounts)/N_mini*30.52*0.67*1000

    signalCounts = signalCounts/N_mini*30.52*0.67*1000
    realDataCounts = realDataCounts*correctionData

    totalData += np.sum(realDataCounts)
    totalSignalCounts+=np.sum(signalCounts)

    ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='MC ggHbb')
    ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data n=4')

    ax.set_xlabel("Dijet Mass [GeV]", fontsize=16)
    ax.set_yscale('log')
    ax.set_xlim(x1, x2)
    ax.tick_params(which='major', length=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=4)
    if idx!=3:
        ax.text(s="%d GeV $\leq$ p$_\mathrm{T}$(jj) $<$ %d GeV"%(pTBins[idx], pTBins[idx+1]), x=1.00, y=1.02, ha='right', fontsize=12,  transform=ax.transAxes, **{'fontname':'Arial'})
    else:
        ax.text(s="%d GeV $\leq$ p$_\mathrm{T}$(jj) "%(pTBins[idx]), x=1.00, y=1.02, ha='right', fontsize=12,  transform=ax.transAxes, **{'fontname':'Arial'})
    if idx>0:
        pass
        #ax.tick_params(axis='y', which='both', left=False, right=False)

    for i in range(len(bins)-1):
        if i ==0 :
            rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///', label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///')
        ax.add_patch(rect)

    for i in range(len(bins)-1):
        if i ==0 :
            rect = patches.Rectangle((bins[i], signalCounts[i] - signalCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  signalCountsErr[i],
                    linewidth=0, edgecolor='blue', facecolor='none', hatch='///', label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], signalCounts[i] - signalCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  signalCountsErr[i],
                    linewidth=0, edgecolor='blue', facecolor='none', hatch='///')
        ax.add_patch(rect)
axes[0,0].set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)

#ax.text(s=r"%.5f fb$^{-1}$ (13 TeV)"%realLumi, x=1.00, y=1.02, ha='right', fontsize=12,  transform=ax.transAxes, **{'fontname':'Arial'})
print(totalSignalCounts)
print(totalData)
axes[0,0].legend(loc='upper right')


fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_differential.pdf", bbox_inches='tight')
