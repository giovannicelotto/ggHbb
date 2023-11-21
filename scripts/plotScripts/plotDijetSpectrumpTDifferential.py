import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from utilsForPlot import loadData, getXSectionBR, loadDataOnlyFeatures
import mplhep as hep
hep.style.use("CMS")

afterCut = False
if afterCut:
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/afterCutWithMoreFeatures/signalCut.npy"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/afterCutWithMoreFeatures/realDataCut.npy"
    signal = np.load(signalPath)    
    realData = np.load(realDataPath)
else:

    # Loading files
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"
    signal, realData = loadDataOnlyFeatures(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=-1, features=[0,19])

# Correction factors and counters
N_SignalMini = np.load("/t3home/gcelotto/bbar_analysis/output/N_mini.npy")
N_DataNano = 190153970   
correctionSignal = 1/N_SignalMini*getXSectionBR()*0.67*1000
correctionData = N_DataNano/len(realData)*0.88
if afterCut:
    print("Watchout")
    correctionData = N_DataNano/len(realData)*0.88*0.0009
totalSignalCounts = 0                               # total counts for the signal after rescaled
totalData = 0                   
print(realData.shape)
# Plot
fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharey=False)
fig.subplots_adjust(wspace=0.25)
fig.subplots_adjust(hspace=0.25)


pTBins = [0, 20, 100, 230, 9999999]

#bins for showing pT distribution
pTDiffBins = np.linspace(0, 500, 101)

for idx, ax in enumerate(axes[1,:]):
    pTSignalCounts = np.histogram (np.clip(signal[:,0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionSignal
    pTrealDataCounts = np.histogram (np.clip(realData[:,0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, histtype=u'step', color='blue')
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, histtype=u'step', color='red')
    pTSignalCounts = np.histogram (np.clip(signal[(signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1]),0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionSignal
    pTrealDataCounts = np.histogram (np.clip(realData[(realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1]),0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, alpha=0.4, color='blue')
    ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, alpha=0.4, color='red')

    #ax.set_xlabel("Dijet Pt [GeV]", fontsize=16)
    ax.set_yscale('log')
    ax.set_xlim(0, 500)
axes[1,0].set_ylabel("Events / %.1f GeV"%(pTDiffBins[1]-pTDiffBins[0]))
for idx, ax in enumerate(axes[0,:]):
    # invariant mass bins and # Classes of pT
    x1, x2, nbin = 0, [300, 600, 800, 1000][idx], 51
    bins = np.linspace(x1, x2, nbin)

    maskSignal = (signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1]) #& (signal[:,2]>30)  & (signal[:,3]>150) & (signal[:,4]>0.9)
    maskData = (realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1]) #& (realData[:,2]>30) & (realData[:,3]>150) & (realData[:,4]>0.9)
    signalCounts = np.histogram (np.clip(signal[maskSignal, 1], bins[0], bins[-1]), bins=bins)[0]
    realDataCounts = np.histogram(np.clip(realData[maskData, 1], bins[0], bins[-1]), bins=bins)[0]
#realLumi = 0.67*np.sum(realDataCounts)/N_DataNano
    
    realDataCountsErr= np.sqrt(realDataCounts)*correctionData
    signalCountsErr= np.sqrt(signalCounts)*correctionSignal

    signalCounts = signalCounts*correctionSignal
    realDataCounts = realDataCounts*correctionData

    totalData += np.sum(realDataCounts)
    totalSignalCounts+=np.sum(signalCounts)

    ax.hist(bins[:-1], bins=bins, weights=signalCounts*10**4, color='blue', histtype=u'step', label=r'MC ggHbb $\times10^4$')
    ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')

    #How many events between 95 and 165 GeV
    maskSignal = (maskSignal) & (signal[:,1]>95) & (signal[:,1]<165)
    maskData = (maskData) & (realData[:,1]>95) & (realData[:,1]<165)
    
    S = np.sum(maskSignal)*correctionSignal
    print(idx, S)
    B = np.sum(maskData)*correctionData
    SErr = np.sqrt(np.sum(maskSignal))*correctionSignal
    BErr = np.sqrt(np.sum(maskData))*correctionData
    
    Scommon_exponent = int("{:e}".format(SErr).split('e')[1])
    Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])
    Scommon_exponent = 0 if Scommon_exponent<0 else Scommon_exponent
    ax.text(s="0.67 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
    if Scommon_exponent==0:
        ax.text(s=r"S = (%d $\pm$ %d)"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent)),                                     x=0.93, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(s=r"S = (%d $\pm$ %d) $\times e%d$"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent), Scommon_exponent),      x=0.93, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
    ax.text(s=r"B = (%d $\pm$ %d) $\times 10^{%d}$"%(B/(10**Bcommon_exponent), BErr/(10**Bcommon_exponent), Bcommon_exponent),      x=0.93, y=0.55,  ha='right', transform=ax.transAxes, fontsize=12)
    ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S/(np.sqrt(B))),                                                                      x=0.93, y=0.5,  ha='right', transform=ax.transAxes, fontsize=12)
    ax.text(s="S/B = %.1e"%(S/B),                                                                                                   x=0.93, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
# @ Full lumi
    ax.text(s="@Full BPH Lumi (42 fb$^{-1}$)",                                                                                      x=0.93, y=0.35,  ha='right', transform=ax.transAxes, fontsize=12)
    ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S*np.sqrt(42/0.67)/(np.sqrt(B))),                                                     x=0.93, y=0.3,  ha='right', transform=ax.transAxes, fontsize=12)
    

    ax.set_xlabel("Dijet Mass [GeV]", fontsize=14)
    #ax.set_yscale('log')
    ax.set_xlim(x1, x2)
    if idx!=3:
        ax.text(s="%d GeV $\leq$ p$_\mathrm{T}$(jj) $<$ %d GeV"%(pTBins[idx], pTBins[idx+1]), x=0.93, y=0.65, ha='right', fontsize=12,  transform=ax.transAxes)
    else:
        ax.text(s="%d GeV $\leq$ p$_\mathrm{T}$(jj) "%(pTBins[idx]), x=0.93, y=0.65, ha='right', fontsize=12,  transform=ax.transAxes)
    if idx>0:
        pass
        #ax.tick_params(axis='y', which='both', left=False, right=False)

    for i in range(len(bins)-1):
        if i ==0 :
            rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///') #label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///')
        ax.add_patch(rect)

    for i in range(len(bins)-1):
        if i ==0 :
            rect = patches.Rectangle((bins[i], signalCounts[i] - signalCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  signalCountsErr[i],
                    linewidth=0, edgecolor='blue', facecolor='none', hatch='///') #label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], signalCounts[i] - signalCountsErr[i]),
                    bins[i+1]-bins[i], 2 *  signalCountsErr[i],
                    linewidth=0, edgecolor='blue', facecolor='none', hatch='///')
        ax.add_patch(rect)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(loc='upper right', fontsize=16)
    ax.vlines(x=[95, 165], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='blue', alpha=0.2)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
axes[0,0].set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)

#ax.text(s=r"%.5f fb$^{-1}$ (13 TeV)"%realLumi, x=1.00, y=1.02, ha='right', fontsize=12,  transform=ax.transAxes, **{'fontname':'Arial'})
print(totalSignalCounts)
print(totalData)

outName = "/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_differentialCut.pdf" if afterCut else "/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_differential.pdf"
fig.savefig(outName, bbox_inches='tight')
