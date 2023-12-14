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

def plotDijetpTDifferential(realFiles=1, afterCut = False):
    
    if afterCut:
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/signalCut.npy"#_fullLargeMemory
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/realDataCut.npy"#_fullLargeMemory
        signal = np.load(signalPath)    
        realData = np.load(realDataPath)
        signal = signal[:,[18, 21, -1]]
        realData = realData[:,[18, 21, -1]]
    else:

        # Loading files
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatData"
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatData"
        signal, realData = loadDataOnlyFeatures(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles, features=[18, 21, -1])


    # Correction factors and counters
    N_SignalMini = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_mini.npy")*237/240
    currentLumi = 0.774*realFiles/1017
    correctionSignal = 1/N_SignalMini*getXSectionBR()*1000*currentLumi
    correctionData = 1
    print("Current Lumi : ", currentLumi)
    print("Amount of signal : ", np.sum(signal[:,-1])*correctionSignal)
    print("signal shape", signal.shape)

    # Divide in 4 classes of pT of equal lenght:
    signalSF =signal[:,2]
    signal = signal[:,[0, 1]]
    realData = realData[:,[0, 1]]

    # signal = signal[signal[:, 0].argsort()]
    # q1 = signal[int(len(signal)/4),0]
    # q2 = signal[int(len(signal)/2),0]
    # q3 = signal[int(len(signal)*3/4),0]

    # Calculate quartiles based on the sorted dataset
    print("Number of entries : %d"%len(signal))
    print(signal.shape)
    q1, q2, q3 = 50, 100, 200


    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharey=False)
    fig.subplots_adjust(wspace=0.25)
    fig.subplots_adjust(hspace=0.25)

    pTBins = [0, q1, q2, q3, 9999999]

    #bins for showing pT distribution
    pTDiffBins = np.linspace(0, 500, 51)

    for idx, ax in enumerate(axes[1,:]):
        pTSignalCounts = np.histogram (np.clip(signal[:,0], pTDiffBins[0], pTDiffBins[-1]), weights=signalSF, bins=pTDiffBins)[0]*correctionSignal
        pTrealDataCounts = np.histogram (np.clip(realData[:,0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, histtype=u'step', color='blue')
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, histtype=u'step', color='red')
        # to color inside, draw another histo
        maskSigPt = (signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1])
        pTSignalCounts = np.histogram (np.clip(signal[maskSigPt, 0], pTDiffBins[0], pTDiffBins[-1]), weights=signalSF[maskSigPt],bins=pTDiffBins)[0]*correctionSignal
        pTrealDataCounts = np.histogram (np.clip(realData[(realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1]),0], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, alpha=0.4, color='blue')
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, alpha=0.4, color='red')

        
        ax.set_xlabel("Dijet Pt [GeV]", fontsize=18)
        ax.set_yscale('log')
        ax.set_xlim(0, 500)
    axes[1,0].set_ylabel("Events / %.1f GeV"%(pTDiffBins[1]-pTDiffBins[0]), fontsize=18)

    # First row of the plot draw invariant masses
    for idx, ax in enumerate(axes[0,:]):

        x1, x2, nbin = 0, [300, 300, 300, 500][idx], 101
        bins = np.linspace(x1, x2, nbin)

        maskSignal = (signal[:,0]>pTBins[idx]) & (signal[:,0]<pTBins[idx+1]) 
        maskData = (realData[:,0]>pTBins[idx]) & (realData[:,0]<pTBins[idx+1])
        signalCounts = np.histogram(np.clip(signal[maskSignal, 1], bins[0], bins[-1]), weights=signalSF[maskSignal],bins=bins)[0]
        realDataCounts = np.histogram(np.clip(realData[maskData, 1], bins[0], bins[-1])                             , bins=bins)[0]
        print(idx, np.sum(maskSignal))
    #realLumi = 0.774*np.sum(realDataCounts)/N_DataNano

        realDataCountsErr= np.sqrt(realDataCounts)*correctionData
        signalCountsErr= np.sqrt(signalCounts)*correctionSignal

        signalCounts = signalCounts*correctionSignal
        realDataCounts = realDataCounts*correctionData

        ax.hist(bins[:-1], bins=bins, weights=signalCounts*10**3, color='blue', histtype=u'step', label=r'Signal $\times10^3$')
        ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')

        #How many events between line1 and line2 GeV
        line1, line2 = 123.09-17.03*2, 123.09+17.03*2
        maskSignal = (maskSignal) & (signal[:,1]>line1) & (signal[:,1]<line2)
        maskData = (maskData) & (realData[:,1]>line1) & (realData[:,1]<line2)

        S = np.sum(signalSF[maskSignal])*correctionSignal
        print(idx, S)
        B = np.sum(maskData)*correctionData
        SErr = np.sqrt(np.sum(signalSF[maskSignal]))*correctionSignal
        BErr = np.sqrt(np.sum(maskData))*correctionData

        #Scommon_exponent = int(abs(np.log10(SErr))) + 1
        #print("Commong", Scommon_exponent)
        Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])
        #Scommon_exponent = 0 if Scommon_exponent<0 else Scommon_exponent
        ax.text(s="%.3f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
        #if Scommon_exponent==0:
        #    Scommon_exponent=-1
            #ax.text(s=r"S = (%d $\pm$ %d)"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent)),                                     x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
        #else:
        ax.text(s=r"S = %.1f $\pm$ %.1f "%(round(S, 1), round(SErr, 1)),      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s=r"B = (%d $\pm$ %d) $\times 10^{%d}$"%(B/(10**Bcommon_exponent), BErr/(10**Bcommon_exponent), Bcommon_exponent),      x=0.96, y=0.55,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S/(np.sqrt(B))),                                                                      x=0.96, y=0.5,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/B = %.1e"%(S/B),                                                                                                   x=0.96, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
    # @ Full lumi
        ax.text(s="@Full BP Lumi (41.6 fb$^{-1}$)",                                                                                      x=0.96, y=0.35,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S*np.sqrt(41.6/currentLumi)/(np.sqrt(B))),                                                     x=0.96, y=0.3,  ha='right', transform=ax.transAxes, fontsize=12)


        ax.set_xlabel("Dijet Mass [GeV]", fontsize=18)
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
        ax.vlines(x=[line1, line2], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='blue', alpha=0.2)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    axes[0,0].set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=18)



    outName = "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/dijetMass_differentialCut.png" if afterCut else "/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_differential.png"
    print("Saving %s"%outName)
    fig.savefig(outName, bbox_inches='tight')


if __name__=="__main__":
    if len(sys.argv)>1:
        realFiles=int(sys.argv[1])
    if len(sys.argv)>2:
        afterCut = bool(int(sys.argv[2])) if len(sys.argv)>2 else False
        plotDijetpTDifferential(realFiles=realFiles, afterCut=afterCut)
    else:
        plotDijetpTDifferential()