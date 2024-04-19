import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from utilsForPlot import loadData, loadDataOnlyFeatures, loadParquet
from functions import getXSectionBR
sys.path.append("/t3home/gcelotto/ggHbb/NN/")
from helpersForNN import preprocessMultiClass
import mplhep as hep
hep.style.use("CMS")

def plotDijetpTDifferential(realFiles=1, afterCut = False):
    
    #if afterCut:
    #    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/signalCut.parquet"#_fullLargeMemory
    #    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/realDataCut.parquet"#_fullLargeMemory
    #    # load dask dataframe and convert to pandas dataframes
    #    signal = dd.read_parquet(signalPath, names=['dijet_pt', 'dijet_mass', 'sf']).compute()
    #    realData = dd.read_parquet(realDataPath, names=['dijet_pt', 'dijet_mass', 'sf']).compute()
        
    #else:
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**"

        # Loading Pandas dataframe
    signal, realData, numEventsTotal = loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles,
                                                   columns=['dijet_pt', 'dijet_mass', 'sf', 'jet1_pt', 'jet2_pt', 'jet1_eta', 'jet2_eta', 'jet1_mass', 'jet2_mass',
                                                            'jet1_qgl', 'jet2_qgl'], returnNumEventsTotal=True)
    if afterCut:
        print("Cut")
        #signal, realData = signal[(signal.jet1_pt>20) & (signal.jet2_pt>20)], realData[(realData.jet1_pt>20) & (realData.jet2_pt>20)]
        dfs = [signal, realData]
        signal, realData = preprocessMultiClass(dfs)
        
    
    # Correction factors and counters
    #realFiles = 1017 if realFiles==-1 else realFiles
    #N_SignalMini = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_mini.npy")*237/240
    currentLumi = 0.774*realFiles/1017
    correctionSignal = 1/numEventsTotal*getXSectionBR()*1000*currentLumi
    correctionData = 1
    print("Current Lumi : ", currentLumi)
    print("Amount of signal : ", signal.sf.sum()*correctionSignal)


    # Calculate quartiles based on the sorted dataset
    print("Number of entries : %d"%len(signal))
    print(signal.shape)
    q1, q2 = 30, 100

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=False)
    fig.subplots_adjust(wspace=0.25)
    fig.subplots_adjust(hspace=0.25)

    pTBins = [0, q1, q2, 9999999]

    #bins for showing pT distribution
    pTDiffBins = np.linspace(0, 500, 51)

    for idx, ax in enumerate(axes[1,:]):
    # draw second raw of dijet pt
        pTSignalCounts = np.histogram(np.clip(signal.dijet_pt, pTDiffBins[0], pTDiffBins[-1]), weights=signal.sf, bins=pTDiffBins)[0]*correctionSignal
        pTrealDataCounts = np.histogram(np.clip(realData.dijet_pt, pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, histtype=u'step', color='blue')
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, histtype=u'step', color='red')
    # to color inside, draw another histo
        maskSigPt = (signal.dijet_pt>pTBins[idx]) & (signal.dijet_pt<pTBins[idx+1])
        maskDataPt = (realData.dijet_pt>pTBins[idx]) & (realData.dijet_pt<pTBins[idx+1])
        pTSignalCounts = np.histogram   (np.clip(signal.dijet_pt[maskSigPt], pTDiffBins[0], pTDiffBins[-1]), weights=signal.sf[maskSigPt],bins=pTDiffBins)[0]*correctionSignal
        pTrealDataCounts = np.histogram (np.clip(realData.dijet_pt[maskDataPt], pTDiffBins[0], pTDiffBins[-1]), bins=pTDiffBins)[0]*correctionData
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTSignalCounts, alpha=0.4, color='blue')
        ax.hist(pTDiffBins[:-1], bins=pTDiffBins, weights=pTrealDataCounts, alpha=0.4, color='red')

        
        ax.set_xlabel("Dijet Pt [GeV]", fontsize=18)
        ax.set_yscale('log')
        ax.set_xlim(0, 500)
    axes[1,0].set_ylabel("Events / %.1f GeV"%(pTDiffBins[1]-pTDiffBins[0]), fontsize=18)

    # First row of the plot draw invariant masses
    for idx, ax in enumerate(axes[0,:]):

        x1, x2, nbin = 70, [250, 250, 250][idx], 101
        bins = np.linspace(x1, x2, nbin)

        maskSignal = (signal.dijet_pt>pTBins[idx]) & (signal.dijet_pt<pTBins[idx+1]) 
        maskData = (realData.dijet_pt>pTBins[idx]) & (realData.dijet_pt<pTBins[idx+1])
        #signalCounts    = np.histogram(np.clip(signal.dijet_mass[maskSignal], bins[0], bins[-1]), weights=signal.sf[maskSignal],bins=bins)[0]
        #realDataCounts  = np.histogram(np.clip(realData.dijet_mass[maskData], bins[0], bins[-1])                             , bins=bins)[0]
        signalCounts    = np.histogram(signal.dijet_mass[maskSignal], weights=signal.sf[maskSignal],bins=bins)[0]
        realDataCounts  = np.histogram(realData.dijet_mass[maskData]                             , bins=bins)[0]
        print(idx, np.sum(maskSignal))
    #realLumi = 0.774*np.sum(realDataCounts)/N_DataNano

        realDataCountsErr= np.sqrt(realDataCounts)*correctionData
        signalCountsErr= np.sqrt(signalCounts)*correctionSignal

        signalCounts = signalCounts*correctionSignal
        realDataCounts = realDataCounts*correctionData

        ax.hist(bins[:-1], bins=bins, weights=signalCounts*10**3, color='blue', histtype=u'step', label=r'Signal $\times10^3$')
        ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')

        #How many events between line1 and line2 GeV
        line1, line2 = 125.09-17.03*2, 125.09+17.03*2
        maskSignal = (maskSignal) & (signal.dijet_mass>line1) & (signal.dijet_mass<line2)
        maskData = (maskData) & (realData.dijet_mass>line1) & (realData.dijet_mass<line2)

        S = np.sum(signal.sf[maskSignal])*correctionSignal
        print(idx, S)
        B = np.sum(maskData)*correctionData
        SErr = np.sqrt(np.sum(signal.sf[maskSignal]))*correctionSignal
        BErr = np.sqrt(np.sum(maskData))*correctionData

        #Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])
        ax.text(s="%.3f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
        
        ax.text(s=r"S = %.1f $\pm$ %.1f "%(round(S, 1), round(SErr, 1)),      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s=r"B = %.1f $\pm$ %.1f"%(round(B, 1), round(BErr, 1)),      x=0.96, y=0.55,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.2f"%(S/(np.sqrt(B))),                                                                      x=0.96, y=0.5,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/B = %.2f"%(S/B),                                                                                                   x=0.96, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
    # @ Full lumi
        ax.text(s="@Full BP Lumi (41.6 fb$^{-1}$)",                                                                                      x=0.96, y=0.35,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.2f"%(S*np.sqrt(41.6/currentLumi)/(np.sqrt(B))),                                                     x=0.96, y=0.3,  ha='right', transform=ax.transAxes, fontsize=12)


        ax.set_xlabel("Dijet Mass [GeV]", fontsize=18)
        #ax.set_yscale('log')
        ax.set_xlim(x1, x2)
        if idx!=(axes.shape[1]-1):
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



    outName = "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/dijetMass_differentialCut.png" if afterCut else "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/dijetMass_differential.png"
    print("Saving %s"%outName)
    fig.savefig(outName, bbox_inches='tight')


if __name__=="__main__":
    if len(sys.argv)>1:
        realFiles=int(sys.argv[1])
        afterCut = bool(int(sys.argv[2])) if len(sys.argv)>2 else False
        plotDijetpTDifferential(realFiles=realFiles, afterCut=afterCut)
    else:
        plotDijetpTDifferential()