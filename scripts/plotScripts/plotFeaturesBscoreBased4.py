import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import getBins, loadData, getFeaturesBScoreBased, getXSectionBR
import mplhep as hep
hep.style.use("CMS")
'''plot normalized features of signal and background'''





def plotNormalizedFeatures(signal, realData, outFile, toKeep=None):

    # Correction factors and counters
    N_mini = np.load("/t3home/gcelotto/ggHbb/outputs/N_mini.npy")
    totalNanoEntries = np.load("/t3home/gcelotto/ggHbb/outputs/N_BPH_Nano.npy")
    correctionData = totalNanoEntries/len(realData)
    totalSignalCounts, totalData = 0, 0                               # total counts for the signal after rescaled

    labels = getFeaturesBScoreBased(unit=True)
    xlims = getBins()
    nRow, nCol = 7, 4
    if toKeep is not None:
        labels, xlims, signal, realData = np.array(labels)[toKeep], xlims[toKeep], signal[:,toKeep], realData[:,toKeep]
        nRow, nCol = 4, 3
    # plot
    fig, ax = plt.subplots(nRow, nCol, figsize=(15, 17), constrained_layout=True)
    


    fig.align_ylabels(ax[:,0])
    
    # Jet pt
    for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            fig.align_ylabels(ax[:,j])
            if i*nCol+j>=signal.shape[1]:
                break
            bins = np.linspace(xlims[i*nCol+j,0], xlims[i*nCol+j,1], 30)

            countsSignal = np.histogram(np.clip(signal[:,i*nCol+j], bins[0], bins[-1]), bins=bins)[0]
            countsSignalErr = np.sqrt(countsSignal)
            countsSignal, countsSignalErr = countsSignal/N_mini*getXSectionBR()*0.67*1000, countsSignalErr/N_mini*getXSectionBR()*0.67*1000
            countsBkg    = np.histogram(np.clip(realData[:,i*nCol+j], bins[0], bins[-1]), bins=bins)[0]
            countsBkgErr = np.sqrt(countsBkg)
            countsBkg, countsBkgErr = countsBkg*correctionData, countsBkgErr*correctionData

            # Normalize the counts to 1 so also the errors undergo the same operation. Do first the errors, otherwise you lose the info on the signal
            countsSignalErr = countsSignalErr/np.sum(countsSignal)
            countsSignal = countsSignal/np.sum(countsSignal)
            countsBkgErr=countsBkgErr/np.sum(countsBkg)
            countsBkg=countsBkg/np.sum(countsBkg)

            ax[i, j].hist(bins[:-1], bins=bins, weights=countsSignal, label='Signal', histtype=u'step',  color='blue', )[:2]
            ax[i, j].hist(bins[:-1], bins=bins, weights=countsBkg, label='Background', histtype=u'step' , color='red',)

            ax[i, j].set_xlabel(labels[i*nCol+j], fontsize=18)
            ax[i, j].set_xlim(bins[0], bins[-1])
            ax[i, j].set_ylabel("Probability", fontsize=18)

            # Some subplots in log scale
            if any(substring in labels[i * nCol + j] for substring in ['nMuons', 'nElectrons', ]):
                ax[i, j].set_yscale('log')

            ax[i, j].legend(fontsize=18)


            ax[i, j].tick_params(which='major', length=8)
            ax[i, j].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i, j].tick_params(which='minor', length=4)
            # Plot the errors on the normalization
            for idx in range(len(bins)-1):
                if idx ==0 :
                    rect = patches.Rectangle((bins[idx], countsBkg[idx] - countsBkgErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsBkgErr[idx],
                            linewidth=0, edgecolor='red', facecolor='none', hatch='///', label='Uncertainty')
                else:
                    rect = patches.Rectangle((bins[idx], countsBkg[idx] - countsBkgErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsBkgErr[idx],
                            linewidth=0, edgecolor='red', facecolor='none', hatch='///')
                ax[i, j].add_patch(rect)

            for idx in range(len(bins)-1):
                if idx ==0 :
                    rect = patches.Rectangle((bins[idx], countsSignal[idx] - countsSignalErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsSignalErr[idx],
                            linewidth=0, edgecolor='blue', facecolor='none', hatch='///', label='Uncertainty')
                else:
                    rect = patches.Rectangle((bins[idx], countsSignal[idx] - countsSignalErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsSignalErr[idx],
                            linewidth=0, edgecolor='blue', facecolor='none', hatch='///')
                ax[i, j].add_patch(rect)

    
    fig.savefig(outFile, bbox_inches='tight')
    plt.close('all')
def main():
    # Loading files
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

    toKeep = [ 0, 3, 6,
                9, 10, 15,
                23, 24, 26,
                8, 17, 27]
    signal, realData = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=100, nRealDataFiles=20)
    
    plotNormalizedFeatures(signal=signal, realData=realData, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/normalizedFeaturesNew.png", toKeep=toKeep)

if __name__=="__main__":
    main()





