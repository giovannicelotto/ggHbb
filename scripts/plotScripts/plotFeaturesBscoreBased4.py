import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import getBins, loadRoot, getFeaturesBScoreBased, getXSectionBR, loadParquet, loadDask
import mplhep as hep
hep.style.use("CMS")
'''plot normalized features of signal and background'''





def plotNormalizedFeatures(signal, realData, outFile, toKeep=None):

    labels = getFeaturesBScoreBased(unit=True)
    xlims = getBins()
    nRow, nCol = 11, 4
    fig, ax = plt.subplots(nRow, nCol, figsize=(20, 40), constrained_layout=True)
    if toKeep is not None:
        labels, xlims, signal, realData = np.array(labels)[toKeep], xlims[toKeep], signal[:,toKeep], realData[:,toKeep]
        nRow, nCol = 4, 3
        fig, ax = plt.subplots(nRow, nCol, figsize=(14, 16), constrained_layout=True)
    # plot
    


    fig.align_ylabels(ax[:,0])
    
    # Jet pt
    for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            fig.align_ylabels(ax[:,j])
            if i*nCol+j>=len(signal.columns):
                break
            bins = np.linspace(xlims[i*nCol+j,1], xlims[i*nCol+j,2], int(xlims[i*nCol+j,0])+1)
            countsSignal = np.zeros(len(bins)-1)
            countsSignal = np.histogram(np.clip(signal.iloc[:,i*nCol+j], bins[0], bins[-1]),weights=signal.iloc[:,-1] if labels[i*nCol+j]!='SF' else None, bins=bins)[0]

            
            countsSignalErr = np.sqrt(countsSignal)
            countsBkg = np.zeros(len(bins)-1)
                
            countsBkg = np.histogram(np.clip(realData.iloc[:,i*nCol+j], bins[0], bins[-1]),weights=realData.iloc[:,-1] if labels[i*nCol+j]!='SF' else None, bins=bins)[0]
            
            countsBkgErr = np.sqrt(countsBkg)

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
    print("Saving in %s"%outFile)
    plt.close('all')
def main():
    # Loading files
    #signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatDataRoot"
    #realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"

    toKeep=None
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"

    signal, realData = loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=10)
    

    
    plotNormalizedFeatures(signal=signal, realData=realData, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_inclusive.png", toKeep=toKeep)
    #maskSig = signalYields[:,18]<22
    #maskBkg = realDataYields[:,18]<22
    #plotNormalizedFeatures(signalYields=signalYields[maskSig], realDataYields=realDataYields[maskBkg], outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt0to22.png", toKeep=toKeep)
    #maskSig = (signalYields[:,18]>22) & (signalYields[:,18]<40)
    #maskBkg = (realDataYields[:,18]>22) & (realDataYields[:,18]<40)
    #plotNormalizedFeatures(signalYields=signalYields[maskSig], realDataYields=realDataYields[maskBkg], outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt40to58.png", toKeep=toKeep)
    #maskSig = (signalYields[:,18]>40) & (signalYields[:,18]<69.5)
    #maskBkg = (realDataYields[:,18]>40) & (realDataYields[:,18]<69.5)
    #plotNormalizedFeatures(signalYields=signalYields[maskSig], realDataYields=realDataYields[maskBkg], outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt58to69p5.png", toKeep=toKeep)
    #maskSig = (signalYields[:,18]>69.5)# & (signalYields[:,18]<69.5)
    #maskBkg = (realDataYields[:,18]>69.5)# & (signalYields[:,18]<69.5)
    #plotNormalizedFeatures(signalYields=signalYields[maskSig], realDataYields=realDataYields[maskBkg], outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt69p5toInf.png", toKeep=toKeep)
if __name__=="__main__":
    main()
