import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import getBins, loadRoot, getFeaturesBScoreBased, loadParquet, loadDask
from functions import getXSectionBR, loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")


def plotNormalizedFeatures(data, outFile, legendLabels, colors, histtypes=None, alphas=None, figsize=None, autobins=False, weights=None):
    '''
    plot normalized features of signal and background
    data= list of dataframes one for each process'''
    
    xlims = getBins(dictFormat=True)
    nRow, nCol = int(len(data[0].columns)/4+int(bool(len(data[0].columns)%4))), 4
    fig, ax = plt.subplots(nRow, nCol, figsize=(20, 15) if figsize==None else figsize, constrained_layout=True)
    
    fig.align_ylabels(ax[:,0])
    
    for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            if i*nCol+j>=len(data[0].columns):
                break
            featureName = data[0].columns[i*nCol+j]

            
            fig.align_ylabels(ax[:,j])
            if featureName not in xlims.columns:
                
                
                bins = np.linspace(data[0][featureName].min(), data[0][featureName].max(), 20)
            else:
                
                bins = np.linspace(xlims[featureName][1], xlims[featureName][2], int(xlims[featureName][0])+1)
            if autobins:
                try:
                    xmin, xmax = data[0][featureName].quantile(0.1), data[0][featureName].quantile(0.9)
                    for idx in range(len(data)):
                        if data[idx][featureName].quantile(0.1) < xmin:
                            xmin =data[idx][featureName].quantile(0.1)
                        if data[idx][featureName].quantile(0.9) > xmax:
                            xmax = data[idx][featureName].quantile(0.9)
                    bins = np.linspace(xmin, xmax, 20)
                    if featureName=='sf':
                        bins=np.linspace(0, 1, 20)
                except:
                    bins = np.linspace(data[1][featureName].min(), data[1][featureName].max(), 20)
            
            dataIdx = 0
            for idx, df in enumerate(data):
                if weights is None:
                    weightsDf=df.sf
                else:
                    weightsDf = weights[idx]
                
                counts = np.zeros(len(bins)-1)
                counts = np.histogram(np.clip(df[featureName], bins[0], bins[-1]),weights = weightsDf if featureName!='sf' else None, bins=bins)[0]
                
                
                
                assert ((counts>=0).all())
                    
                countsErr = np.sqrt(counts)
                
                # Normalize the counts to 1 so also the errors undergo the same operation. Do first the errors, otherwise you lose the info on the signal
                countsErr = countsErr/np.sum(counts)
                counts = counts/np.sum(counts)
                

                ax[i, j].hist(bins[:-1], bins=bins, weights=counts, label=legendLabels[dataIdx], histtype=u'step' if histtypes==None else histtypes[dataIdx], 
                              alpha=1 if alphas==None else alphas[dataIdx], color=colors[dataIdx], )[:2]
                
                ax[i, j].set_xlabel(featureName, fontsize=18)
                ax[i, j].set_xlim(bins[0], bins[-1])
                ax[i, j].set_ylabel("Probability", fontsize=18)

                # Some subplots in log scale
                if any(substring in df.columns[i * nCol + j] for substring in ['nMuons', 'nElectrons', ]):
                    ax[i, j].set_yscale('log')
                if featureName == 'sf':
                    ax[i, j].legend(fontsize=18)


                ax[i, j].tick_params(which='major', length=8)
                ax[i, j].xaxis.set_minor_locator(AutoMinorLocator())
                ax[i, j].tick_params(which='minor', length=4)


                for idx in range(len(bins)-1):
                    #if idx ==0 :
                    #    rect = patches.Rectangle((bins[idx], counts[idx] - countsErr[idx]),
                    #            bins[idx+1]-bins[idx], 2 *  countsErr[idx],
                    #            linewidth=0, edgecolor=colors[dataIdx], facecolor='none', hatch='///', label='Uncertainty')
                    #else:
                    rect = patches.Rectangle((bins[idx], counts[idx] - countsErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsErr[idx],
                            linewidth=0, edgecolor=colors[dataIdx], facecolor='none', hatch='///')
                    ax[i, j].add_patch(rect)
                dataIdx = dataIdx + 1

    
    fig.savefig(outFile, bbox_inches='tight')
    print("Saving in %s"%outFile)
    plt.close('all')
def main():
    # Loading files
    #signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatDataRoot"
    #realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"

    
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"

    #signal, realData = loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=30)
    paths = [
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/old/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/old/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/old/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/old/ZJets/ZJetsToQQ_HT-800toInf",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets"
            ]
    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=100, nMC=-1, returnNumEventsTotal=True, columns=None)


    
    #signal['dijet_RPhi_1'] = signal['dijet_phi'] - signal['jet1_phi'] 
    #signal['dijet_RPhi_2'] = signal['dijet_phi'] - signal['jet2_phi'] 
    #realData['dijet_RPhi_1'] = realData['dijet_phi'] - realData['jet1_phi'] 
    #realData['dijet_RPhi_2'] = realData['dijet_phi'] - realData['jet2_phi'] 

    
    plotNormalizedFeatures(data=dfs,
                           outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_ZJets_HTclasses.png",
                           legendLabels = ['ZJets200-400', 'ZJets400-600',
                                           'ZJets600-800', 'ZJets800-Inf',],
                                           #'EWKZJets'] ,
                           colors = ['blue', 'red', 'green', 'black'],
                           figsize=(15, 30))
if __name__=="__main__":
    main()
