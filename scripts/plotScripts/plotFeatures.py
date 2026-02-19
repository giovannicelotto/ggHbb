import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import getBins
import mplhep as hep
hep.style.use("CMS")
import sys


def plotNormalizedFeatures(data, outFile, legendLabels, colors, histtypes=None, alphas=None, figsize=None, autobins=False, weights=None, error=True):
    '''
    plot normalized features of signal and background
    data= list of dataframes one for each process'''
    # Find common columns
    common_columns = set(data[0].columns) 
    for df in data[1:]:
        common_columns.intersection_update(df.columns)  # Update with the intersection of columns
    ordered_common_columns = [col for col in data[0].columns if col in common_columns]

    # Retain only the common columns in their original order for each DataFrame
    data = [df[ordered_common_columns] for df in data]


    xlims = getBins(dictFormat=True)
    nRow, nCol = int(len(data[0].columns)/6+int(bool(len(data[0].columns)%6))), 6
    fig, ax = plt.subplots(nRow, nCol, figsize=(20, 15) if figsize==None else figsize, constrained_layout=True)
    fig.align_ylabels(ax[:,0])
    
    for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            if i*nCol+j>=len(data[0].columns):
                break
            featureName = data[0].columns[i*nCol+j]

            print("="*30)
            print(featureName)
            fig.align_ylabels(ax[:,j])
            if featureName not in xlims.columns:
                bins = np.linspace(data[0][featureName].min(), data[0][featureName].max(), 20)
                print("Feature %s not found. Binning Automatically defined"%featureName)
            else:
                bins = np.linspace(xlims[featureName][1], xlims[featureName][2], int(xlims[featureName][0])+1)
            if autobins:
                try:
                    xmin, xmax = data[0][featureName].quantile(0.005), data[0][featureName].quantile(0.995)
                    for idx in range(len(data)):
                        if data[idx][featureName].quantile(0.005) < xmin:
                            xmin =data[idx][featureName].quantile(0.005)
                        if data[idx][featureName].quantile(0.995) > xmax:
                            xmax = data[idx][featureName].quantile(0.995)
                    bins = np.linspace(xmin, xmax, 20)
                except:
                    bins = np.linspace(data[1][featureName].min(), data[1][featureName].max(), 20)
            dataIdx = 0
            for idx, df in enumerate(data):
                
                if weights is None:
                    weightsDf=df.flat_weight if 'flat_weight' in df.columns else None
                else:
                    weightsDf = weights[idx]
                counts = np.zeros(len(bins)-1)
                
                feature_data = df[featureName].astype(int) if df[featureName].dtype == bool else df[featureName]
                counts = np.histogram(np.clip(feature_data, bins[0], bins[-1]),weights = weightsDf if featureName!='sf' else None, bins=bins)[0]
                              
                
                if ((counts<0).any()):
                    print("Negative counts in ", featureName)
                    #print(counts)
                if error:
                    countsErr = np.sqrt(np.histogram(np.clip(feature_data, bins[0], bins[-1]),weights = weightsDf**2 if featureName!='sf' else None, bins=bins)[0])

                
                # Normalize the counts to 1 so also the errors undergo the same operation. Do first the errors, otherwise you lose the info on the signal
                    countsErr = countsErr/np.sum(counts)
                counts = counts/np.sum(counts)
                

                ax[i, j].hist(bins[:-1], bins=bins, weights=counts, label=legendLabels[dataIdx], histtype=u'step' if histtypes==None else histtypes[dataIdx], 
                            alpha=1 if alphas==None else alphas[dataIdx], color=colors[dataIdx], )[:2]
                
                ax[i, j].set_xlabel(featureName, fontsize=18)
                ax[i, j].set_xlim(bins[0], bins[-1])
                ax[i, j].set_ylabel("Probability", fontsize=18)

                # Some subplots in log scale
                if any(substring in df.columns[i * nCol + j] for substring in ['nMuons', 'nElectrons','nTightMuons','dimuon_mass' ]):
                    ax[i, j].set_yscale('log')
                if featureName == 'flat_weight':
                    ax[i, j].legend(fontsize=18)


                ax[i, j].tick_params(which='major', length=8)
                ax[i, j].xaxis.set_minor_locator(AutoMinorLocator())
                ax[i, j].tick_params(which='minor', length=4)

                if error:
                    for idx in range(len(bins)-1):
                        rect = patches.Rectangle((bins[idx], counts[idx] - countsErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsErr[idx],
                            linewidth=0, edgecolor=colors[dataIdx], facecolor='none', hatch='///')
                        ax[i, j].add_patch(rect)
                dataIdx = dataIdx + 1

    if outFile is None:
        print("[INFO] No outFile provided")
        plt.show()
    else:
        fig.savefig(outFile, bbox_inches='tight')
        print("Saving in %s"%outFile)
    plt.close('all')
