from utilsForPlot import loadParquet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"
signal, realData = loadParquet(signalPath, realDataPath, nRealDataFiles=1000, columns=['dijet_mass', 'dijet_pt', 'sf', 'jet1_pt', 'jet2_pt',
                                                                                       'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'ht'])
nRow, nCol = 2, 3
thresholds = [40, 50, 60,
              70, 80, 90]
fig, ax = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(15, 10))
bins = np.linspace(20, 300, 60)
for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            if i*nCol+j>=6:
                  break
            mS = (signal.jet1_pt>20) #& (signal.jet2_pt>45) & (signal.jet1_btagDeepFlavB > 0.95) & (signal.jet2_btagDeepFlavB > 0.95) & (signal.dijet_pt> thresholds[i*nCol+j]) & (signal.ht> 700) & (signal.dijet_mass< 300)                            #np.ones(len(signal))#
            mD = (realData.jet1_pt>20) #& (realData.jet2_pt>45) & (realData.jet1_btagDeepFlavB > 0.95) & (realData.jet2_btagDeepFlavB > 0.95) & (realData.dijet_pt> thresholds[i*nCol+j]) & (realData.ht> 700) & (realData.dijet_mass< 300)                          #np.ones(len(realData))#
            
            cS = np.histogram(np.clip(signal.dijet_mass[mS], bins[0], bins[-1]), bins=bins, weights=signal.sf[mS])[0]
            cD = np.histogram(np.clip(realData.dijet_mass[mD], bins[0], bins[-1]), bins=bins)[0]

            cS = cS/np.sum(cS)
            cD = cD/np.sum(cD)
            ax[i, j].hist(bins[:-1], bins=bins, weights=cS, color='blue', histtype=u'step')
            ax[i, j].hist(bins[:-1], bins=bins, weights=cD, color='red', histtype=u'step')
            ax[i, j].text(x=0.95, y=0.9, s="dijet pT >  %d "%thresholds[i*nCol+j], ha='right', transform=ax[i, j].transAxes)
            

            fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_dijetPtCut.png", bbox_inches='tight')