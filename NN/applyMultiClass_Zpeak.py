import numpy as np
import pandas as pd
from functions import loadMultiParquet, cut
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import sys, os, glob, re
import math
from applyMultiClass_Hpeak import getPredictions, splitPtFunc

def main(nReal, pTClass):
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]

    pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
    # check for which fileNumbers the predictions is available
    isMCList = [0, 36, 20, 21, 22, 23]
    fileNumberList = []
    for isMC in isMCList:
        fileNumberProcess = []
        fileNamesProcess = glob.glob(pathToPredictions+"/yMC%d_fn*pt%d*.parquet"%(isMC, pTClass))
        for fileName in fileNamesProcess:
            match = re.search(r'_fn(\d+)_pt', fileName)
            if match:
                fn = match.group(1)
                fileNumberProcess.append(int(fn))
                
            else:
                pass
                #print("Number not found")
        fileNumberList.append(fileNumberProcess)
        print(len(fileNumberProcess), " predictions files for process MC : ", isMC)


    # load the files where the prediction is available
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl'], returnNumEventsTotal=True, selectFileNumberList=fileNumberList, returnFileNumberList=True)
    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][pTClass]    
    dfs = preprocessMultiClass(dfs, pTmin, pTmax, suffix)   # get the dfs with the cut in the pt class

    # mask in dijet_pt. If you want to have a further mask in the dijet pt class
    minPt, maxPt = None, None #180, -1
    if (minPt is not None) | (maxPt is not None):
        dfs, masks = splitPtFunc(dfs, minPt, maxPt)
        splitPt = True
    else:
        masks=None
        splitPt=False

    W = dfs[0].sf
    W_1 = 5.261e+03/numEventsList[1]*dfs[1].sf*nReal*0.774/1017*1000
    W_2 = 1012./numEventsList[2]*dfs[2].sf*nReal*0.774/1017*1000
    W_3 = 114.2/numEventsList[3]*dfs[3].sf*nReal*0.774/1017*1000
    W_4 = 25.34/numEventsList[4]*dfs[4].sf*nReal*0.774/1017*1000
    W_5 = 12.99/numEventsList[5]*dfs[5].sf*nReal*0.774/1017*1000
    dfs = [dfs[0], pd.concat(dfs[1:])]
    
    W_Z = np.concatenate([W_1, W_2, W_3, W_4, W_5])
    
    for idx, df in enumerate(dfs):
        print("Length of df %d : %d"%(idx, len(df)))
    

    print("Amount of total Z signal : %.3f"%(np.sum(W_Z)))
    print("Amount of tota Data      : %.3f"%(np.sum(W  )))
    print("Initial significance whole spectrum : %.2f"%(np.sum(W_Z)/np.sqrt(np.sum(W))))
    
    
    YPred_data, YPred_Z100to200, YPred_Z200to400, YPred_Z400to600, YPred_Z600to800, YPred_Z800toInf = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass)
    YPred_Z = np.concatenate((YPred_Z100to200, YPred_Z200to400, YPred_Z400to600, YPred_Z600to800, YPred_Z800toInf))
    assert len(YPred_data)==len(dfs[0]), "%d %d"%(len(YPred_data), len(dfs[0]))
    def eff(t0=1, t1=1, t2=0):
        mask = (YPred_data[:,2]>t2) & (YPred_data[:,1]<t1) & (YPred_data[:,0]<t0) & np.array(dfs[0].dijet_mass>75) & np.array(dfs[0].dijet_mass<105)
        data = np.sum(W[mask])
        mask = (YPred_Z[:,2]>t2) & (YPred_Z[:,1]<t1) & (YPred_Z[:,0]<t0)  & np.array(dfs[1].dijet_mass>75) & np.array(dfs[1].dijet_mass<105)
        signal= np.sum(W_Z[mask])
        print(signal, t0, t1, t2, signal/np.sqrt(data))
        
        if math.isclose(data, 0):
            return 0
        else:
            return signal/np.sqrt(data+0.0000001)
    bayes=True
    maxt0, maxt1, maxt2 = 1, 1, -1
    if bayes:
        maxt2 = -1
        pbounds = { 
                    #'t0': (0.05, 0.99),
                    #'t1': (0.05, 0.99),
                    't2': (0.0, 0.95),
                    }
        optimizer = BayesianOptimization(
        f=eff,        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1, allow_duplicate_points=True
        )
        optimizer.maximize(init_points=10,         n_iter=50)
        maxt2=optimizer.max["params"]["t2"]
        #maxt0=optimizer.max["params"]["t0"]
    print("significance in mass region before cut : %.2f"%eff(1, 1, 0))
    print("significance in mass region after cut : ", eff(t0=maxt0, t1=maxt1, t2=maxt2))

# cuts
    #dfs=cut(dfs, 'dijet_pt', 125, None)
    maskData = (YPred_data[:,2]>maxt2) & (YPred_data[:,1]<maxt1) &  (YPred_data[:,0]<maxt0)
    maskZ = (YPred_Z[:,2]>maxt2) & (YPred_Z[:,1]<maxt1) &  (YPred_Z[:,0]<maxt0)
    print("Amount of Z signal after cut: %.3f"%(np.sum(W_Z[maskZ])))
    print("Amount of Data     after cut: %.3f"%(np.sum(W[maskData])))
    print("Eff of Z signal    after cut: %.1f"%(np.sum(W_Z[maskZ])/np.sum(W_Z)*100))
    print("Eff of Data        after cut: %.2f"%(np.sum(W[maskData])/np.sum(W)*100))
    visibilityFactor=100

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))
    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
    bins = np.linspace(0, 450, 200)
    
    counts = np.histogram(dfs[0].dijet_mass[maskData], bins=bins)[0]
    ax[0].errorbar(x=(bins[:-1] + bins[1:])/2, y=counts/np.diff(bins), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none', markersize=5)
    #ax[0].text(x=0.9, y=0.6)

    counts_sig = np.histogram(dfs[1].dijet_mass[maskZ], bins=bins, weights=W_Z[maskZ])[0]
    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig*visibilityFactor/np.diff(bins), color='blue', label=r'ZJets MC $\times %d$'%visibilityFactor)
    

    # do fit
    x1, x2 = 70, 80
    x3, x4 = 110, 190
    x = (bins[1:]+bins[:-1])/2
    fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))

    coefficients = np.polyfit(x=x[fitregion],
                              y=(counts/np.diff(bins))[fitregion], deg=6,
                              w=1/(np.sqrt(counts[fitregion])+0.0000001))
    fitted_polynomial = np.poly1d(coefficients)
    y_values = fitted_polynomial(x)

    
    ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')
    # subtract the fit

    #plot the data - fit
    
    
    ax[1].errorbar(x, (counts/np.diff(bins) - y_values), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none')
    ylim  = np.max(abs((counts/(np.diff(bins)))[(x>x1) & (x<x4)]-y_values[(x>x1) & (x<x4)]))*1.1
    ax[1].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig/np.diff(bins), color='blue', label='ZJets MC')
    ax[1].hlines(y=0 ,xmin=bins[0], xmax = bins[-1], color='red')
    ax[1].set_ylim(-ylim, ylim)

    ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))
    #ax[1].set_ylabel(r"$\frac{\text{Data - Fit }}{\sqrt{\text{Data}}}$")
    ax[1].set_ylabel(r"Data - Fit")
    ax[1].set_xlabel("Dijet Mass [GeV]")
    from matplotlib.patches import Rectangle
    rect1 = Rectangle((x1, ax[0].get_ylim()[0]), x2 - x1, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green', label='Fit region')
    rect2 = Rectangle((x3, ax[0].get_ylim()[0]), x4 - x3, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')
    ax[0].add_patch(rect1)
    ax[0].add_patch(rect2)
    rect1 = Rectangle((x1, ax[1].get_ylim()[0]), x2 - x1, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
    rect2 = Rectangle((x3, ax[1].get_ylim()[0]), x4 - x3, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')

    ax[1].add_patch(rect1)
    ax[1].add_patch(rect2)

    ax[0].set_xlim(40, 250)
    ax[0].legend()


    if splitPt:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/%s/dijetMass/dijet_mass_HS_%d-%dGeV.png"%(suffix, minPt, maxPt)
        
    else:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/%s/dijetMass/dijet_mass_HS.png"%suffix
    fig.savefig(outName, bbox_inches='tight')
    print("Saved in " + outName)

    






    #################


    #threshold = np.linspace(0, 0.9, 10)
    #for t in threshold:
    #    maskData = (YPred_data[:,2]>t) & (YPred_data[:,1]<1) &  (YPred_data[:,0]<1)
    #    maskZ = (YPred_Z[:,2]>t) & (YPred_Z[:,1]<1) &  (YPred_Z[:,0]<1)
    #    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))
    #    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
    #    bins = np.linspace(0, 450, 200)
#
    #    counts = np.histogram(dfs[0].dijet_mass[maskData], bins=bins)[0]
    #    ax[0].errorbar(x=(bins[:-1] + bins[1:])/2, y=counts/np.diff(bins), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none', markersize=5)
    #    #ax[0].text(x=0.9, y=0.6)
#
    #    counts_sig = np.histogram(dfs[1].dijet_mass[maskZ], bins=bins, weights=W_Z[maskZ])[0]
    #    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig*visibilityFactor/np.diff(bins), color='blue', label=r'ZJets MC $\times %d$'%visibilityFactor)
#
#
    #    # do fit
    #    x1, x2 = 60, 80
    #    x3, x4 = 110, 150
    #    x = (bins[1:]+bins[:-1])/2
    #    fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))
#
    #    coefficients = np.polyfit(x=x[fitregion],
    #                              y=(counts/np.diff(bins))[fitregion], deg=6,
    #                              w=1/(np.sqrt(counts[fitregion])+0.0000001))
    #    fitted_polynomial = np.poly1d(coefficients)
    #    y_values = fitted_polynomial(x)
#
#
    #    ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')
    #    # subtract the fit
#
    #    #plot the data - fit
    #    ax[1].errorbar(x, (counts/np.diff(bins) - y_values), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none')
    #    ylim  = np.max(abs((counts/(np.diff(bins)))[(x>x1) & (x<x4)]-y_values[(x>x1) & (x<x4)]))*1.1
    #    ax[1].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig/np.diff(bins), color='blue', label='ZJets MC')
    #    ax[1].hlines(y=0 ,xmin=bins[0], xmax = bins[-1], color='red')
    #    ax[1].set_ylim(-ylim, ylim)
#
    #    ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))
    #    #ax[1].set_ylabel(r"$\frac{\text{Data - Fit }}{\sqrt{\text{Data}}}$")
    #    ax[1].set_ylabel(r"Data - Fit")
    #    ax[1].set_xlabel("Dijet Mass [GeV]")
    #    from matplotlib.patches import Rectangle
    #    rect1 = Rectangle((x1, ax[0].get_ylim()[0]), x2 - x1, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green', label='Fit region')
    #    rect2 = Rectangle((x3, ax[0].get_ylim()[0]), x4 - x3, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')
    #    ax[0].add_patch(rect1)
    #    ax[0].add_patch(rect2)
    #    rect1 = Rectangle((x1, ax[1].get_ylim()[0]), x2 - x1, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
    #    rect2 = Rectangle((x3, ax[1].get_ylim()[0]), x4 - x3, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
#
    #    ax[1].add_patch(rect1)
    #    ax[1].add_patch(rect2)
#
    #    ax[0].set_xlim(40, 250)
    #    ax[0].legend()
    #    outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/%s/dijetMass/dijet_mass_HS_%.1f.png"%(suffix, t)
    #    fig.savefig(outName, bbox_inches='tight')
    #    print("Saved in " + outName)
    return

if __name__ =="__main__":
    nReal = int(sys.argv[1])    # number of real files
    pTClass = int(sys.argv[2])  # pT class 0 inclusive 1 low, 2 medium, 3 high
    
    main(nReal=nReal, pTClass=pTClass)