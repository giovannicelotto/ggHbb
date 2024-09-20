import numpy as np
import pandas as pd
import glob
from functions import loadMultiParquet, cut, getXSectionBR
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import sys, os
import math
def splitPtFunc(dfs, minPt, maxPt):
        ' if splitPt is true this func is used to cut all the dfs in that class of pt'
        masks = []
        dfsNew=[]
        if maxPt==-1:
            for df in dfs:
                mask = np.array(minPt<=df.dijet_pt)
                df = df[mask]
                dfsNew.append(df)
                masks.append(mask)
        else:
            for df in dfs:
                mask = np.array((minPt<=df.dijet_pt) & (df.dijet_pt<maxPt))
                df = df[mask]
                dfsNew.append(df)
                masks.append((minPt<=df.dijet_pt) & (df.dijet_pt<maxPt))
        
        return dfsNew, masks

def getPredictions(fileNumberList, pathToPredictions, splitPt, masks, isMC, pTClass):
        '''
        Open for each sample the corresponding NN predictions previously computed and saved somewhere in pathToPredictions.
        If splitPt is true also apply the same mask used for the dfs
        '''
        YPredictions = []
        for idx, fileNumberListProcess in enumerate(fileNumberList):
            fileNames=[]
            for fileNumber in fileNumberListProcess:  #data of bparking1A
                #print(fileNumber)
                if os.path.exists(pathToPredictions+"/yMC%d_fn%d_pt%d.parquet"%(int(isMC[idx]), int(fileNumber), int(pTClass))):
                    fileNames.append(pathToPredictions+"/yMC%d_fn%d_pt%d.parquet"%(int(isMC[idx]), int(fileNumber), int(pTClass)))
                else:
                    print("FileNotFound")
                    pass
                    #print("File not found. skipping")
            YPred = pd.read_parquet(fileNames)
            if splitPt:
                print(len(masks[idx]), len(YPred))
                YPred=YPred[masks[idx]]
        
            YPred = np.array(YPred)
            YPredictions.append(YPred)


        return YPredictions

def main(nReal, minPt, maxPt, pTClass):
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others",
            ]
        
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl'], returnNumEventsTotal=True, returnFileNumberList=True)
        
    dfs = preprocessMultiClass(dfs)

    # mask in dijet_pt
    if (minPt is not None) & (maxPt is not None):
        dfs, masks = splitPtFunc(dfs, minPt, maxPt)
        splitPt = True
    else:
        masks=None
        splitPt=False

    W = dfs[0].sf
    W_1 = getXSectionBR()/numEventsList[1]*dfs[1].sf*nReal*0.774/1017*1000
    
    
    for idx, df in enumerate(dfs):
        print("Length of df %d : %d"%(idx, len(df)))
    

    print("Amount of H signal: %.3f"%(np.sum(W_1)))
    print("Amount of Data    : %.3f"%(np.sum(W  )))
    print("Initial significance : %.2f"%(np.sum(W_1)/np.sqrt(np.sum(W))))
    
    pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
    
    YPred_data, YPred_H = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=[0, 1], pTClass=pTClass)

    fig, ax = plt.subplots(1, 1)
    x_bins, y_bins = np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    hist, x_edges, y_edges = np.histogram2d(x=YPred_data[:,0], y=YPred_data[:,1], bins=[x_bins, y_bins])
    ax.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='YlGnBu', alpha=0.6)
    hist=hist/np.sum(hist)*100
    for y in range(len(y_bins)-1):
        for x in range(len(x_bins)-1):
            plt.text((x+0.5)/(len(y_bins)-1) , (y+0.5)/(len(y_bins)-1) , '%.1f' % hist[x, y],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16,

                    )
    ax.set_xlabel("Data score")
    ax.set_ylabel("ggH score")
    fig.savefig("/t3home/gcelotto/ggHbb/NN/output/multiClass/dataPredictions.png", bbox_inches='tight')


    def eff(t0, t1):
        mask = (YPred_data[:,1]>t1) &  (YPred_data[:,0]<t0) & np.array(dfs[0].dijet_mass>95) & np.array(dfs[0].dijet_mass<155)
        data = np.sum(W[mask])
        mask = (YPred_H[:,1]>t1) &  (YPred_H[:,0]<t0)  & np.array(dfs[1].dijet_mass>95) & np.array(dfs[1].dijet_mass<155)
        signal= np.sum(W_1[mask])
        print(signal, t0, t1, signal/np.sqrt(data))
        
        if math.isclose(data, 0):
            return 0
        else:
            return signal/np.sqrt(data+0.0000001)

    maxt0, maxt1 = -1, -1
    
    pbounds = {'t0': (0., 1.),
               't1': (0., 1.)}
    optimizer = BayesianOptimization(
    f=eff,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    allow_duplicate_points=True
)
    
    optimizer.maximize(
    init_points=5,
    n_iter=50,
)

    maxt0=optimizer.max["params"]["t0"]
    maxt1=optimizer.max["params"]["t1"]
    print("significance in mass region before cut : %.2f"%eff(1, 0))
    #print("maximum of efficiency after bayesian optimization in mass region 75-105: ", eff(t0=maxt0, t1=maxt1))

    maskData = (YPred_data[:,1]>maxt1) &  (YPred_data[:,0]<maxt0)
    maskH = (YPred_H[:,1]>maxt1) &  (YPred_H[:,0]<maxt0)
    print("Amount of H signal after cut: %.3f"%(np.sum(W_1[maskH])))
    print("Amount of Data     after cut: %.3f"%(np.sum(W[maskData])))
    print("Eff of H signal    after cut: %.1f"%(np.sum(W_1[maskH])/np.sum(W_1)*100))
    print("Eff of Data        after cut: %.2f"%(np.sum(W[maskData])/np.sum(W)*100))
    visibilityFactor=1000

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))
    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
    bins = np.linspace(0, 450, 200)
    
    counts = np.histogram(dfs[0].dijet_mass[maskData], bins=bins)[0]
    ax[0].errorbar(x=(bins[:-1] + bins[1:])/2, y=counts/np.diff(bins), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none', markersize=5)
    #ax[0].text(x=0.9, y=0.6)

    counts_sig = np.histogram(dfs[1].dijet_mass[maskH], bins=bins, weights=W_1[maskH])[0]
    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig*visibilityFactor/np.diff(bins), color='blue', label=r'ZJets MC $\times %d$'%visibilityFactor)
    

    # do fit
    x1, x2 = 75, 100
    x3, x4 = 150, 220
    x = (bins[1:]+bins[:-1])/2
    fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))

    coefficients = np.polyfit(x=x[fitregion],
                              y=(counts/np.diff(bins))[fitregion], deg=5,
                              w=1/(np.sqrt(counts[fitregion])+0.0000001))
    fitted_polynomial = np.poly1d(coefficients)
    y_values = fitted_polynomial(x)

    
    ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')
    # subtract the fit

    #plot the data - fit
    
    
    ax[1].errorbar(x, (counts/np.diff(bins) - y_values), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none')
    ylim  = np.max(abs((counts/(np.diff(bins)))[(x>x1) & (x<x4)]-y_values[(x>x1) & (x<x4)]))*1.1
    ax[1].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig/np.diff(bins), color='blue', label='ggH MC')
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

    ax[0].set_xlim(65, 250)
    ax[0].legend()


    if splitPt:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/dijetMass/dijetMassHiggs_HS_%d-%dGeV.png"%(minPt, maxPt)
        
    else:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/dijetMass/dijetMassHiggs_HS_inclusive.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saved in " + outName)
    return

if __name__ =="__main__":
    nReal = int(sys.argv[1])
    if len(sys.argv)>2:
        minPt = int(sys.argv[2])
        maxPt = int(sys.argv[3])
        print("Lim to %d - %d"%(minPt, maxPt))
        pTClass = int(sys.argv[4])
    else:
        minPt = None
        maxPt = None
    main(nReal=nReal, minPt=minPt, maxPt=maxPt, pTClass=pTClass)