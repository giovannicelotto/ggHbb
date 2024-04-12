import numpy as np
import pandas as pd
import glob
from NN_multiclass import getFeatures
from tensorflow.keras.models import load_model
from functions import loadMultiParquet, cut
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import sys
import math
def splitPtFunc(dfs, minPt, maxPt):
        ' if splitPt is true this func is used to cut all the dfs in that class of pt'
        if maxPt==-1:
            maskPtData = np.array(minPt<=dfs[0].dijet_pt)
            maskPtMC1  = np.array(minPt<=dfs[1].dijet_pt)
            maskPtMC2  = np.array(minPt<=dfs[2].dijet_pt)
            maskPtMC3  = np.array(minPt<=dfs[3].dijet_pt)
            maskPtMC4  = np.array(minPt<=dfs[4].dijet_pt)
        else:
            maskPtData = np.array((minPt<=dfs[0].dijet_pt) & (dfs[0].dijet_pt<maxPt))
            maskPtMC1  = np.array((minPt<=dfs[1].dijet_pt) & (dfs[1].dijet_pt<maxPt))
            maskPtMC2  = np.array((minPt<=dfs[2].dijet_pt) & (dfs[2].dijet_pt<maxPt))
            maskPtMC3  = np.array((minPt<=dfs[3].dijet_pt) & (dfs[3].dijet_pt<maxPt))
            maskPtMC4  = np.array((minPt<=dfs[4].dijet_pt) & (dfs[4].dijet_pt<maxPt))
        dfs[0] = dfs[0][maskPtData]
        dfs[1] = dfs[1][maskPtMC1]
        dfs[2] = dfs[2][maskPtMC2]
        dfs[3] = dfs[3][maskPtMC3]
        dfs[4] = dfs[4][maskPtMC4]
        masks = [maskPtData, maskPtMC1, maskPtMC2, maskPtMC3, maskPtMC4]
        return dfs, masks

def getPredictions(fileNumberList, pathToPredictions, splitPt, masks):
        '''Open for each sample the corresponding NN predictions previously copmuted and saved somewhere in pathToPredictions.
        If splitPt is true also apply the same mask used for the dfs'''
        fileNames=[]
        for fileNumber in fileNumberList[0]:  #data of bparking1A
            #print("Opening ", pathToPredictions+"/y0_%d.parquet"%(int(fileNumber)))
            fileNames.append(pathToPredictions+"/y0_%d.parquet"%(int(fileNumber)))
        YPred_data = pd.read_parquet(fileNames)
        if splitPt:
            YPred_data=YPred_data[masks[0]]
        fileNames=[]
        for fileNumber in fileNumberList[1]:  #data of ZJets200to400
            #print("Opening ", pathToPredictions+"/y20_%d.parquet"%(int(fileNumber)))
            fileNames.append(pathToPredictions+"/y20_%d.parquet"%(int(fileNumber)))
        YPred_Z200to400 = pd.read_parquet(fileNames)
        if splitPt:
            YPred_Z200to400=YPred_Z200to400[masks[1]]
        fileNames=[]
        for fileNumber in fileNumberList[2]:  #data of ZJets400to600
            #print("Opening ", pathToPredictions+"/y21_%d.parquet"%(int(fileNumber)))
            fileNames.append(pathToPredictions+"/y21_%d.parquet"%(int(fileNumber)))
        YPred_Z400to600 = pd.read_parquet(fileNames)
        if splitPt:
            YPred_Z400to600=YPred_Z400to600[masks[2]]
        fileNames=[]
        for fileNumber in fileNumberList[3]:  #data of ZJets600to800
            #print("Opening ", pathToPredictions+"/y22_%d.parquet"%(int(fileNumber)))
            fileNames.append(pathToPredictions+"/y22_%d.parquet"%(int(fileNumber)))
        YPred_Z600to800 = pd.read_parquet(fileNames)
        if splitPt:
            YPred_Z600to800=YPred_Z600to800[masks[3]]
        fileNames=[]
        for fileNumber in fileNumberList[4]:  #data of ZJets800toInf
            fileNames.append(pathToPredictions+"/y23_%d.parquet"%(int(fileNumber)))
        YPred_Z800toInf = pd.read_parquet(fileNames)
        if splitPt:
            YPred_Z800toInf=YPred_Z800toInf[masks[4]]
            
        del fileNames
        YPred_data = np.array(YPred_data)
        YPred_Z = np.concatenate((np.array(YPred_Z200to400), np.array(YPred_Z400to600), np.array(YPred_Z600to800), np.array(YPred_Z800toInf)))

        return YPred_data, YPred_Z

def main(nReal, minPt, maxPt):
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]
        
    featuresForTraining, columnsToRead = getFeatures()
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl'], returnNumEventsTotal=True, returnFileNumberList=True)
        
    dfs = preprocessMultiClass(dfs)

    # mask in dijet_pt
    
    if (minPt is not None) & (maxPt is not None):
        dfs, masks = splitPtFunc(dfs, minPt, maxPt)
        splitPt = True
    else:
        masks=None
        splitPt=False

    W = dfs[0].sf
    W_1 = 1012./numEventsList[1]*dfs[1].sf*nReal*0.774/1017*1000
    W_2 = 114.2/numEventsList[2]*dfs[2].sf*nReal*0.774/1017*1000
    W_3 = 25.34/numEventsList[3]*dfs[3].sf*nReal*0.774/1017*1000
    W_4 = 12.99/numEventsList[4]*dfs[4].sf*nReal*0.774/1017*1000
    dfs = [dfs[0], pd.concat(dfs[1:])]
    print(len(dfs[1]))
    W_Z = np.concatenate([W_1, W_2, W_3, W_4])
    
    for idx, df in enumerate(dfs):
        print("Length of df %d : %d"%(idx, len(df)))
    

    print("Amount of Z signal: %.3f"%(np.sum(W_Z)))
    print("Amount of Data    : %.3f"%(np.sum(W  )))
    print("Initial significance : %.2f"%(np.sum(W_Z)/np.sqrt(np.sum(W))))
    
    pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
    print("my files", fileNumberList)
    
    YPred_data, YPred_Z = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks)

    def eff(t0, t2):
        mask = (YPred_data[:,2]>t2) &  (YPred_data[:,0]<t0) & np.array(dfs[0].dijet_mass>75) & np.array(dfs[0].dijet_mass<105)
        data = np.sum(W[mask])
        mask = (YPred_Z[:,2]>t2) &  (YPred_Z[:,0]<t0)  & np.array(dfs[1].dijet_mass>75) & np.array(dfs[1].dijet_mass<105)
        signal= np.sum(W_Z[mask])
        print(signal, t0, t2, signal/np.sqrt(data))
        
        if math.isnan(signal/np.sqrt(data)):
            return 0
        else:
            return signal/np.sqrt(data)

    maxt0, maxt2 = -1, -1
    
    pbounds = {'t0': (0.05, 0.9),
               't2': (0.05, 0.9)}
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
    maxt2=optimizer.max["params"]["t2"]
    print("significance in mass region before cut : %.2f"%eff(1, 0))
    print("maximum of efficiency after bayesian optimization in mass region 75-105: ", eff(t0=maxt0, t2=maxt2))
    #for t0 in np.linspace(0.01, 0.99, 20):
    #    for t2 in np.linspace(0.05, 0.99, 20):
    #        sig=eff(t0, t2)
    #        if sig>maxsig:
    #            maxsig=sig
    #            maxt0=t0
    #            maxt2=t2


    #maxt0=1
    #maxt2=0

# cuts
    #dfs=cut(dfs, 'dijet_pt', 125, None)
    maskData = (YPred_data[:,2]>maxt2) &  (YPred_data[:,0]<maxt0)
    maskZ = (YPred_Z[:,2]>maxt2) &  (YPred_Z[:,0]<maxt0)
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
    x3, x4 = 110, 220
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

    ax[0].set_xlim(65, 250)
    ax[0].legend()


    if splitPt:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/dijetMass/dijet_mass_HS_%d-%dGeV.png"%(minPt, maxPt)
        
    else:
        outName = "/t3home/gcelotto/ggHbb/NN/output/multiClass/dijetMass/dijet_mass_HS_inclusive.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saved in " + outName)
    return

if __name__ =="__main__":
    nReal = int(sys.argv[1])
    if len(sys.argv)>2:
        minPt = int(sys.argv[2])
        maxPt = int(sys.argv[3])
        print("Lim to %d - %d"%(minPt, maxPt))
    else:
        minPt = None
        maxPt = None
    main(nReal=nReal, minPt=minPt, maxPt=maxPt)