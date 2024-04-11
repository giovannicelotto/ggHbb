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

def main(nReal):
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]
        
    featuresForTraining, columnsToRead = getFeatures()
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, columns=['sf', 'dijet_mass', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl'], returnNumEventsTotal=True, returnFileNumberList=True)
    print(fileNumberList)
    
    print("Length")
    print(len(dfs[0]))
    print(len(dfs[1]))
    print(len(dfs[2]))
    print(len(dfs[3]))
    print(len(dfs[4]))
    print(len(dfs[1])+len(dfs[2])+len(dfs[3])+len(dfs[4]))
    print("Cut applied")
    dfs = preprocessMultiClass(dfs)
    print(len(dfs[0]))
    print(len(dfs[1]))
    print(len(dfs[2]))
    print(len(dfs[3]))
    print(len(dfs[4]))
    print(len(dfs[1])+len(dfs[2])+len(dfs[3])+len(dfs[4]))
    
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
    
    
    #model = load_model("/t3home/gcelotto/ggHbb/NN/output/multiClass/model/model.h5")

    pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
    fileNames=[]
    print("my files", fileNumberList)
    for fileNumber in fileNumberList[0]:  #data of bparking1A
        #print("Opening ", pathToPredictions+"/y0_%d.parquet"%(int(fileNumber)))
        fileNames.append(pathToPredictions+"/y0_%d.parquet"%(int(fileNumber)))
    YPred_data = pd.read_parquet(fileNames)
    fileNames=[]
    for fileNumber in fileNumberList[1]:  #data of ZJets200to400
        print("Opening ", pathToPredictions+"/y20_%d.parquet"%(int(fileNumber)))
        fileNames.append(pathToPredictions+"/y20_%d.parquet"%(int(fileNumber)))
    YPred_Z200to400 = pd.read_parquet(fileNames)
    fileNames=[]
    for fileNumber in fileNumberList[2]:  #data of ZJets400to600
        #print("Opening ", pathToPredictions+"/y21_%d.parquet"%(int(fileNumber)))
        fileNames.append(pathToPredictions+"/y21_%d.parquet"%(int(fileNumber)))
    YPred_Z400to600 = pd.read_parquet(fileNames)
    fileNames=[]
    for fileNumber in fileNumberList[3]:  #data of ZJets600to800
        #print("Opening ", pathToPredictions+"/y22_%d.parquet"%(int(fileNumber)))
        fileNames.append(pathToPredictions+"/y22_%d.parquet"%(int(fileNumber)))
    YPred_Z600to800 = pd.read_parquet(fileNames)
    fileNames=[]
    for fileNumber in fileNumberList[4]:  #data of ZJets800toInf
        #print("Opening ", pathToPredictions+"/y23_%d.parquet"%(int(fileNumber)))
        fileNames.append(pathToPredictions+"/y23_%d.parquet"%(int(fileNumber)))
    YPred_Z800toInf = pd.read_parquet(fileNames)
        
    del fileNames
    YPred_data = np.array(YPred_data)
    print(len(YPred_Z200to400))
    print(len(YPred_Z400to600))
    print(len(YPred_Z600to800))
    print(len(YPred_Z800toInf))
    
    YPred_Z = np.concatenate((np.array(YPred_Z200to400), np.array(YPred_Z400to600), np.array(YPred_Z600to800), np.array(YPred_Z800toInf)))

    print("Len ypreddata", len(YPred_data))
    print("Len dfs 0", len(dfs[0]))
    print("len ypred_z", len(YPred_Z))
    #dfs[1]  = scale(dfs[1], scalerName= "/t3home/gcelotto/ggHbb/NN/input/multiclass/myScaler.pkl" ,fit=False)
    #YPred_Z = model.predict(dfs[1][featuresForTraining])
    #dfs[1] = unscale(dfs[1], scalerName= "/t3home/gcelotto/ggHbb/NN/input/multiclass/myScaler.pkl")

    def eff(t0, t2):
        mask = (YPred_data[:,2]>t2) &  (YPred_data[:,0]<t0) & np.array(dfs[0].dijet_mass>75) & np.array(dfs[0].dijet_mass<105)
        data = np.sum(W[mask])
        mask = (YPred_Z[:,2]>t2) &  (YPred_Z[:,0]<t0)  & np.array(dfs[1].dijet_mass>75) & np.array(dfs[1].dijet_mass<105)
        signal= np.sum(W_Z[mask])
        print(signal, t0, t2, signal/np.sqrt(data))
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
    print("Eff of Z signal after cut: %.1f"%(np.sum(W_Z[maskZ])/np.sum(W_Z)*100))
    print("Eff of Data     after cut: %.2f"%(np.sum(W[maskData])/np.sum(W)*100))
    visibilityFactor=100

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))
    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
    bins = np.linspace(0, 450, 200)
    
    counts = np.histogram(dfs[0].dijet_mass[maskData], bins=bins)[0]
    ax[0].errorbar(x=(bins[:-1] + bins[1:])/2, y=counts/np.diff(bins), yerr=np.sqrt(counts)/np.diff(bins), color='black', marker='o', linestyle='none', markersize=5)

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



    fig.savefig("/t3home/gcelotto/ggHbb/NN/output/multiClass/dijet_mass_HS.png", bbox_inches='tight')
    print("Saved in " + "/t3home/gcelotto/ggHbb/NN/output/multiClass/dijet_mass_HS.png")
    return

if __name__ =="__main__":
    nReal = int(sys.argv[1])
    main(nReal=nReal)