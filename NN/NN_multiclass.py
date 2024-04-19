import shap
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import sys
import numpy as np
import pandas as pd
from models import getModelMultiClass
from plotsForNN import doPlotLoss, getShap
from functions import loadMultiParquet
from helpersForNN import scale, unscale, preprocessMultiClass
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
'''
nReal   = number of Files used for realData training, testing (10 recommended). All the datasets are cut after 3*1e4 events
nMC     = number of Files used for    MC    training, testing (10 recommended). All the datasets are cut after 3*1e4 events
doTrain = bool. Yes: Do training and do performance plots. False: Load X, Y, Y_Pred, W  and redo the plots.
ptClass = ptClass in dijetPT:
            0       inclusive
            1       0-30 
            2       30-100
            3       100-inf
'''
def doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder):
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    Ytest=pd.DataFrame(Ytest).reset_index(drop=True)
    Ytrain=pd.DataFrame(Ytrain).reset_index(drop=True)
    YPredTest = pd.DataFrame(YPredTest).reset_index(drop=True)
    YPredTrain = pd.DataFrame(YPredTrain).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1)
    thresholds = np.linspace(0, 1, 1000)
    labels = ['Data', 'ggH', 'ZJets']
    for classIdx in range(3):
        classesIdx = [0, 1, 2]
        classesIdx.remove(classIdx)
        
        maskClass = Ytest.iloc[:,classIdx]>0.99
        maskClassTrain = Ytrain.iloc[:,classIdx]>0.99
        print(len(Ytest[maskClass]), " events in class ", classIdx)

        tpr, fpr = [], []
        tprTrain, fprTrain = [], []
        for t in thresholds:
            #print((YPredTest[maskClass].iloc[:,classIdx] > t).sum())
            tpr.append(Wtest[(YPredTest.iloc[:,classIdx] > t) & (maskClass)].sum()/Wtest[maskClass].sum())
            fpr.append(Wtest[(YPredTest.iloc[:,classIdx] > t) & (~maskClass)].sum()/Wtest[~maskClass].sum())

            tprTrain.append(Wtrain[(YPredTrain.iloc[:,classIdx]> t) & (maskClassTrain)].sum()/Wtrain[maskClassTrain].sum())
            fprTrain.append(Wtrain[(YPredTrain.iloc[:,classIdx]> t) & (~maskClassTrain)].sum()/Wtrain[~maskClassTrain].sum())

        
        from scipy.integrate import simpson
        
        auc = -simpson(tpr, fpr)
        auc_train = -simpson(tprTrain, fprTrain)
        ax.plot(fpr, tpr, marker='o', markersize=1, label='%s Test AUC %.2f'%(labels[classIdx], auc))
        ax.plot(fprTrain, tprTrain, linestyle='dotted', label='%s Train AUC %.2f'%(labels[classIdx], auc_train))
        ax.plot(thresholds, thresholds, linestyle='dotted', color='green')

        ax.grid(True)
        ax.set_ylabel("Signal Efficiency")
        ax.set_xlabel("Background Efficiency")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        

    handles, newLabels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in [0, 2, 4, 1, 3, 5]]
    newLabels = [newLabels[i] for i in [0, 2, 4, 1, 3, 5]]
    ax.legend(handles, newLabels, ncols=2)
    hep.cms.label(ax=ax)
    fig.savefig(outFolder +"/performance/roc.png", bbox_inches='tight')



# confusion matrix
    ax.clear()
    confusionM = np.ones((3, 3))
    for trueLabel in range(3):
        # true label
        maskClass = Ytest.iloc[:,trueLabel]>0.99
        den = Wtest[maskClass].sum()
        for predictedLabel in range(3):
            #print(trueLabel, predictedLabel)
            #print(YPredTest[maskClass].idxmax(axis=1))
            num = (Wtest[(YPredTest.idxmax(axis=1)==predictedLabel) & (maskClass)]).sum()
            confusionM[trueLabel, predictedLabel] = num/den
            #print(trueLabel, predictedLabel, " : ", num)
    print("Confusion matrix : \n",confusionM)
    im = ax.matshow(confusionM, cmap=plt.cm.jet, alpha=0.7, norm=colors.LogNorm(vmin=0.01, vmax=1))
    for y in range(3):
        for x in range(3):
            plt.text(x , y , '%.2f' % confusionM[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16
                    )
    #ax.set_xlabel("Generated (bin number)", fontsize=18)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks([0, 1, 2], ['Data', 'ggH', 'ZJets'])
    ax.set_yticks([0, 1, 2], ['Data', 'ggH', 'ZJets'])
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
        
    #ax.legend()
    hep.cms.label(ax=ax)
    fig.savefig(outFolder +"/performance/confusionMatrix.png", bbox_inches='tight')
    plt.close('all')

# want to see if the dijet mass for qcd is smoothly falling.
# take data events with high ggH score
    fig, ax = plt.subplots(1, 1)
    maskClass = Ytest.iloc[:,0]>0.99 # data events (QCD)
    Xtest=Xtest.reset_index(drop=True)
    bins=np.linspace(0, 450, 100)
    #print(  "my maass", Xtest[(maskClass) & (YPredTest.iloc[:,1]>0.5)].dijet_mass)
    ax.set_title("BParking data with ggH score > 0.5")
    ax.hist(Xtest[(maskClass) & (YPredTest.iloc[:,0]<0.1)].dijet_mass, bins=bins, weights=Xtest[(maskClass) & (YPredTest.iloc[:,0]<0.1)].sf)
    fig.savefig(outFolder +"/performance/dijetMass.png", bbox_inches='tight')
    plt.close('all')

# outputTrainTest
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bins= np.linspace(0, 1, 10)
    xlabels = ['Data score', 'ggH score', 'ZJets score']
    for idx, ax in enumerate(axes):
        mask0 = Ytrain.iloc[:,0]>0.99   #real class 0 
        mask1 = Ytrain.iloc[:,1]>0.99
        mask2 = Ytrain.iloc[:,2]>0.99
        counts0 = np.histogram(YPredTrain[mask0].iloc[:,idx], bins=bins, weights=Wtrain[mask0])[0]
        counts1 = np.histogram(YPredTrain[mask1].iloc[:,idx], bins=bins, weights=Wtrain[mask1])[0]
        counts2 = np.histogram(YPredTrain[mask2].iloc[:,idx], bins=bins, weights=Wtrain[mask2])[0]
        counts0, counts1, counts2, = counts0/np.sum(counts0), counts1/np.sum(counts1), counts2/np.sum(counts2)
        ax.hist(bins[:-1], bins=bins, weights=counts0, label=labels[0]+" train", histtype=u'step', color='C0')
        ax.hist(bins[:-1], bins=bins, weights=counts1, label=labels[1]+" train", histtype=u'step', color='C1')
        ax.hist(bins[:-1], bins=bins, weights=counts2, label=labels[2]+" train", histtype=u'step', color='C2')
        
        mask0 = Ytest.iloc[:,0]>0.99
        mask1 = Ytest.iloc[:,1]>0.99
        mask2 = Ytest.iloc[:,2]>0.99
        counts0 = np.histogram(YPredTest[mask0].iloc[:,idx], bins=bins, weights=Wtest[mask0])[0]
        counts1 = np.histogram(YPredTest[mask1].iloc[:,idx], bins=bins, weights=Wtest[mask1])[0]
        counts2 = np.histogram(YPredTest[mask2].iloc[:,idx], bins=bins, weights=Wtest[mask2])[0]
        counts0, counts1, counts2, = counts0/np.sum(counts0), counts1/np.sum(counts1), counts2/np.sum(counts2)
        ax.errorbar((bins[1:]+bins[:-1])/2, counts0, label=labels[0] + " test", alpha=1, marker = 'o', linestyle='none', color='C0')
        ax.errorbar((bins[1:]+bins[:-1])/2, counts1, label=labels[1] + " test", alpha=1, marker = 'o', linestyle='none', color='C1')
        ax.errorbar((bins[1:]+bins[:-1])/2, counts2, label=labels[2] + " test", alpha=1, marker = 'o', linestyle='none', color='C2')

        ax.set_xlabel(xlabels[idx])
        ax.set_yscale('log')
    
    axes[2].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.5)
    fig.savefig(outFolder +"/performance/outputTrainTest.png", bbox_inches='tight')
    plt.close('all')

    return

def getFeatures():
    featuresForTraining=[
       #'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
        'jet1_nMuons', 'jet1_nElectrons', 'jet1_btagDeepFlavB', #'jet1_area',
        'jet1_qgl',
       #'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
       'jet2_nMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB', #'jet2_area',
       'jet2_qgl',
       #'dijet_pt',
       'dijet_eta', 'dijet_phi',
       #'dijet_mass,
       'dijet_dR',
       'dijet_dEta', 'dijet_dPhi', 'dijet_twist',# 'nJets',
       'nJets_20GeV',
       #'ht',
       #'muon_pt',
       #'muon_eta',
       'muon_dxySig',  'muon_dzSig', 'muon_IP3d',
       'muon_sIP3d',
       'dijet_cs',
       'muon_pfRelIso03_all', #'muon_tkIsoId'
       #'muon_tightId',
       ]
    
    columnsToRead = [   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
    'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons',
    'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR',
    'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable', 'dijet_twist', 'nJets',
    'nJets_20GeV',
    'ht', 'muon_pt', 'muon_eta', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d',
    'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_tkIsoId',
    'dijet_cs',
    'sf']
    
    return featuresForTraining, columnsToRead

def loadData(pTmin, pTmax, outFolder):
    featuresForTraining, columnsToRead = getFeatures()
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]

    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=True)
    

    
    nData, nHiggs, nZ = int(1e5), int(1e5) , int(1e5)
    dfs = preprocessMultiClass(dfs, pTmin, pTmax, suffix)


    for idx, df in enumerate(dfs):
        if idx==0:
            dfs[idx] = df.head(nData)
        elif idx==1:
            dfs[idx] = df.head(nHiggs)
        else:
            dfs[idx] = df.head(int(nZ/4))

    for idx, df in enumerate(dfs):
        print("Length of df %d : %d"%(idx, len(df)))
    with open(outFolder+"/status.txt", 'a+') as file:
        for idx, df in enumerate(dfs):
            file.write("Length of df %d : %d"%(idx, len(df)))


    Y_0 = pd.DataFrame([np.ones(len(dfs[0])),  np.zeros(len(dfs[0])), np.zeros(len(dfs[0]))]).T
    Y_1 = pd.DataFrame([np.zeros(len(dfs[1])), np.ones(len(dfs[1])),  np.zeros(len(dfs[1]))]).T
    Y_2 = pd.DataFrame([np.zeros(len(dfs[2])), np.zeros(len(dfs[2])), np.ones(len(dfs[2]))]).T
    Y_3 = pd.DataFrame([np.zeros(len(dfs[3])), np.zeros(len(dfs[3])), np.ones(len(dfs[3]))]).T
    Y_4 = pd.DataFrame([np.zeros(len(dfs[4])), np.zeros(len(dfs[4])), np.ones(len(dfs[4]))]).T
    Y_5 = pd.DataFrame([np.zeros(len(dfs[5])), np.zeros(len(dfs[5])), np.ones(len(dfs[5]))]).T

    # define a weights vector 1 for data 1 for hbb, xsection for the Z boson dataframes. then concat z bosons. divide every weights by the average of the weights
    W_0 = dfs[0].sf
    W_1 = dfs[1].sf
    W_2 = 1012/numEventsList[2]*dfs[2].sf
    W_3 = 114.2/numEventsList[3]*dfs[3].sf
    W_4 = 25.34/numEventsList[4]*dfs[4].sf
    W_5 = 12.99/numEventsList[5]*dfs[5].sf
    W_ZBos = pd.concat((W_2, W_3, W_4, W_5))
    Y_ZBos = pd.concat((Y_2, Y_3, Y_4, Y_5))

    W_ZBos = W_ZBos/W_ZBos.mean()
    W_1 = W_1/W_1.mean()

    Y = np.concatenate((Y_0, Y_1, Y_ZBos))
    X = pd.concat((dfs[0], dfs[1], dfs[2], dfs[3], dfs[4], dfs[5]))
    W = np.concatenate((W_0, W_1, W_ZBos))
    
    X, Y, W = shuffle(X, Y, W, random_state=1999)
    Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = train_test_split(X, Y, W, test_size=0.2, random_state=1999)
    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)
    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest




def HbbClassifier(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, hp, nReal, nMC, inFolder, outFolder):
    
    featuresForTraining, columnsToRead = getFeatures()
    
    with open(outFolder+"/status.txt", 'a+') as file:
        for key, value in hp.items():
            file.write(f"{key}: {value}\n")
    

# get the model, optimizer, compile it and fit to data
    model = getModelMultiClass(inputDim=len(featuresForTraining), nDense=hp['nDense'], nNodes=hp['nNodes'])
    optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") 
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), weighted_metrics=['accuracy'])
    callbacks = []
    earlyStop = EarlyStopping(monitor = 'val_loss', min_delta=hp['min_delta'], patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(Xtrain[featuresForTraining], Ytrain, 
                    sample_weight = Wtrain,
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    callbacks=callbacks, shuffle=True)
    model.save(outFolder +"/model/model.h5")

    model = load_model(outFolder +"/model/model.h5")
    doPlotLoss(fit=fit, outName=outFolder +"/performance/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])
    getShap(Xtest.iloc[:500,:][featuresForTraining], model, outName = outFolder+'/performance/shap.png',
            class_names=['Data', 'ggH', 'ZJets'])
    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])

    Xtrain = unscale(Xtrain,    scalerName= inFolder + "/myScaler.pkl")
    Xtest = unscale(Xtest,      scalerName =  inFolder + "/myScaler.pkl")
    if (True):
        plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder +"/performance/features_train_unscaled.png",
                                    legendLabels = ['Train 0 ', 'Train 1', 'Train 2 ', 'Test 0 ', 'Test 1', 'Test 2 ',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=True,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]])

    np.save(inFolder +"/YPredTest.npy", YPredTest)
    np.save(inFolder +"/YPredTrain.npy", YPredTrain)
    Xtrain.to_parquet(inFolder +"/XTrain.parquet")
    Xtest.to_parquet(inFolder +"/XTest.parquet")
    np.save(inFolder +"/WTest.npy", Wtest)
    np.save(inFolder +"/WTrain.npy", Wtrain)
    np.save(inFolder +"/YTest.npy", Ytest)
    np.save(inFolder +"/YTrain.npy", Ytrain)
    
if __name__ =="__main__":
    nReal, nMC = int(sys.argv[1]), int(sys.argv[2]), 
    doTrain = bool(int(sys.argv[3]))
    ptClass = int(sys.argv[4])

    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][ptClass]
    inFolder, outFolder = "/t3home/gcelotto/ggHbb/NN/input/multiclass/"+suffix, "/t3home/gcelotto/ggHbb/NN/output/multiClass/"+suffix
    if doTrain:
        hp = {
            'epochs'            : 300,
            'patienceES'        : 50,
            'validation_split'  : 0.2,
            'learning_rate'     : 5*1e-5,
            'min_delta'         : 0.005,
            'nNodes'            : [32, 16],
            }
        hp['nDense']=len(hp['nNodes'])
        assert len(hp['nNodes'])==hp['nDense']
        
        data = loadData(pTmin, pTmax, outFolder)
        Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = data


        # do pt cut here

        #if (suffix == 'lowPt') | (suffix == 'mediumPt'):
        #    maskTrain = (Xtrain.dijet_pt>pTmin) & (Xtrain.dijet_pt<pTmax)
        #    maskTest  =  (Xtest.dijet_pt>pTmin)   & (Xtest.dijet_pt<pTmax)
        #    Xtrain, Wtrain, Ytrain = Xtrain.loc[maskTrain], Wtrain.loc[maskTrain], Ytrain.loc[maskTrain]
        #    Xtest, Wtest, Ytest = Xtest.loc[maskTest], Wtest.loc[maskTest], Ytest.loc[maskTest]
        #elif (suffix == 'highPt'):
        #    maskTrain = (Xtrain.dijet_pt>pTmin) 
        #    maskTest  =  (Xtest.dijet_pt>pTmin) 
        #    Xtrain, Wtrain, Ytrain = Xtrain.loc[maskTrain], Wtrain.loc[maskTrain], Ytrain.loc[maskTrain]
        #    Xtest, Wtest, Ytest = Xtest.loc[maskTest], Wtest.loc[maskTest], Ytest.loc[maskTest]
        #elif (suffix=='inclusive'):
        #    pass
        #else:
        #    assert False

        

        # define suffix = inclusive or other
        
        
        with open(outFolder+"/status.txt", 'w') as file:
            for idx, df in enumerate([Xtrain, Xtest]):
                file.write("Length of %s: %d\n"%(['Xtrain', 'Xtest'][idx], len(df)))
        if not os.path.exists(inFolder):
            os.makedirs(inFolder)
        if not os.path.exists(outFolder+"/performance"):
            os.makedirs(outFolder+"/performance")
            os.makedirs(outFolder+"/model")
        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder+"/performance/features_train.png",
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=False,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]])
    
    
        
        Xtrain = scale(Xtrain, scalerName= inFolder + "/myScaler.pkl" ,fit=True)
        Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
    

        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder +"/performance/features_train_scaled.png",
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=True,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]])
            
        HbbClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest, hp=hp, nReal=nReal, nMC=nMC, inFolder=inFolder, outFolder=outFolder)
    Xtrain, Xtest =         pd.read_parquet(inFolder +"/XTrain.parquet"),   pd.read_parquet(inFolder +"/XTest.parquet")
    YPredTrain, YPredTest = np.load(inFolder +"/YPredTrain.npy"),           np.load(inFolder +"/YPredTest.npy")
    Wtrain, Wtest =         np.load(inFolder +"/WTrain.npy"),               np.load(inFolder +"/WTest.npy")
    Ytrain, Ytest =         np.load(inFolder +"/YTrain.npy"),               np.load(inFolder +"/YTest.npy")
    doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder)