import shap
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import sys
import numpy as np
import pandas as pd
from models import getModelMultiClass
from plotsForNN import doPlotLoss, getShap, getShapNew
from functions import loadMultiParquet
from helpersForNN import scale, unscale, preprocessMultiClass
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
import logging

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
suffixWeight = "_flat"
modelName = "model%s.h5"%suffixWeight
def doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder, logging):
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
        logging.info("%s events in class %d"%(len(Ytest[maskClass]), classIdx))

        tpr, fpr = [], []
        tprTrain, fprTrain = [], []
        for t in thresholds:
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
    fig.savefig(outFolder +"/performance/roc%s.png"%suffixWeight, bbox_inches='tight')



# confusion matrix
    ax.clear()
    confusionM = np.ones((3, 3))
    for trueLabel in range(3):
        # true label
        maskClass = Ytest.iloc[:,trueLabel]>0.99
        den = Wtest[maskClass].sum()
        for predictedLabel in range(3):

            num = (Wtest[(YPredTest.idxmax(axis=1)==predictedLabel) & (maskClass)]).sum()
            confusionM[trueLabel, predictedLabel] = num/den
    logging.info("Confusion matrix : \n%s"%str(confusionM))
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
    fig.savefig(outFolder +"/performance/confusionMatrix%s.png"%suffixWeight, bbox_inches='tight')
    plt.close('all')

# want to see if the dijet mass for qcd is smoothly falling.
# take data events with high ggH score
    fig, ax = plt.subplots(1, 1)
    maskClass = Ytest.iloc[:,0]>0.99 # data events (QCD)
    Xtest=Xtest.reset_index(drop=True)
    bins=np.linspace(0, 450, 40)

    ax.set_title("BParking data with Z score > 0.5 ")
    ax.hist(Xtest[(maskClass) & (YPredTest.iloc[:,2]>0.5)].dijet_mass, bins=bins, weights=Xtest[(maskClass) & (YPredTest.iloc[:,2]>0.5)].sf)
    fig.savefig(outFolder +"/performance/dijetMass%s.png"%suffixWeight, bbox_inches='tight')
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
    fig.savefig(outFolder +"/performance/outputTrainTest%s.png"%suffixWeight, bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(1, 1)
    print("\n\n\n")
    Xtrain=Xtrain.reset_index(drop=True)
    bins=np.linspace(0, 2, 20)
    counts_Datatrain =  np.histogram(np.clip(Xtrain[Ytrain.iloc[:,0]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_ggHtrain =   np.histogram(np.clip(Xtrain[Ytrain.iloc[:,1]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Ztrain =     np.histogram(np.clip(Xtrain[Ytrain.iloc[:,2]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Datatest =   np.histogram(np.clip(Xtest[Ytest.iloc[:,0]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_ggHtest =    np.histogram(np.clip(Xtest[Ytest.iloc[:,1]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Ztest =      np.histogram(np.clip(Xtest[Ytest.iloc[:,2]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Datatrain = counts_Datatrain/np.sum(counts_Datatrain)
    counts_ggHtrain = counts_ggHtrain/np.sum(counts_ggHtrain)
    counts_Ztrain = counts_Ztrain/np.sum(counts_Ztrain)
    counts_Datatest = counts_Datatest/np.sum(counts_Datatest)
    counts_ggHtest = counts_ggHtest/np.sum(counts_ggHtest)
    counts_Ztest = counts_Ztest/np.sum(counts_Ztest)

    ax.hist(bins[:-1], bins=bins, weights=counts_Datatrain, histtype=u'step', linewidth=2 ,label='Data train %d'%(Xtrain[Ytrain.iloc[:,0]==1].sf.sum()))    
    ax.hist(bins[:-1], bins=bins, weights=counts_ggHtrain, histtype=u'step', linewidth=2 ,label='ggH train %d'%(Xtrain[Ytrain.iloc[:,1]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Ztrain, histtype=u'step', linewidth=2 ,label='Z train %d'%(Xtrain[Ytrain.iloc[:,2]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Datatest, histtype=u'step', linewidth=2 ,label='Data test %d'%(Xtest[Ytest.iloc[:,0]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_ggHtest, histtype=u'step', linewidth=2 ,label='ggH test %d'%(Xtest[Ytest.iloc[:,1]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Ztest, histtype=u'step', linewidth=2 ,label='Z test %d'%(Xtest[Ytest.iloc[:,2]==1].sf.sum()))
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(outFolder +"/performance/weightsDistribution%s.png"%suffixWeight)



    
    return

def getFeatures():
    featuresForTraining=[
       'jet1_pt',
       'jet1_eta', 'jet1_phi', 'jet1_mass',
        'jet1_nMuons', 'jet1_nElectrons',
        'jet1_btagDeepFlavB', #'jet1_area',
        'jet1_qgl',
       'jet2_pt',
       'jet2_eta', 'jet2_phi', 'jet2_mass',
       'jet2_nMuons', 'jet2_nElectrons',
       'jet2_btagDeepFlavB', #'jet2_area',
       'jet2_qgl',
       #'dijet_pt',
       #'dijet_eta', 'dijet_phi',
       #'dijet_mass,
       #'dijet_dR',
       #'dijet_dEta', 'dijet_dPhi',
       #'dijet_twist',# 'nJets',
       'nJets_20GeV',
       #'ht',
       #'muon_pt',
       #'muon_eta',
       'muon_dxySig',  
       #'muon_dzSig',
       'muon_IP3d',
       'muon_sIP3d',
       #'dijet_cs',
       'muon_pfRelIso03_all',
       'muon_tkIsoId',
       'muon_tightId',
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

def loadData(pTmin, pTmax, suffix, outFolder, hp, logging):
    featuresForTraining, columnsToRead = getFeatures()
    logging.info("Features for training")
    for feature in featuresForTraining:
        logging.info(">  %s"%feature)
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training"]
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf")

    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=True)
    
    #dfs[1]=dfs[1][(dfs[1].dijet_mass>75) & (dfs[1].dijet_mass<175)]
    dfs[0]=dfs[0][(dfs[0].dijet_mass<200)]
    dfs[1]=dfs[1][(dfs[1].dijet_mass<200)]
    dfs[2]=dfs[2][(dfs[2].dijet_mass<200)]
    #dfs[2]=dfs[2][(dfs[2].dijet_mass>40) & (dfs[2].dijet_mass<140)]
    #dfs[3]=dfs[3][(dfs[3].dijet_mass>40) & (dfs[3].dijet_mass<140)]
    #dfs[4]=dfs[4][(dfs[4].dijet_mass>40) & (dfs[4].dijet_mass<140)]
    #dfs[5]=dfs[5][(dfs[5].dijet_mass>40) & (dfs[5].dijet_mass<140)]
    #dfs[6]=dfs[6][(dfs[6].dijet_mass>40) & (dfs[6].dijet_mass<140)]    
    print("my max:", dfs[0].dijet_mass.max())
    
    nData, nHiggs, nZ = int(5*1e4), int(5*1e4) , int(5*1e4)
    dfs = preprocessMultiClass(dfs, pTmin, pTmax, suffix)

    for idx, df in enumerate(dfs):
        if idx==0:
            #print(len(dfs[idx]))
            
            #print(len(dfs[idx]))
            dfs[idx] = df.head(nData)
            #sys.exit()
        elif idx==1:
            dfs[idx] = df.head(nHiggs)
        else:
            #pass
            dfs[idx] = df.head(int(nZ/(len(paths)-2)))

    for idx, df in enumerate(dfs):
        logging.info("Length of df %d : %d\n"%(idx, len(df)))

    Y_0 = pd.DataFrame([np.ones(len(dfs[0])),  np.zeros(len(dfs[0])), np.zeros(len(dfs[0]))]).T
    Y_1 = pd.DataFrame([np.zeros(len(dfs[1])), np.ones(len(dfs[1])),  np.zeros(len(dfs[1]))]).T
    #ZJets
    Y_Z, W_Z = [],[]
    for i in range(2, len(paths)):
        Y = pd.DataFrame([np.zeros(len(dfs[i])), np.zeros(len(dfs[i])), np.ones(len(dfs[i]))]).T
        Y_Z.append(Y)

    # define a weights vector 1 for data 1 for hbb, xsection for the Z boson dataframes. then concat z bosons. divide every weights by the average of the weights
    W_0 = dfs[0].sf
    W_1 = dfs[1].sf
    #ZJets
    for i in range(2, len(paths)):
        W = ([5.261e+03, 1012, 114.2, 25.34, 12.99][i-2])/numEventsList[i]*dfs[i].sf
        #W = ([1012, 114.2, 25.34, 12.99][i-2])/numEventsList[i]*dfs[i].sf
        W_Z.append(W)
    
    W_ZBos = pd.concat(W_Z)
    Y_ZBos = pd.concat(Y_Z)
    df_ZBos = pd.concat(dfs[2:])

# Each sample has a weight equal to 1 no matter how many events are there
    rW_0 = W_0.copy()
    rW_1 = W_1.copy()
    rW_ZBos = W_ZBos.copy()
    
    W_0 = W_0/W_0.sum()
    W_1 = W_1/W_1.sum()
    W_ZBos = W_ZBos/W_ZBos.sum()

    bins_0 = np.linspace(0, 200, 40)
    bins_1 = np.linspace(75, 175, 20)
    bins_2 = np.linspace(40, 140, 20)
    counts_0 = np.histogram(dfs[0].dijet_mass, bins=bins_0, weights=rW_0)[0]
    counts_Z = np.histogram(df_ZBos.dijet_mass, bins=bins_2, weights=W_ZBos)[0]
    counts_H = np.histogram(dfs[1].dijet_mass, bins=bins_1, weights=W_1)[0]

    if suffixWeight=='_flat':
        rW_ZBos=(rW_ZBos)*(1/counts_Z[np.digitize(np.clip(df_ZBos.dijet_mass, bins_2[0], bins_2[-1]-0.0001), bins_2)-1])
        rW_0=pd.DataFrame(1/counts_0[np.digitize(np.clip(dfs[0].dijet_mass, bins_0[0], bins_0[-1]-0.0001), bins_0)-1])
        rW_1=pd.DataFrame(1/counts_H[np.digitize(np.clip(dfs[1].dijet_mass, bins_1[0], bins_1[-1]-0.0001), bins_1)-1])
    #rW_1=rW_1*(counts_0/counts_H)[np.digitize(np.clip(dfs[1].dijet_mass, bins[0], bins[-1]-0.0001), bins)-1]
    rW_0 = rW_0/rW_0.sum()
    rW_1 = rW_1/rW_1.sum()
    rW_ZBos = rW_ZBos/rW_ZBos.sum()
    plt.close('all')
    bins = np.linspace(0, 300, 35)
    fig, ax =plt.subplots(1, 1)
    ax.hist(df_ZBos.dijet_mass, bins=bins_0, weights=rW_ZBos, histtype=u'step', label='Z reweighted', linewidth=5)
    ax.hist(df_ZBos.dijet_mass, bins=bins_0, weights=W_ZBos, histtype=u'step', label='Z', linewidth=5)
    ax.hist(dfs[1].dijet_mass, bins=bins_0, weights=rW_1, histtype=u'step', label='H reweighted', linewidth=5)
    ax.hist(dfs[1].dijet_mass, bins=bins_0, weights=W_1, histtype=u'step', label='H', linewidth=5)
    ax.hist(dfs[0].dijet_mass, bins=bins_0, weights=W_0, label='Data', alpha=0.4)
    ax.hist(dfs[0].dijet_mass, bins=bins_0, weights=rW_0, label='Data reweighted', histtype=u'step', linewidth=5)
    ax.legend()
    ax.set_yscale('log')
    fig.savefig("/t3home/gcelotto/ggHbb/NN/output/multiClass/inclusive/dijetMass/mass_temp%s.png"%suffixWeight)

    
    

    Y_ZBos, W_ZBos, rW_ZBos, df_ZBos = shuffle(Y_ZBos, W_ZBos, rW_ZBos, df_ZBos)
    Y_ZBos.head(nZ), W_ZBos.head(nZ), rW_ZBos.head(nZ), df_ZBos.head(nZ)
    Y = np.concatenate((Y_0, Y_1, Y_ZBos))
    X = pd.concat((dfs[0], dfs[1], df_ZBos))
    rW = np.concatenate((rW_0, rW_1, rW_ZBos))
    W = np.concatenate((W_0, W_1, W_ZBos))
    
    X, Y, W, rW = shuffle(X, Y, W, rW, random_state=1999)
    Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest = train_test_split(X, Y, W, rW, test_size=hp['test_split'], random_state=1999)
    #Xtrain[Ytrain[:,0]==1] = Xtrain[(Ytrain[:,0]==1) & (Xtrain['dijet_mass']<80) | (Xtrain['dijet_mass']>100) | (Ytrain[:,1]==1) | (Ytrain[:,2]==1)]
    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)
    print(rW_0.sum(), rW_1.sum(), rW_ZBos.sum())
    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest



def HbbClassifier(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest, hp, nReal, nMC, inFolder, outFolder, logging):
    
    featuresForTraining, columnsToRead = getFeatures()
    
    for key, value in hp.items():
        logging.info(f"{key}: {value}")
    

# get the model, optimizer, compile it and fit to data
    model = getModelMultiClass(inputDim=len(featuresForTraining), nDense=hp['nDense'], nNodes=hp['nNodes'])
    optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") 
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), weighted_metrics=['accuracy'])
    callbacks = []
    earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(Xtrain[featuresForTraining], Ytrain, 
                    sample_weight = rWtrain,
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    callbacks=callbacks, shuffle=True)
    
    model.save(outFolder +"/model/"+modelName)

    model = load_model(outFolder +"/model/"+modelName)
    doPlotLoss(fit=fit, outName=outFolder +"/performance/loss%s.png"%suffixWeight, earlyStop=earlyStop, patience=hp['patienceES'])
    #getShapNew(Xtest.iloc[:500,:][featuresForTraining], model, outName = outFolder+'/performance/shap%s.png'%suffixWeight,
    #        class_names=['Data', 'ggH', 'ZJets'])
    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])

    Xtrain = unscale(Xtrain,    scalerName= inFolder + "/myScaler.pkl")
    Xtest = unscale(Xtest,      scalerName =  inFolder + "/myScaler.pkl")
    if (True):
        plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder +"/performance/features_train_unscaled%s.png"%suffixWeight,
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
    np.save(inFolder +"/rWTest.npy", rWtest)
    np.save(inFolder +"/rWTrain.npy", rWtrain)
    np.save(inFolder +"/YTest.npy", Ytest)
    np.save(inFolder +"/YTrain.npy", Ytrain)

if __name__ =="__main__":
    nReal, nMC = int(sys.argv[1]), int(sys.argv[2]), 
    doTrain = bool(int(sys.argv[3]))
    ptClass = int(sys.argv[4])


    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][ptClass]
    inFolder, outFolder = "/t3home/gcelotto/ggHbb/NN/input/multiclass/"+suffix, "/t3home/gcelotto/ggHbb/NN/output/multiClass/"+suffix
    if os.path.exists(outFolder+"/status.txt"):
        os.remove(outFolder+"/status.txt")
    logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s - %(levelname)s - %(message)s',
                    format='',
                    handlers=[
                        logging.FileHandler(outFolder+"/status.txt"),
                        logging.StreamHandler()
                    ])
    if doTrain:
        hp = {
            'epochs'            : 400,
            'patienceES'        : 30,
            'validation_split'  : 0.2,
            'test_split'        : 0.2,
            'learning_rate'     : 5e-4,
            'min_delta'         : 1e-7,
            'nNodes'            : [32, 16],
            }
        hp['nDense']=len(hp['nNodes'])
        assert len(hp['nNodes'])==hp['nDense']
        
        data = loadData(pTmin, pTmax, suffix, outFolder,  hp, logging)
        Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest = data


        # define suffix = inclusive or other
        if not os.path.exists(inFolder):
            os.makedirs(inFolder)
        if not os.path.exists(outFolder+"/performance"):
            os.makedirs(outFolder+"/performance")
            os.makedirs(outFolder+"/model")
        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder+"/performance/features_train%s.png"%suffixWeight,
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=False,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]],
                                    error=False)
    
    
        
        Xtrain = scale(Xtrain, scalerName= inFolder + "/myScaler.pkl" ,fit=True)
        Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
    

        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder +"/performance/features_train_scaled%s.png"%suffixWeight,
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=True,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]])
            
        HbbClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest,
                        rWtrain=rWtrain, rWtest=rWtest,
                         hp=hp, nReal=nReal, nMC=nMC, inFolder=inFolder, outFolder=outFolder, logging=logging)
    Xtrain, Xtest =         pd.read_parquet(inFolder +"/XTrain.parquet"),   pd.read_parquet(inFolder +"/XTest.parquet")
    YPredTrain, YPredTest = np.load(inFolder +"/YPredTrain.npy"),           np.load(inFolder +"/YPredTest.npy")
    Wtrain, Wtest =         np.load(inFolder +"/WTrain.npy"),               np.load(inFolder +"/WTest.npy")
    Ytrain, Ytest =         np.load(inFolder +"/YTrain.npy"),               np.load(inFolder +"/YTest.npy")
    doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder, logging)
    featuresForTraining, columnsToRead = getFeatures()
    model = load_model(outFolder +"/model/"+modelName)
    #from tensorflow.compat.v1.keras.backend import get_session
#
    #tensorflow.compat.v1.disable_v2_behavior()
    #getShapNew(Xtest.iloc[:500,:][featuresForTraining], model, outName = outFolder+'/performance/shap%s.png'%suffixWeight,
    #        class_names=['Data', 'ggH', 'ZJets'])