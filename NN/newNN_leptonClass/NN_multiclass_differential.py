import shap
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys
import numpy as np
import pandas as pd
sys.path.append("/t3home/gcelotto/ggHbb/NN")
from models import getModelMultiClass
from plotsForNN import doPlotLoss, getShap, getShapNew
from helpersForNN import scale, unscale, preprocessMultiClass
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os
import matplotlib.pyplot as plt
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
import matplotlib.pyplot as plt
import mplhep as hep
import logging
hep.style.use("CMS")
from doPlots import doPlots
from loadData import loadData
from getFeatures import getFeatures


'''
nReal   = number of Files used for realData training, testing (10 recommended). All the datasets are cut after N events
nMC     = number of Files used for    MC    training, testing (10 recommended). All the datasets are cut after N events
doTrain = bool. Yes: Do training and do performance plots. False: Load X, Y, Y_Pred, W  and redo the plots.
ptClass = ptClass in dijetPT:
            0       inclusive
            1       0-30 
            2       30-100
            3       100-inf
leptonClass = leptonClass according to the leptons in the second jet:
            0       inclusive
            1       second jet has a trig muon 
            2       second jet has a muon but non triggering
            3       second jet does not have leptons

'''
modelName = "model.h5"


def HbbClassifier(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest, leptonClass, hp, inFolder, outFolder, logging):
    
    featuresForTraining, columnsToRead = getFeatures(leptonClass=leptonClass)
    
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
                    batch_size=hp['batch_size'],
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    callbacks=callbacks, shuffle=True)
    
    model.save(outFolder +"/model/"+modelName)
    model = load_model(outFolder +"/model/"+modelName)
    doPlotLoss(fit=fit, outName=outFolder +"/performance/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])
    #getShapNew(Xtest.iloc[:500,:][featuresForTraining], model, outName = outFolder+'/performance/shap.png',
    #        class_names=['Data', 'ggH', 'ZJets'])
    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])

    Xtrain = unscale(Xtrain,    scalerName= inFolder + "/myScaler.pkl")
    Xtest = unscale(Xtest,      scalerName =  inFolder + "/myScaler.pkl")
    #if (False):
    #    plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
    #                                outFile = outFolder +"/performance/features_train_unscaled.png",
    #                                legendLabels = ['Train 0 ', 'Train 1', 'Train 2 ', 'Test 0 ', 'Test 1', 'Test 2 ',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
    #                                histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
    #                                alphas=[1, 1, 1, 0.4, 0.4, 0.4],
    #                                figsize=(20, 30),
    #                                autobins=True,
    #                                weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]])

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
    leptonClass = int(sys.argv[5])
    print("nReal : %d"%nReal)
    print("nMC : %d"%nMC)
    print("doTrain : %d"%doTrain)
    print("ptClass : %d"%ptClass)
    print("leptonClass : %d"%leptonClass)

    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][ptClass]
    suffix = suffix + "_lep%d"%leptonClass


    inFolder, outFolder = "/t3home/gcelotto/ggHbb/NN/input/multiclass/"+suffix, "/t3home/gcelotto/ggHbb/NN/output/multiClass/"+suffix
    if not os.path.exists(inFolder):
        os.makedirs(inFolder)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
        os.makedirs(outFolder+'/performance')
        os.makedirs(outFolder+'/model')
    if os.path.exists(outFolder+"/status.txt"):
        os.remove(outFolder+"/status.txt")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s - %(levelname)s - %(message)s',
                    format='',
                    handlers=[
                        logging.FileHandler(outFolder+"/status.txt"),
                        logging.StreamHandler()
                    ])
    featuresForTraining, columnsToRead = getFeatures(leptonClass)


    if doTrain:
        np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)
        hp = {
            'epochs'            : 900,
            'batch_size'        : 4096,
            'validbatch_size'   : 4096,
            'patienceES'        : 30,
            'validation_split'  : 0.2,
            'test_split'        : 0.2,
            'learning_rate'     : 5*1e-5,
            'nNodes'            : [64, 32, 16],
            }
        hp['nDense']=len(hp['nNodes'])
        assert len(hp['nNodes'])==hp['nDense']
        data = loadData(featuresForTraining, columnsToRead, leptonClass, nReal, nMC, pTmin, pTmax, suffix, outFolder, hp, logging)
        Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest = data
        
        # define suffix = inclusive or other
        if not os.path.exists(inFolder):
            os.makedirs(inFolder)
        if not os.path.exists(outFolder+"/performance"):
            os.makedirs(outFolder+"/performance")
            os.makedirs(outFolder+"/model")
        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1][featuresForTraining], Xtrain[Ytrain[:,1]==1][featuresForTraining], Xtrain[Ytrain[:,2]==1][featuresForTraining], Xtest[Ytest[:,0]==1][featuresForTraining],   Xtest[Ytest[:,1]==1][featuresForTraining], Xtest[Ytest[:,2]==1][featuresForTraining]],
                                    outFile = outFolder+"/performance/features4Train_train.png",
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=False,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]],
                                    error=False)
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder+"/performance/features_train.png",
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=False,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]],
                                    error=False)
    
    
        

        Xtrain = scale(Xtrain, scalerName= inFolder + "/myScaler.pkl" ,fit=True)
        Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
        for f in Xtrain.columns:
            print(f, " has nan ", Xtrain[f].isna().any())
    

        if True:
            plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
                                    outFile = outFolder +"/performance/features_train_scaled.png",
                                    legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
                                    histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
                                    alphas=[1, 1, 1, 0.4, 0.4, 0.4],
                                    figsize=(20, 30),
                                    autobins=True,
                                    weights=[Wtrain[Ytrain[:,0]==1], Wtrain[Ytrain[:,1]==1], Wtrain[Ytrain[:,2]==1], Wtest[Ytest[:,0]==1],   Wtest[Ytest[:,1]==1], Wtest[Ytest[:,2]==1]],
                                    error=False)
            
        HbbClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest,
                        rWtrain=rWtrain, rWtest=rWtest, leptonClass=leptonClass, hp=hp, inFolder=inFolder, outFolder=outFolder, logging=logging)
    featuresForTraining = np.load(outFolder+"/featuresForTraining.npy")


    Xtrain, Xtest =         pd.read_parquet(inFolder +"/XTrain.parquet"),   pd.read_parquet(inFolder +"/XTest.parquet")
    YPredTrain, YPredTest = np.load(inFolder +"/YPredTrain.npy"),           np.load(inFolder +"/YPredTest.npy")
    Wtrain, Wtest =         np.load(inFolder +"/WTrain.npy"),               np.load(inFolder +"/WTest.npy")
    rWtrain, rWtest =         np.load(inFolder +"/rWTrain.npy"),               np.load(inFolder +"/rWTest.npy")
    Ytrain, Ytest =         np.load(inFolder +"/YTrain.npy"),               np.load(inFolder +"/YTest.npy")
        ##plotNormalizedFeatures(data =   [Xtrain[Ytrain[:,0]==1], Xtrain[Ytrain[:,1]==1], Xtrain[Ytrain[:,2]==1], Xtest[Ytest[:,0]==1],   Xtest[Ytest[:,1]==1], Xtest[Ytest[:,2]==1]],
        ##                                outFile = outFolder +"/performance/features_train_reWeighted.png",
        ##                                legendLabels = ['Data Train', 'ggH Train', 'ZJets Train', 'Data Test ', 'ggH Test', 'ZJets Tets',] , colors = ['blue', 'red', 'green', 'blue', 'red', 'green'],
        ##                                histtypes=[u'step', u'step', u'step', 'bar', 'bar', 'bar'],
        ##                                alphas=[1, 1, 1, 0.4, 0.4, 0.4],
        ##                                figsize=(20, 30),
        ##                                autobins=True,
        ##                                weights=[rWtrain[Ytrain[:,0]==1], rWtrain[Ytrain[:,1]==1], rWtrain[Ytrain[:,2]==1], rWtest[Ytest[:,0]==1],   rWtest[Ytest[:,1]==1], rWtest[Ytest[:,2]==1]],
        ##                                error=False)
    doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder, logging)
    model = load_model(outFolder +"/model/"+modelName)
    Xtrain = scale(Xtrain, scalerName= inFolder + "/myScaler.pkl" ,fit=True)
    Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
    getShapNew(Xtest.iloc[:3500,:][featuresForTraining], model, outName = outFolder+'/performance/shap.png',
        nFeatures=10,class_names=['Data', 'ggH', 'ZJets'])