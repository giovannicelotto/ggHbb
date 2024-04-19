import shap
import tensorflow as tf
import random
#from tensorflow.compat.v1.keras.backend import get_session
#tensorflow.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import sys
from models import getModel
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from utilsForPlot import loadParquet, loadDask, getXSectionBR
from plotFeatures import plotNormalizedFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from plotsForNN import doPlotLoss, roc, WorkingPoint, massSpectrum, NNoutputs, getShap
from sklearn import preprocessing
import pickle

from keras import backend as K

def scale(data, fit=False):
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            print("feature: %s min: %.1f max: %.1f"%(colName, data[colName].min(), data[colName].max()))
            data[colName] = np.log(1+data[colName])
            print("log done for %s"%colName)
    if fit:
        scaler  = preprocessing.StandardScaler().fit(data)
        scaled_array = scaler.transform(data)
    else:
        with open("/t3home/gcelotto/ggHbb/NN/input/myScaler.pkl", 'rb') as file:
            scalers = pickle.load(file)
            scaler = scalers['scaler']
            scaled_array = scaler.transform(data)
            
    data = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

    
    scalers = {
'type'  : 'standard',
'scaler': scaler,
}
    with open("/t3home/gcelotto/ggHbb/NN/input/myScaler.pkl", 'wb') as file:
        pickle.dump(scalers, file)
    return data

def unscale(data):
    with open("/t3home/gcelotto/ggHbb/NN/input/myScaler.pkl", 'rb') as file:
        scalers = pickle.load(file)
        scaler = scalers['scaler']
        scaled_array = scaler.inverse_transform(data)
        data = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName):
            data[colName] = np.exp(data[colName])
    return data

def custom_loss(y_true, y_pred):
    # Compute custom loss batch per batch
    s = 700*K.sum(y_pred * y_true)
    # Compute b = sum_batch(y_pred * (1 - y_true))
    b = 40000000*K.sum(y_pred * (1 - y_true))
    # Compute the loss using the provided formula
    loss = -s / (s + b + K.epsilon())  # Adding epsilon to avoid division by zero
    return loss

def writeFeatures():
    featuresForTraining=[
       'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
       'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
       'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons',
       'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
       'dijet_pt', 'dijet_eta', 'dijet_phi',
       'dijet_dR',
       'dijet_dEta', 'dijet_dPhi', 'dijet_twist', 'nJets',
       'nJets_20GeV',
       'ht', 'muon_pt', 'muon_eta', 'muon_dxySig', 
       'muon_dzSig', 'muon_IP3d',
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
       'dijet_cs', 'sf']
    np.save("/t3home/gcelotto/ggHbb/NN/input/featuresForTraining.npy", featuresForTraining)
    np.save("/t3home/gcelotto/ggHbb/NN/input/columnsToRead.npy", columnsToRead)
    return

def readFeatures():
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/NN/input/featuresForTraining.npy")
    columnsToRead = np.load("/t3home/gcelotto/ggHbb/NN/input/columnsToRead.npy")
    return featuresForTraining, columnsToRead
def preprocess(signal, bkg):
    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    signal, bkg      = signal[(signal.jet1_pt>20) & (signal.jet2_pt>20)], bkg[(bkg.jet1_pt>20) & (bkg.jet2_pt>20)]
    signal, bkg      = signal[(signal.jet1_eta<2.5) & (signal.jet1_eta>-2.5)], bkg[(bkg.jet1_eta<2.5) & (bkg.jet1_eta>-2.5)]
    signal, bkg      = signal[(signal.jet2_eta<2.5) & (signal.jet2_eta>-2.5)], bkg[(bkg.jet2_eta<2.5) & (bkg.jet2_eta>-2.5)]
    signal, bkg      = signal[(signal.jet2_mass>0)], bkg[(bkg.jet2_mass>0)]
    signal, bkg      = signal[(signal.jet1_mass>0)], bkg[(bkg.jet1_mass>0)]
    signal, bkg      = signal[(signal.dijet_mass>125-2.5*17) & (signal.dijet_mass<125+2.5*17)], bkg[(bkg.dijet_mass>125-2.5*17) & (bkg.dijet_mass<125+2.5*17)]
    #signal, bkg      = signal[(signal.dijet_pt<100) ], bkg[(bkg.dijet_pt<100)]
    print("Nan values : %d (S)   %d (B)"%(signal.isna().sum().sum(), bkg.isna().sum().sum()))
    print("Filling jet qgl with 0.5")
    signal.jet1_qgl = signal.jet1_qgl.fillna(0.5)
    bkg.jet1_qgl = bkg.jet1_qgl.fillna(0.5)
    signal.jet2_qgl = signal.jet2_qgl.fillna(0.5)
    bkg.jet2_qgl = bkg.jet2_qgl.fillna(0.5)
    assert signal.isna().sum().sum()==0
    assert bkg.isna().sum().sum()==0
    print("No Nan values after filling")

    return signal, bkg

def HbbClassifier(doTrain, nRealDataFiles):
    writeFeatures()
    featuresForTraining, columnsToRead = readFeatures()
    
    hp = {
        'epochs'            : 1000,
        'patienceES'        : 30,
        'validation_split'  : 0.2,
        'learning_rate'     : 5*1e-4
        }
    if doTrain:
        realDataPath_train =    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training"
        signalPath_train =      "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training"
        signal_train, realData_train = loadParquet(signalPath_train, realDataPath_train, nSignalFiles=-1, nRealDataFiles=-1, 
                                                   columns=columnsToRead)
        
        totalDataFlat_train = len(realData_train)
        
        np.save("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_train.npy", totalDataFlat_train)
        
        
        #signalRegion
        print("Performing the cut in pt and eta")
        print("Before preprocessing", len(signal_train), len(realData_train))
        signal_train, realData_train = preprocess(signal_train, realData_train)
        print("After preprocessing", len(signal_train), len(realData_train))

        print("%d events for signal fiducial\n%d events for background fiducial"%(len(signal_train), len(realData_train)))      
        
    # calcola la significance come control check if want
        if (True):
            totalDataFlat_train = np.load("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_train.npy")
            lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
            numEventsTotalTest = 0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
            numEventsTotalTrain = df[(df.process=='GluGluHToBB') & (df.fileNumber<=45)].numEventsTotal.sum()
            x1_sb, x2_sb  = 123.11 - 2*17, 123.11 + 2*17
            maskSignal =    (signal_train.dijet_mass>x1_sb) & (signal_train.dijet_mass<x2_sb)
            maskData =      (realData_train.dijet_mass>x1_sb) & (realData_train.dijet_mass<x2_sb)
            S = np.sum(signal_train.sf[maskSignal])*lumiPerEvent*totalDataFlat_train/numEventsTotalTrain*getXSectionBR()*1000
            B = np.sum(maskData)
            print("Signal 2sigma", S)
            print("Data 2sigma", B)
            print("Sig", S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat_train)))

    # create the label for training
        signal_train = signal_train.head(200000)
        realData_train = realData_train.head(200000)
        y_signal_train = pd.DataFrame(np.ones(len(signal_train)), columns=['label'])
        y_realData_train = pd.DataFrame(np.zeros(len(realData_train)), columns=['label'])
        Xtrain = pd.concat([signal_train, realData_train],       ignore_index=True)
        Ytrain = pd.concat([y_signal_train, y_realData_train],               ignore_index=True)
        from sklearn.utils import shuffle
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=1999)
        
        #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=hp['test_split'], random_state=1999, shuffle=True)
        #Xtrain = Xtrain[(Ytrain==1) | ((Ytrain==0) & ((Xtrain.dijet_mass>175) | (Xtrain.dijet_mass<75)))]
        #Ytrain = Ytrain[(Ytrain==1) | ((Ytrain==0) & ((Xtrain.dijet_mass>175) | (Xtrain.dijet_mass<75)))]

        #Xtrain.drop(columns=['dijet_mass', 'jet1_eta', 'jet2_eta'])
        #Xtest.drop(columns=['dijet_mass', 'jet1_eta', 'jet2_eta'])
        
        Xtrain.to_parquet("/t3home/gcelotto/ggHbb/NN/input/Xtrain.parquet")
        Ytrain.to_parquet("/t3home/gcelotto/ggHbb/NN/input/Ytrain.parquet")
        

# log of pt and scaling for train
        Xtrain=scale(Xtrain, fit=True)
            
        
        model = getModel(len(Xtrain[featuresForTraining].columns))
        losses = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy()]
        optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        callbacks = []
        earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
        callbacks.append(earlyStop)
        
        fit = model.fit(Xtrain[featuresForTraining], Ytrain,
                        epochs=hp['epochs'], callbacks=callbacks, validation_split=hp['validation_split'], shuffle=True)
        model.save("/t3home/gcelotto/ggHbb/outputs/model_inclusive.keras")
        doPlotLoss(fit=fit, outName="/t3home/gcelotto/ggHbb/NN/output/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])
    

    # load the train for plots
    Xtrain      = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Xtrain.parquet")
    Ytrain      = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Ytrain.parquet")['label']
    
    

    #for colName in Xtrain.columns:
    #    if ("_pt" in colName) | ("_mass" in colName):
    #        Xtrain[colName] = np.log(1+Xtrain[colName])
    #with open("/t3home/gcelotto/ggHbb/NN/input/myScaler.pkl", 'rb') as file:
    #    scalers = pickle.load(file)
    #    scaler = scalers['scaler']
    #    scaled_array = scaler.transform(Xtrain)
    #    Xtrain = pd.DataFrame(scaled_array, columns=Xtrain.columns, index=Xtrain.index)



    # load the test
    realDataPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"
    signalPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others"
    import glob, re, os
    fileNames = glob.glob(signalPath_test+"/*.parquet")
    numEventsTotalTest = 0
    df = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
    for fileName in fileNames:
        filename = os.path.splitext(os.path.basename(fileName))[0]
        process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
        fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
        numEventsTotalTest = numEventsTotalTest + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]

    signal_test, realData_test = loadParquet(   signalPath_test, realDataPath_test, nSignalFiles=-1, nRealDataFiles=nRealDataFiles, 
                                                columns=columnsToRead)
    
    totalDataFlat_test = len(realData_test)
    np.save("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_test.npy", totalDataFlat_test)
    
    # signalRegion
    print("Performing the cut in pt and eta for the testing")
    signal_test, realData_test      = preprocess(signal_test, realData_test)

    y_signal_test = pd.DataFrame(np.ones(len(signal_test)), columns=['label'])
    y_realData_test = pd.DataFrame(np.zeros(len(realData_test)), columns=['label'])
    Xtest = pd.concat([signal_test, realData_test],         ignore_index=True)
    Ytest = pd.concat([y_signal_test, y_realData_test],     ignore_index=True)
    SFtest = Xtest.sf   
    Ytest = Ytest.reset_index(drop=True)
    Xtest = Xtest.reset_index(drop=True)
    SFtest = SFtest.reset_index(drop=True)
    Xtest.to_parquet("/t3home/gcelotto/ggHbb/NN/input/Xtest.parquet")
    Ytest.to_parquet("/t3home/gcelotto/ggHbb/NN/input/Ytest.parquet")
    
    plotNormalizedFeatures(data =   [Xtrain[Ytrain==1], Xtrain[Ytrain==0], Xtest[Ytest.label==1],   Xtest[Ytest.label==0]],
                                    outFile = "/t3home/gcelotto/ggHbb/NN/output/features_train.png",
                                    legendLabels = ['Signal train ', 'BParking train', 'Signal test ', 'BParking test'] , colors = ['blue', 'red', 'blue', 'red'],
                                    histtypes=[u'step', u'step', 'bar', 'bar'],
                                    alphas=[1, 1, 0.4, 0.4],
                                    figsize=(20, 30))

    Xtrain = scale(Xtrain, fit=False)
    Xtest  = scale(Xtest, fit=False)
    #for colName in Xtest.columns:
    #        if ("_pt" in colName) | ("_mass" in colName):
    #            Xtest[colName] = np.log(1+Xtest[colName])
    #            print("log done for %s"%colName)
    #scaled_array = scaler.transform(Xtest)
    #Xtest = pd.DataFrame(scaled_array, columns=Xtest.columns, index=Xtest.index)
    
    plotNormalizedFeatures(data =   [Xtrain[Ytrain==1], Xtrain[Ytrain==0], Xtest[Ytest.label==1],   Xtest[Ytest.label==0]],
                                    outFile = "/t3home/gcelotto/ggHbb/NN/output/features_train_scaled.png",
                                    legendLabels = ['Signal train ', 'BParking train', 'Signal test ', 'BParking test'] , colors = ['blue', 'red', 'blue', 'red'],
                                    histtypes=[u'step', u'step', 'bar', 'bar'],
                                    alphas=[1, 1, 0.4, 0.4],
                                    autobins=True)
    assert(len(Xtrain)==len(Ytrain))

    model = load_model("/t3home/gcelotto/ggHbb/outputs/model_inclusive.keras")
    #model = load_model(modelFile, custom_objects={ 'loss': penalized_loss(noise) })
    y_predict = model.predict(Xtest[featuresForTraining])
    yTrain_predict = model.predict(Xtrain[featuresForTraining])
    np.save("/t3home/gcelotto/ggHbb/NN/input/yTest_predict.npy", y_predict)
    np.save("/t3home/gcelotto/ggHbb/NN/input/yTrain_predict.npy", yTrain_predict)
    thresholds = np.linspace(0, 0.99, 100)
    thresholds = np.concatenate((thresholds, np.linspace(0.99, 1., 1000)))
    
    signal_predictions = y_predict[Ytest==1]
    realData_predictions = y_predict[Ytest==0]
    
    signalTrain_predictions = yTrain_predict[Ytrain==1]
    realDataTrain_predictions = yTrain_predict[Ytrain==0]
    print("Signal train predictions", len(signalTrain_predictions))
    print("Signal test predictions", len(signal_predictions))
    print("Data train predictions", len(realDataTrain_predictions))
    print("Data test predictions", len(realData_predictions))
    y_predict = pd.DataFrame(y_predict.flatten(), columns=['label'])
    y_predict = y_predict.reset_index(drop=True)
    
    getShap(Xtest[:1000][featuresForTraining], model, outName = '/t3home/gcelotto/ggHbb/NN/output/shap.png')

    Xtrain = unscale(Xtrain)
    Xtest = unscale(Xtest)

    
    #with open("/t3home/gcelotto/ggHbb/NN/input/myScaler.pkl", 'rb') as file:
    #    scalers = pickle.load(file)
    #    scaler = scalers['scaler']
    #    scaled_array = scaler.inverse_transform(Xtest)
    #    Xtest = pd.DataFrame(scaled_array, columns=Xtest.columns, index=Xtest.index)
#
    #    scaled_array = scaler.inverse_transform(Xtrain)
    #    Xtrain = pd.DataFrame(scaled_array, columns=Xtrain.columns, index=Xtrain.index)
#
#
    #for colName in Xtrain.columns:
    #    if ("_pt" in colName) | ("_mass" in colName):
    #        Xtest[colName] = np.exp(Xtest[colName])
    #        Xtrain[colName] = np.exp(Xtrain[colName])
                
    roc(thresholds, signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName="/t3home/gcelotto/ggHbb/NN/output/nn_roc.png")
    maskSignal = (Xtest[Ytest.label==1].dijet_mass>123.11-2*17) & (Xtest[Ytest.label==1].dijet_mass<123.11+2*17)
    maskRealData = (Xtest[Ytest.label==0].dijet_mass>123.11-2*17) & (Xtest[Ytest.label==0].dijet_mass<123.11+2*17)
    WorkingPoint(signal_predictions[maskSignal], realData_predictions[maskRealData], SFtest[Ytest.label==1][maskSignal], outName="/t3home/gcelotto/ggHbb/NN/output/cut_on_NN_output.png")
    NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName = "/t3home/gcelotto/ggHbb/NN/output/nn_outputs.png")    
    assert len(SFtest)==len(Xtest)
    massSpectrum(Xtest, Ytest, y_predict, SFtest, hp, numEventsTotalTest, outName="/t3home/gcelotto/ggHbb/NN/output/dijetMass_afterCut.png")

    return

if __name__ =="__main__":
    doTrain = bool(int(sys.argv[1])) if len(sys.argv)>1 else False
    nRealDataFiles = int(sys.argv[2]) if len(sys.argv)>2 else False
    print("doTrain", doTrain)
    HbbClassifier(doTrain=doTrain, nRealDataFiles=nRealDataFiles)