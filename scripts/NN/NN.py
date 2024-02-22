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
import glob
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from utilsForPlot import loadParquet, loadDask, getXSectionBR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from plotsForNN import doPlotLoss, roc, WorkingPoint, massSpectrum, NNoutputs, getShap


def writeFeatures():
    featuresForTraining=['jet1_eta', 'jet1_btagDeepFlavB', 'jet1_qgl',
                         'jet2_eta', 'jet2_btagDeepFlavB', 'jet2_qgl',
                         'dijet_eta',
                         'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_twist',
                         'muon_pt',  'nJets',     'ht',   'muon_pfRelIso03_all']
    columnsToRead = ['jet1_pt', 'jet1_eta', 'jet1_btagDeepFlavB',   'jet1_qgl',
                    'jet2_pt', 'jet2_eta', 'jet2_btagDeepFlavB',   'jet2_qgl',
                    'dijet_pt', 'dijet_eta',            'dijet_mass',
                    'dijet_dR', 'dijet_dEta',           'dijet_dPhi',           'dijet_twist',
                    'muon_pt',  'nJets',     'ht',           'muon_pfRelIso03_all',  'sf']
    np.save("/t3home/gcelotto/ggHbb/scripts/NN/input/featuresForTraining.npy", featuresForTraining)
    np.save("/t3home/gcelotto/ggHbb/scripts/NN/input/columnsToRead.npy", columnsToRead)
def readFeatures():
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/scripts/NN/input/featuresForTraining.npy")
    columnsToRead = np.load("/t3home/gcelotto/ggHbb/scripts/NN/input/columnsToRead.npy")
    return featuresForTraining, columnsToRead
def preprocess(signal, bkg):
    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    signal, bkg      = signal[(signal.jet1_pt>20) & (signal.jet2_pt>20)], bkg[(bkg.jet1_pt>20) & (bkg.jet2_pt>20)]
    signal, bkg      = signal[(signal.jet1_eta<2.5) & (signal.jet1_eta>-2.5)], bkg[(bkg.jet1_eta<2.5) & (bkg.jet1_eta>-2.5)]
    signal, bkg      = signal[(signal.jet2_eta<2.5) & (signal.jet2_eta>-2.5)], bkg[(bkg.jet2_eta<2.5) & (bkg.jet2_eta>-2.5)]
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
        'learning_rate'     : 1e-5
        }
    if doTrain:
        realDataPath_train = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training"
        signalPath_train = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training"
        signal_train, realData_train = loadParquet(signalPath_train, realDataPath_train, nSignalFiles=-1, nRealDataFiles=-1, 
                                       columns=columnsToRead)
        
        totalDataFlat_train = len(realData_train)
        #nSignalTrainFilesUsed = len(glob.glob(signalPath_train+"*.parquet"))
        np.save("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_train.npy", totalDataFlat_train)
        
        #signalRegion
        print("Performing the cut in pt and eta")
        print("Before preprocessing", len(signal_train), len(realData_train))
        signal_train, realData_train = preprocess(signal_train, realData_train)
        print("After preprocessing", len(signal_train), len(realData_train))

        print("%d events for signal fiducial\n%d events for background fiducial"%(len(signal_train), len(realData_train)))      
        
        
    # calcola la significance come control check if want
        if (False):
            totalDataFlat_train = np.load("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_train.npy")
            lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
            N_SignalMini = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_mini.npy")*nSignalTrainFilesUsed/240  # mini per produrre 45
            x1_sb, x2_sb  = 123.11 - 2*17, 123.11 + 2*17
            maskSignal =    (signal_train.dijet_mass>x1_sb) & (signal_train.dijet_mass<x2_sb)
            maskData =      (realData_train.dijet_mass>x1_sb) & (realData_train.dijet_mass<x2_sb)
            S = np.sum(signal_train.sf[maskSignal])*lumiPerEvent*totalDataFlat_train/N_SignalMini*getXSectionBR()*1000
            B = np.sum(maskData)
            print("Signal 2sigma", S)
            print("Data 2sigma", B)
            print("Sig", S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat_train)))

    # create the label for training
        signal_train = signal_train.head(100000)
        realData_train = realData_train.head(100000)
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
        
        Xtrain.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain.parquet")
        Ytrain.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain.parquet")
        
        model = Sequential()
        model.add(tf.keras.layers.Input(shape = len(Xtrain[featuresForTraining].columns))) 
        model.add(Dense(units=32, activation='relu', kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
        model.add(Dense(units=16, activation='relu', kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
        model.add(Dense(units=8, activation='relu', kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
        optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        callbacks=[]
        earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
        callbacks.append(earlyStop)
        
        fit = model.fit(Xtrain[featuresForTraining], Ytrain,
                        epochs=hp['epochs'], callbacks=callbacks, validation_split=hp['validation_split'], shuffle=True)
        model.save("/t3home/gcelotto/ggHbb/outputs/model_inclusive.h5")
        doPlotLoss(fit=fit, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])
    

    # load the train for plots
    Xtrain      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain.parquet")
    Ytrain      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain.parquet")['label']
    # load the test
    realDataPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"
    signalPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others"
    signal_test, realData_test = loadParquet(signalPath_test, realDataPath_test, nSignalFiles=-1, nRealDataFiles=nRealDataFiles, 
                                    columns=columnsToRead)
    
    totalDataFlat_test = len(realData_test)
    np.save("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_test.npy", totalDataFlat_test)
    
    # signalRegion
    print("Performing the cut in pt and eta for the testing")
    signal_test, realData_test      = signal_test[(signal_test.jet1_pt>20) & (signal_test.jet2_pt>20)], realData_test[(realData_test.jet1_pt>20) & (realData_test.jet2_pt>20)]
    signal_test, realData_test      = signal_test[(signal_test.jet1_eta<2.5) & (signal_test.jet1_eta>-2.5)], realData_test[(realData_test.jet1_eta<2.5) & (realData_test.jet1_eta>-2.5)]
    signal_test, realData_test      = signal_test[(signal_test.jet2_eta<2.5) & (signal_test.jet2_eta>-2.5)], realData_test[(realData_test.jet2_eta<2.5) & (realData_test.jet2_eta>-2.5)]

    # emulate the cut based analysis:
    #mask1 = signal_test.dijet_pt<22
    #mask2 = (signal_test.dijet_pt > 22) & signal_test.dijet_pt < 40
    #mask3 = (signal_test.dijet_pt>40) & (signal_test.dijet_pt<69.5)
    #mask4 = (signal_test.dijet_pt>69.5)
    #cut1 = (signal_test.jet2_btagDeepFlavB>0.2) &  (signal_test.jet1_btagDeepFlavB>0.2) & (signal_test.jet1_qgl>0.25) & (signal_test.jet2_qgl>0.25) & (signal_test.dijet_twist>0.8) & (signal_test.dijet_dPhi>2.6) & (signal_test.dijet_dR<4)
    #cut2 = (signal_test.jet2_btagDeepFlavB>0.25) &  (signal_test.jet1_btagDeepFlavB>0.25) & (signal_test.jet1_qgl>0.2) & (signal_test.jet2_qgl>0.2) & (signal_test.dijet_dPhi>2.3) & (signal_test.dijet_dR<4) & (signal_test.dijet_dEta<2.7)
    #cut3 = (signal_test.jet2_btagDeepFlavB>0.3) &  (signal_test.jet1_btagDeepFlavB>0.2) & (signal_test.jet1_qgl>0.2) & (signal_test.jet2_qgl>0.2) & (signal_test.dijet_dPhi>2.) & (signal_test.dijet_dR<3.5) & (signal_test.dijet_twist>0.5) & (signal_test.jet1_pt<120) & (signal_test.jet2_pt<80)
    #cut4 = (signal_test.jet2_btagDeepFlavB>0.2) &  (signal_test.jet1_btagDeepFlavB>0.2) & (signal_test.jet1_qgl>0.2) & (signal_test.jet2_qgl>0.2) & (signal_test.dijet_dPhi>0.6) & (signal_test.dijet_dEta<2.8) & (signal_test.dijet_twist>0.2) & (signal_test.ht>200)
    #signal_test = signal_test[(mask1 & cut1) | (mask2&cut2) | (mask3 & cut3) | (mask4&cut4)]
    #mask1 = realData_test.dijet_pt<22
    #mask2 = (realData_test.dijet_pt > 22) & realData_test.dijet_pt < 40
    #mask3 = (realData_test.dijet_pt>40) & (realData_test.dijet_pt<69.5)
    #mask4 = (realData_test.dijet_pt>69.5)
    #cut1 = (realData_test.jet2_btagDeepFlavB>0.2) &  (realData_test.jet1_btagDeepFlavB>0.2) & (realData_test.jet1_qgl>0.25) & (realData_test.jet2_qgl>0.25) & (realData_test.dijet_twist>0.8) & (realData_test.dijet_dPhi>2.6) & (realData_test.dijet_dR<4)
    #cut2 = (realData_test.jet2_btagDeepFlavB>0.25) &  (realData_test.jet1_btagDeepFlavB>0.25) & (realData_test.jet1_qgl>0.2) & (realData_test.jet2_qgl>0.2) & (realData_test.dijet_dPhi>2.3) & (realData_test.dijet_dR<4) & (realData_test.dijet_dEta<2.7)
    #cut3 = (realData_test.jet2_btagDeepFlavB>0.3) &  (realData_test.jet1_btagDeepFlavB>0.2) & (realData_test.jet1_qgl>0.2) & (realData_test.jet2_qgl>0.2) & (realData_test.dijet_dPhi>2.) & (realData_test.dijet_dR<3.5) & (realData_test.dijet_twist>0.5) & (realData_test.jet1_pt<120) & (realData_test.jet2_pt<80)
    #cut4 = (realData_test.jet2_btagDeepFlavB>0.2) &  (realData_test.jet1_btagDeepFlavB>0.2) & (realData_test.jet1_qgl>0.2) & (realData_test.jet2_qgl>0.2) & (realData_test.dijet_dPhi>0.6) & (realData_test.dijet_dEta<2.8) & (realData_test.dijet_twist>0.2) & (realData_test.ht>200)
    #realData_test = realData_test[(mask1 & cut1) | (mask2&cut2) | (mask3 & cut3) | (mask4&cut4)]

    y_signal_test = pd.DataFrame(np.ones(len(signal_test)), columns=['label'])
    y_realData_test = pd.DataFrame(np.zeros(len(realData_test)), columns=['label'])
    Xtest = pd.concat([signal_test, realData_test],         ignore_index=True)
    Ytest = pd.concat([y_signal_test, y_realData_test],     ignore_index=True)
    SFtest = Xtest.sf    
    assert(len(Xtrain)==len(Ytrain))
      

    model = load_model("/t3home/gcelotto/ggHbb/outputs/model_inclusive.h5")
    y_predict = model.predict(Xtest[featuresForTraining])
    yTrain_predict = model.predict(Xtrain[featuresForTraining])
    thresholds = np.linspace(0, 0.99, 100)
    thresholds = np.concatenate((thresholds, np.linspace(0.99, 1., 1000)))
    
    signal_predictions = y_predict[Ytest==1]
    realData_predictions = y_predict[Ytest==0]
    
    signalTrain_predictions = yTrain_predict[Ytrain==1]
    print("train predictions", len(signalTrain_predictions))
    print("test predictions", len(signal_predictions))
    realDataTrain_predictions = yTrain_predict[Ytrain==0]
    y_predict = pd.DataFrame(y_predict.flatten(), columns=['label'])
    Ytest = Ytest.reset_index(drop=True)
    Xtest = Xtest.reset_index(drop=True)
    SFtest = SFtest.reset_index(drop=True)
    y_predict = y_predict.reset_index(drop=True)


    
    roc(thresholds, signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_roc.png")
    maskSignal = (Xtest[Ytest.label==1].dijet_mass>123.11-2*17) & (Xtest[Ytest.label==1].dijet_mass<123.11+2*17)
    maskRealData = (Xtest[Ytest.label==0].dijet_mass>123.11-2*17) & (Xtest[Ytest.label==0].dijet_mass<123.11+2*17)
    WorkingPoint(signal_predictions[maskSignal], realData_predictions[maskRealData], outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/cut_on_NN_output.png")
    NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName = "/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_outputs.png")    
    assert len(SFtest)==len(Xtest)
    massSpectrum(Xtest, Ytest, y_predict, SFtest, hp, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/dijetMass_afterCut.png")
    getShap(Xtest[:1000][featuresForTraining], model, outName = '/t3home/gcelotto/ggHbb/outputs/plots/NN/shap.png')

    

    return

if __name__ =="__main__":
    doTrain = bool(int(sys.argv[1])) if len(sys.argv)>1 else False
    nRealDataFiles = int(sys.argv[2]) if len(sys.argv)>2 else False
    print("doTrain", doTrain)
    HbbClassifier(doTrain=doTrain, nRealDataFiles=nRealDataFiles)