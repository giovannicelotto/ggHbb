import shap
import tensorflow as tf
from sklearn.utils import shuffle
#from tensorflow.compat.v1.keras.backend import get_session
#tensorflow.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from models import getLowPtModel, getHighPtModel
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import sys
import glob
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
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
                         'muon_pt', 'nJets',     'ht','muon_pfRelIso03_all'
    ]
    columnsToRead = ['jet1_pt', 'jet1_eta', 'jet1_btagDeepFlavB',   'jet1_qgl',
                    'jet2_pt', 'jet2_eta', 'jet2_btagDeepFlavB',   'jet2_qgl',
                    'dijet_pt', 'dijet_eta',            'dijet_mass',
                    'dijet_dR', 'dijet_dEta',           'dijet_dPhi',           'dijet_twist',
                    'muon_pt',  'nJets',     'ht', 'muon_pfRelIso03_all', 
                    'sf'    ]
    np.save("/t3home/gcelotto/ggHbb/scripts/NN/input/featuresForTraining_ptClass.npy", featuresForTraining)
    np.save("/t3home/gcelotto/ggHbb/scripts/NN/input/columnsToRead_ptClass.npy", columnsToRead)

def readFeatures():
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/scripts/NN/input/featuresForTraining_ptClass.npy")
    columnsToRead = np.load("/t3home/gcelotto/ggHbb/scripts/NN/input/columnsToRead_ptClass.npy")
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
    ptDivider = 100
    hp = {
        'epochs'            : 3000,
        'patienceES'        : 200,
        'validation_split'  : 0.2,
        'learning_rate'     : 1e-4
        }
    if doTrain:
        realDataPath_train = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training"
        signalPath_train = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/training"
        signal_train, realData_train = loadParquet(signalPath_train, realDataPath_train, nSignalFiles=-1, nRealDataFiles=-1, 
                                       columns=columnsToRead)
        

        nSignalTrainFilesUsed = len(glob.glob(signalPath_train+"*.parquet"))

        
        #signalRegion
        print("Before preprocessing", len(signal_train), len(realData_train))
        signal_train, realData_train = preprocess(signal_train, realData_train)
        print("After preprocessing", len(signal_train), len(realData_train))
        
        
        
        
    # calcola la significance come control check if want
        if (False):
            totalDataFlat_train = np.load("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_train.npy")
            lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
            N_SignalMini = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_mini.npy")*nSignalTrainFilesUsed/240  # mini per produrre 45
            x1_sb, x2_sb  = 123 - 2*17, 123 + 2*17
            maskSignal =    (signal_train.dijet_mass>x1_sb) & (signal_train.dijet_mass<x2_sb)
            maskData =      (realData_train.dijet_mass>x1_sb) & (realData_train.dijet_mass<x2_sb)
            S = np.sum(signal_train.sf[maskSignal])*lumiPerEvent*totalDataFlat_train/N_SignalMini*getXSectionBR()*1000
            B = np.sum(maskData)
            print("Signal 2sigma", S)
            print("Data 2sigma", B)
            print("Sig", S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat_train)))

    # create the label for training
        
# Create two classes of
        signal_train_lowPt = signal_train[signal_train.dijet_pt<ptDivider]
        signal_train_highPt = signal_train[signal_train.dijet_pt>ptDivider]
        realData_train_lowPt = realData_train[realData_train.dijet_pt<ptDivider]
        realData_train_highPt = realData_train[realData_train.dijet_pt>ptDivider]
        
        print("%d events for signal fiducial (Jetpt>20 GeV) LOW dijet_Pt (diJetpt<%d GeV) \n%d events for background fiducial"%(len(signal_train_lowPt), ptDivider, len(realData_train_lowPt)))
        print("%d events for signal fiducial (Jetpt>20 GeV) HIGH dijet_Pt (diJetpt>%d GeV) \n%d events for background fiducial"%(len(signal_train_highPt), ptDivider, len(realData_train_highPt)))
        lowPtLength = np.min((len(realData_train_lowPt), len(signal_train_lowPt) ))
        highPtLength = np.min((len(realData_train_highPt), len(signal_train_highPt) ))
        signal_train_lowPt = signal_train_lowPt[:lowPtLength]
        realData_train_lowPt = realData_train_lowPt[:lowPtLength]
        signal_train_highPt = signal_train_highPt[:highPtLength]
        realData_train_highPt = realData_train_highPt[:highPtLength]
        print("%d events for signal fiducial (Jetpt>20 GeV) LOW dijet_Pt (diJetpt<%d GeV) \n%d events for background fiducial"%(len(signal_train_lowPt), ptDivider, len(realData_train_lowPt)))
        print("%d events for signal fiducial (Jetpt>20 GeV) HIGH dijet_Pt (diJetpt>%d GeV) \n%d events for background fiducial"%(len(signal_train_highPt), ptDivider, len(realData_train_highPt)))
        
# build y labels for train for two classes
        y_signal_train_lowPt = pd.DataFrame(np.ones(len(signal_train_lowPt)), columns=['label'])
        y_realData_train_lowPt = pd.DataFrame(np.zeros(len(realData_train_lowPt)), columns=['label'])
        y_signal_train_highPt = pd.DataFrame(np.ones(len(signal_train_highPt)), columns=['label'])
        y_realData_train_highPt = pd.DataFrame(np.zeros(len(realData_train_highPt)), columns=['label'])
        
# align consistently X and Y
        Xtrain_lowPt = pd.concat([signal_train_lowPt, realData_train_lowPt],       ignore_index=True)
        Ytrain_lowPt = pd.concat([y_signal_train_lowPt, y_realData_train_lowPt],   ignore_index=True)
        Xtrain_highPt = pd.concat([signal_train_highPt, realData_train_highPt],    ignore_index=True)
        Ytrain_highPt = pd.concat([y_signal_train_highPt, y_realData_train_highPt],ignore_index=True)

        
        
        Xtrain_lowPt, Ytrain_lowPt = shuffle(Xtrain_lowPt, Ytrain_lowPt)
        Xtrain_highPt, Ytrain_highPt = shuffle(Xtrain_highPt, Ytrain_highPt)

        Xtrain_lowPt.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain_lowPt.parquet")
        Ytrain_lowPt.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain_lowPt.parquet")
        Xtrain_highPt.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain_highPt.parquet")
        Ytrain_highPt.to_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain_highPt.parquet")
        
        model_lowPt = getLowPtModel(len(Xtrain_lowPt[featuresForTraining].columns))
        optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model_lowPt.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        model_highPt = getHighPtModel(len(Xtrain_highPt[featuresForTraining].columns))
        optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model_highPt.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        callbacks=[]
        earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
        callbacks.append(earlyStop)
        
        #fit = model_lowPt.fit(Xtrain_lowPt[featuresForTraining], Ytrain_lowPt,
        #                epochs=hp['epochs'], #batch_size=128, 
        #                callbacks=callbacks, validation_split=hp['validation_split'], shuffle=True)
        #model_lowPt.save("/t3home/gcelotto/ggHbb/outputs/model_lowPt.h5")
        #doPlotLoss(fit=fit, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/loss_lowPt.png", earlyStop=earlyStop, patience=hp['patienceES'])
        #print("Feaures for training : ",featuresForTraining)
        fit = model_highPt.fit(Xtrain_highPt[featuresForTraining], Ytrain_highPt,
                        epochs=hp['epochs'], callbacks=callbacks, validation_split=hp['validation_split'], shuffle=True)
        model_highPt.save("/t3home/gcelotto/ggHbb/outputs/model_highPt.h5")
        yTrain_predict_highPt = model_highPt.predict(Xtrain_highPt[featuresForTraining])
        
        doPlotLoss(fit=fit, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/loss_highPt.png", earlyStop=earlyStop, patience=hp['patienceES'])
    

    # load the train for plots
    Xtrain_lowPt      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain_lowPt.parquet")
    Ytrain_lowPt      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain_lowPt.parquet")['label']
    Xtrain_highPt      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Xtrain_highPt.parquet")
    Ytrain_highPt      = pd.read_parquet("/t3home/gcelotto/ggHbb/outputs/df_NN/Ytrain_highPt.parquet")['label']
    # load the test
    realDataPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"
    signalPath_test = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/others"
    signal_test, realData_test = loadParquet(signalPath_test, realDataPath_test, nSignalFiles=-1, nRealDataFiles=nRealDataFiles, 
                                    columns=columnsToRead)
    totalDataFlat_test = len(realData_test)
    np.save("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_test.npy", totalDataFlat_test)
    
    # signalRegion
    print("Performing the cut in pt and eta for the testing")
    signal_test, realData_test      = preprocess(signal_test, realData_test)
    signal_test_lowPt = signal_test[signal_test.dijet_pt<ptDivider]
    signal_test_highPt = signal_test[signal_test.dijet_pt>ptDivider]
    realData_test_lowPt = realData_test[realData_test.dijet_pt<ptDivider]
    realData_test_highPt = realData_test[realData_test.dijet_pt>ptDivider]
    if doTrain:
        plotNormalizedFeatures(data=[signal_train, realData_train,
                                 signal_test, realData_test],
                               outFile = "/t3home/gcelotto/ggHbb/outputs/plots/NN/features_train.png",
                            legendLabels = ['Signal train ', 'BParking train', 'Signal test ', 'BParking test'] , colors = ['blue', 'red', 'blue', 'red'],
                            histtypes=[u'step', u'step', 'bar', 'bar'],
                            alphas=[1, 1, 0.4, 0.4])

    y_signal_test_lowPt = pd.DataFrame(np.ones(len(signal_test_lowPt)), columns=['label'])
    y_realData_test_lowPt = pd.DataFrame(np.zeros(len(realData_test_lowPt)), columns=['label'])
    Xtest_lowPt = pd.concat([signal_test_lowPt, realData_test_lowPt],         ignore_index=True)
    Ytest_lowPt = pd.concat([y_signal_test_lowPt, y_realData_test_lowPt],     ignore_index=True)
    Xtest_lowPt, Ytest_lowPt = shuffle(Xtest_lowPt, Ytest_lowPt)
    SFtest_lowPt = Xtest_lowPt.sf    
    del y_signal_test_lowPt
    del y_realData_test_lowPt
    assert(len(Xtrain_lowPt)==len(Ytrain_lowPt))
    assert(len(Xtest_lowPt)==len(Ytest_lowPt))
    y_signal_test_highPt = pd.DataFrame(np.ones(len(signal_test_highPt)), columns=['label'])
    y_realData_test_highPt = pd.DataFrame(np.zeros(len(realData_test_highPt)), columns=['label'])
    Xtest_highPt = pd.concat([signal_test_highPt, realData_test_highPt],         ignore_index=True)
    Ytest_highPt = pd.concat([y_signal_test_highPt, y_realData_test_highPt],     ignore_index=True)
    Xtest_highPt, Ytest_highPt = shuffle(Xtest_highPt, Ytest_highPt)
    SFtest_highPt = Xtest_highPt.sf    
    del y_signal_test_highPt
    del y_realData_test_highPt
    assert(len(Xtrain_highPt)==len(Ytrain_highPt))
    assert(len(Xtest_highPt)==len(Ytest_highPt))
      

    model_lowPt = load_model("/t3home/gcelotto/ggHbb/outputs/model_lowPt.h5")
    y_predict_lowPt = model_lowPt.predict(Xtest_lowPt[featuresForTraining])
    yTrain_predict_lowPt = model_lowPt.predict(Xtrain_lowPt[featuresForTraining])
    model_highPt = load_model("/t3home/gcelotto/ggHbb/outputs/model_highPt.h5")
    y_predict_highPt = model_highPt.predict(Xtest_highPt[featuresForTraining])
    yTrain_predict_highPt = model_highPt.predict(Xtrain_highPt[featuresForTraining])
    thresholds = np.linspace(0, 0.99, 100)
    thresholds = np.concatenate((thresholds, np.linspace(0.99, 1., 1000)))
    
    signal_predictions_lowPt = y_predict_lowPt[Ytest_lowPt==1]
    realData_predictions_lowPt = y_predict_lowPt[Ytest_lowPt==0]

    
    signalTrain_predictions_lowPt = yTrain_predict_lowPt[Ytrain_lowPt==1]
    print("train predictions", len(signalTrain_predictions_lowPt))
    print("test predictions", len(signal_predictions_lowPt))
    realDataTrain_predictions_lowPt = yTrain_predict_lowPt[Ytrain_lowPt==0]
    y_predict_lowPt = pd.DataFrame(y_predict_lowPt.flatten(), columns=['label'])
    Ytest_lowPt = Ytest_lowPt.reset_index(drop=True)
    Xtest_lowPt = Xtest_lowPt.reset_index(drop=True)
    SFtest_lowPt = SFtest_lowPt.reset_index(drop=True)
    y_predict_lowPt = y_predict_lowPt.reset_index(drop=True)

    signal_predictions_highPt = y_predict_highPt[Ytest_highPt==1]
    realData_predictions_highPt = y_predict_highPt[Ytest_highPt==0]
    signalTrain_predictions_highPt = yTrain_predict_highPt[Ytrain_highPt==1]
    print("train predictions", len(signalTrain_predictions_highPt))
    print("test predictions", len(signal_predictions_highPt))
    realDataTrain_predictions_highPt = yTrain_predict_highPt[Ytrain_highPt==0]
    y_predict_highPt = pd.DataFrame(y_predict_highPt.flatten(), columns=['label'])
    Ytest_highPt = Ytest_highPt.reset_index(drop=True)
    Xtest_highPt = Xtest_highPt.reset_index(drop=True)
    SFtest_highPt = SFtest_highPt.reset_index(drop=True)
    y_predict_highPt = y_predict_highPt.reset_index(drop=True)


    print("Minimum dijet pt of high class : ",Xtest_highPt.dijet_pt.min())
    massSpectrum(Xtest_lowPt, Ytest_lowPt, y_predict_lowPt, SFtest_lowPt, hp, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/dijetMass_afterCut_lowPt.png")
    massSpectrum(Xtest_highPt, Ytest_highPt, y_predict_highPt, SFtest_highPt, hp, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/dijetMass_afterCut_highPt.png")
    
    roc(thresholds, signal_predictions_lowPt, realData_predictions_lowPt, signalTrain_predictions_lowPt, realDataTrain_predictions_lowPt, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_roc_lowPt.png")
    signalMask = (Xtest_lowPt[Ytest_lowPt.label==1].dijet_mass>123.11-2*17) & (Xtest_lowPt[Ytest_lowPt.label==1].dijet_mass<123.11+2*17)
    realDataMask = (Xtest_lowPt[Ytest_lowPt.label==0].dijet_mass>123.11-2*17) & (Xtest_lowPt[Ytest_lowPt.label==0].dijet_mass<123.11+2*17)
    WorkingPoint(signal_predictions_lowPt[signalMask], realData_predictions_lowPt[realDataMask], outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/cut_on_NN_output_lowPt.png")
    NNoutputs(signal_predictions_lowPt, realData_predictions_lowPt, signalTrain_predictions_lowPt, realDataTrain_predictions_lowPt, outName = "/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_outputs_lowPt.png")    
    print(type(SFtest_lowPt), type(Xtest_lowPt))
    
    assert len(SFtest_lowPt)==len(Xtest_lowPt)
    getShap(Xtest_lowPt[:300][featuresForTraining], model_lowPt, outName = "/t3home/gcelotto/ggHbb/outputs/plots/NN/shap_lowPt.png")

    roc(thresholds, signal_predictions_highPt, realData_predictions_highPt, signalTrain_predictions_highPt, realDataTrain_predictions_highPt, outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_roc_highPt.png")
    signalMask = (Xtest_highPt[Ytest_highPt.label==1].dijet_mass>123.11-2*17) & (Xtest_highPt[Ytest_highPt.label==1].dijet_mass<123.11+2*17)
    realDataMask = (Xtest_highPt[Ytest_highPt.label==0].dijet_mass>123.11-2*17) & (Xtest_highPt[Ytest_highPt.label==0].dijet_mass<123.11+2*17)
    WorkingPoint(signal_predictions_highPt[signalMask], realData_predictions_highPt[realDataMask], outName="/t3home/gcelotto/ggHbb/outputs/plots/NN/cut_on_NN_output_highPt.png")
    NNoutputs(signal_predictions_highPt, realData_predictions_highPt, signalTrain_predictions_highPt, realDataTrain_predictions_highPt, outName = "/t3home/gcelotto/ggHbb/outputs/plots/NN/nn_outputs_highPt.png")    
    assert len(SFtest_highPt)==len(Xtest_highPt)
    getShap(Xtest_highPt[:300][featuresForTraining], model_highPt, outName = "/t3home/gcelotto/ggHbb/outputs/plots/NN/shap_highPt.png")
    return

if __name__ =="__main__":
    doTrain = bool(int(sys.argv[1])) if len(sys.argv)>1 else False
    nRealDataFiles = int(sys.argv[2]) if len(sys.argv)>2 else False
    print("doTrain", doTrain)
    HbbClassifier(doTrain=doTrain, nRealDataFiles=nRealDataFiles)