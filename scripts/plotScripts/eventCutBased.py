import numpy as np
import matplotlib.pyplot as plt
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import loadData, getFeaturesBScoreBased, getXSectionBR, scatter2Features,getBins
from fuzzywuzzy import process
from plotFeaturesBscoreBased4 import plotNormalizedFeatures
from plotDijetSpectrumpTDifferential import plotDijetpTDifferential
from plotDijetSpectrum import plotDijetMass
import os
import sys

def cutArray(array, featureID, min, max, featureNames = None):
    '''
    Perform a cut on a feature of the array. The selected features is cut to be >min and <max
    The featureID is either the number of the features or the name.
    If the name is provided the featureNames argument must be given.
    The array is passed as a (N_evts, n_features) numpy array
    '''

    if featureNames is not None:
        #print("featureNames are passed as argument")
        if featureID not in featureNames:
            similarFeatureID = process.extractOne(featureID, featureNames)[0]
            print(featureID, " not found in featureNames but %s was taken as the most similar"%similarFeatureID)
            featureID=similarFeatureID
        assert len(featureNames)==array.shape[1], "Dimensions of labels and features not matching"
        featureNumber = featureNames.index(featureID)
    else:
        assert featureID>=0, "featureID must be a number >= 0 when featureNames is not None"
        featureNumber = featureID
    #initialLength = len(array)
    mask = np.ones(len(array), dtype=bool)
    if min is not None:
        mask = (mask) & (array[:,featureNumber]>min) 
    if max is not None:
        mask = (mask) & (array[:,featureNumber]<max)
    array = array[mask]
    #finalLength = len(array)
    #print("With the cut in feature %d,  %.2f%% of the data are left: %d%% entries"%(featureNumber, finalLength/initialLength*100, finalLength))
    
    return array

def cutSignalAndBackground(signal, realData, featureID, min, max, featureNames = None):
    initialSignalLength, initialBkgLength = len(signal), len(realData)
    signal = cutArray(signal, featureID, min, max, featureNames)
    realData = cutArray(realData, featureID, min, max, featureNames)
    finalSignalLength, finalBkgLength = len(signal), len(realData)
    print(min, max)
    print("   %.2f %.2f"%(finalSignalLength*100/initialSignalLength, finalBkgLength*100/initialBkgLength))
    return signal, realData
def main(doCut=False, realFiles=1):
    labels = getFeaturesBScoreBased()
    signalCutFileName = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/signalCut.npy"
    bkgCutFileName = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/realDataCut.npy"
    #pt classes
    q1, q2, q3 = 50, 100, 200
    
    if doCut:
        # Loading files
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatData"
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatData"
        
        signal, realData = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles)

        m1, m2 = 125-3*16, 125+3*16
        print("Keeping only dijet mass within %d-%d"%(m1, m2))
        signal, realData = cutSignalAndBackground(signal, realData, "DijetMass", m1, m2, labels)
        #signal, realData = cutSignalAndBackground(signal, realData, "DijetMass", 125-3*17, 125+3*17, labels)
        
        initialSignalLength, initialBkgLength = len(signal), len(realData)
        print("Initial Length of signal: %d\nInitial Length of bkg: %d\n\n"%(len(signal), len(realData)))

        print("SIGNAL\n")
        # Check QGLikelihood minimum pt for which it works
        mask=(signal[:,8]>0)
        print("Minimum pt of jet1 for which Jet_QGLikelihood > 0 : ",np.min(signal[mask,0]))
        
        # These lines are used to find classes then just use float numbers to fix them 
#signal = signal[signal[:, 18].argsort()]
#cumulative_weights = np.cumsum(signal[:, -1])
#q1_index = np.argmax(cumulative_weights >= cumulative_weights[-1] / 4)
#q2_index = np.argmax(cumulative_weights >= cumulative_weights[-1] / 2)
#q3_index = np.argmax(cumulative_weights >= 3 * cumulative_weights[-1] / 4)
#q1 = signal[q1_index, 18]
#q2 = signal[q2_index, 18]
#q3 = signal[q3_index, 18]
        
        # Pt CLASSES
        
        print("Classes of pT with SF applied : ", q1, q2, q3)
        maskSig1, maskBkg1 = signal[:,18]<q1,                               realData[:,18]< q1
        maskSig2, maskBkg2 = (signal[:,18]>=q1) & (signal[:,18]<q2),        (realData[:,18]>=q1) & (realData[:,18]<q2)
        maskSig3, maskBkg3 = (signal[:,18]>=q2) & (signal[:,18]<q3),        (realData[:,18]>=q2) & (realData[:,18]<q3)
        maskSig4, maskBkg4 = (signal[:,18]>=q3),                            (realData[:,18]>=q3)
        print(np.sum(signal[:,-1][maskSig1]))
        print(np.sum(signal[:,-1][maskSig2]))
        print(np.sum(signal[:,-1][maskSig3]))
        print(np.sum(signal[:,-1][maskSig4]))
        
        signal1, realData1 = signal[maskSig1],  realData[maskBkg1]
        signal2, realData2 = signal[maskSig2],  realData[maskBkg2]
        signal3, realData3 = signal[maskSig3],  realData[maskBkg3]
        signal4, realData4 = signal[maskSig4],  realData[maskBkg4]


        signal1, realData1 = signal[maskSig1],  realData[maskBkg1]     
        signal2, realData2 = signal[maskSig2],  realData[maskBkg2]
        signal3, realData3 = signal[maskSig3],  realData[maskBkg3]
        signal4, realData4 = signal[maskSig4],  realData[maskBkg4]


        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet1Pt", 20, None, labels)
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet2Pt", 20, None, labels)                 # 9
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet2_btagDeepFlavB", 0.2, None, labels)      # 15
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet1_btagDeepFlavB", 0.2, None, labels)
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet1_QGl", 0.25, None, labels)
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "Jet2_QGl", 0.25, None, labels)
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "twist_Dijet", 0.8, None, labels)
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "dPhi_Dijet", 2.6, None, labels)            #  24
        signal1, realData1 = cutSignalAndBackground(signal1, realData1, "dR_Dijet", 0, 4, labels)
        

    # second class
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet1Pt", 20, None, labels)
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet2Pt", 20, None, labels)
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "dPhi_Dijet", 2.3, None, labels)            #  24
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "dEta_Dijet", 0, 2.7, labels)
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet1_btagDeepFlavB", 0.25, None, labels)    
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet2_btagDeepFlavB", 0.25, None, labels)    
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet1_QGl", 0.2, None, labels)
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "Jet2_QGl", 0.2, None, labels)
        signal2, realData2 = cutSignalAndBackground(signal2, realData2, "dR_Dijet", 0, 4, labels)
    
    # third class
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet1Pt", 20, 120, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet2Pt", 20, 80, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "twist_Dijet", 0.5, None, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet1_btagDeepFlavB", 0.2, None, labels)    # 6
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet2_btagDeepFlavB", 0.3, None, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "dPhi_Dijet", 2.0, None, labels) 
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "dR_Dijet", 0, 3.5, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet1_QGl", 0.2, None, labels)
        signal3, realData3 = cutSignalAndBackground(signal3, realData3, "Jet2_QGl", 0.2, None, labels)

    # fourth class
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet1Pt", 20, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet2Pt", 20, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "twist_Dijet", 0.2, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "ht", 200, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "dPhi_Dijet", 0.6, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "dEta_Dijet", 0., 2.8, labels) 
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet1_QGl", 0.2, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet2_QGl", 0.2, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet1_btagDeepFlavB", 0.2, None, labels)
        signal4, realData4 = cutSignalAndBackground(signal4, realData4, "Jet2_btagDeepFlavB", 0.2, None, labels)







        signal      = np.vstack((signal1, signal2, signal3, signal4))
        realData    = np.vstack((realData1, realData2, realData3, realData4))

        print("After cuts percentage of each class")
        print("%.1f%% %.1f%%"%(100*np.sum(signal1[:,-1])/np.sum(signal[:,-1]), 100*np.sum(realData1[:,-1])/np.sum(realData[:,-1])))
        print("%.1f%% %.1f%%"%(100*np.sum(signal2[:,-1])/np.sum(signal[:,-1]), 100*np.sum(realData2[:,-1])/np.sum(realData[:,-1])))
        print("%.1f%% %.1f%%"%(100*np.sum(signal3[:,-1])/np.sum(signal[:,-1]), 100*np.sum(realData3[:,-1])/np.sum(realData[:,-1])))
        print("%.1f%% %.1f%%"%(100*np.sum(signal4[:,-1])/np.sum(signal[:,-1]), 100*np.sum(realData4[:,-1])/np.sum(realData[:,-1])))

        # a few lines just to print the pt and IP significance to check if SF are applied correctly
        #print("TEST THE SF")
        #import random
        #rndInt = random.randint(1, 100)
        #print(signal[rndInt, 29], signal[rndInt, 33], signal[rndInt,-1])

        try:
            if os.path.exists(signalCutFileName):
                os.remove(signalCutFileName)
            if os.path.exists(bkgCutFileName):
                os.remove(bkgCutFileName)
            print("Saving signal in trivcat...")
            np.save(signalCutFileName, signal)
            print("Saving bkg    in trivcat...")
            np.save(bkgCutFileName, realData)
        except:
            print("saving in scratch...")
            np.save("/scratch/signalCut.npy", signal)
            np.save("/scratch/realDataCut.npy", realData)
        
        finalSignalLength, finalBkgLength = len(signal), len(realData)
        print("Final Length of signal:\n%d = %.3f%%\nFinal Length of bkg:\n%d = %.4f%%\n\n"%(finalSignalLength, finalSignalLength/initialSignalLength*100, finalBkgLength, finalBkgLength/initialBkgLength*100))

    # Load the files just saved or the ones previously saved    
    signal=np.load(signalCutFileName)
    realData=np.load(bkgCutFileName)
    maskSig1 = signal[:,18]<q1
    maskSig2 = (signal[:,18]>=q1) & (signal[:,18]<q2)
    maskSig3 = (signal[:,18]>=q2) & (signal[:,18]<q3)
    maskSig4 = (signal[:,18]>=q3)
    maskBkg1 = realData[:,18]<q1
    maskBkg2 = (realData[:,18]>=q1) & (realData[:,18]<q2)
    maskBkg3 = (realData[:,18]>=q2) & (realData[:,18]<q3)
    maskBkg4 = (realData[:,18]>=q3)
    signal1 = signal[maskSig1]
    signal2 = signal[maskSig2]
    signal3 = signal[maskSig3]
    signal4 = signal[maskSig4]
    realData1 = realData[maskBkg1]
    realData2 = realData[maskBkg2]
    realData3 = realData[maskBkg3]
    realData4 = realData[maskBkg4]
    
    #toKeep =  [ 0, 6,
    #                9, 15,
    #                23, 24, -1]
    #if toKeep[-1]!=-1:
    #    c=input("Sure to continue toKeep[-1]!=-1 (y/n)")
    #    if c!='y':
    #        return
    #    else:
    #        del c
    toKeep = [0, 6, 8 ,
              9, 15, 17,
              22, 23, 24, 25,26,-1
            ]
    plotDijetpTDifferential(realFiles=realFiles)
    plotNormalizedFeatures(signal=signal1, realData=realData1, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt0to%d_CUT.png"%(q1), toKeep=toKeep)
    plotNormalizedFeatures(signal=signal2, realData=realData2, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt%dto%d_CUT.png"%(q1, q2), toKeep=toKeep)
    plotNormalizedFeatures(signal=signal3, realData=realData3, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt%dto%d_CUT.png"%(q2, q3), toKeep=toKeep)
    plotNormalizedFeatures(signal=signal4, realData=realData4, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_pt%dtoInf_CUT.png"%(q3), toKeep=toKeep)
    #plotDijetMass(afterCut=True, log=True, fit=True, realFiles=realFiles)
    plotDijetMass(afterCut=True, log=False, fit=True, realFiles=realFiles)



    print("\n\n\n")
    
    

if __name__=="__main__":
    doCut = bool(int(sys.argv[1])) if len(sys.argv)>1 else True
    realFiles = int(sys.argv[2]) if len(sys.argv)>2 else -1
    
    main(doCut=doCut, realFiles=realFiles)

