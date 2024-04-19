import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import loadData, getFeaturesBScoreBased, getXSectionBR, scatter2Features,getBins, loadDask
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
    

    #if featureNames is not None:
#
    #    #print("featureNames are passed as argument")
    #    if featureID not in featureNames:
    #        similarFeatureID = process.extractOne(featureID, featureNames)[0]
    #        print(featureID, " not found in featureNames but %s was taken as the most similar"%similarFeatureID)
    #        featureID=similarFeatureID
    #    assert len(featureNames)==array.shape[1], "Dimensions of labels and features not matching"
    #    featureNumber = featureNames.index(featureID)
    #else:
    #    assert featureID>=0, "featureID must be a number >= 0 when featureNames is not None"
    #    featureNumber = featureID
    #initialLength = len(array)
    #array.set_index('jet1_pt')
    if min is None:
        sys.exit("Give a min")
    mask = array.loc[featureID]>min
    if max is not None:
        mask = (mask) & (array.loc[featureID]<max)
    array = array[mask]
    #finalLength = len(array)
    #print("With the cut in feature %d,  %.2f%% of the data are left: %d%% entries"%(featureNumber, finalLength/initialLength*100, finalLength))
    
    return array
def cutDaskDataFrame(signal, realData, featureName, min, max):
    maskSignal = (signal.dijet_pt)>-9999 # always true
    if min is not None:
        maskSignal = (maskSignal) & (signal[featureName] > min)
    if max is not None:
        maskSignal = (maskSignal) & (signal[featureName] < max)

    maskData = (realData.dijet_pt)>-9999 # always true
    if min is not None:
        maskData = (maskData) & (realData[featureName] > min)
    if max is not None:
        maskData = (maskData) & (realData[featureName] < max)

    return signal[maskSignal], realData[maskData]
        
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
    signalCutFileName = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/signalCut.parquet"
    bkgCutFileName = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/realDataCut.parquet"
    #pt classes
    q1, q2, q3 = 50, 100, 200
    
    if doCut:
        # Loading files
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData"
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
        
        signal, realData = loadDask(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles)
        
        m1, m2 = 125-3*16, 125+3*16
        print("Keeping only dijet mass within %d-%d"%(m1, m2))
        #signal, realData = cutSignalAndBackground(signal, realData, "dijet_mass", m1, m2, labels)
        initialSignalLength, initialBkgLength = len(signal), len(realData)
        print("Initial Length of signal: %d\nInitial Length of bkg: %d\n\n"%(len(signal), len(realData)))
        maskS = (signal.dijet_mass>m1) & (signal.dijet_mass<m2)
        maskB = (realData.dijet_mass>m1) & (realData.dijet_mass<m2)
        signal, realData = signal[maskS], realData[maskB]
        initialSignalLength, initialBkgLength = len(signal), len(realData)
        print("Final Length of signal: %d\nFinal Length of bkg: %d\n\n"%(len(signal), len(realData)))

        print("SIGNAL\n")
        # Check QGLikelihood minimum pt for which it works
        #mask=(signal[:,8]>0)
        #print("Minimum pt of jet1 for which Jet_QGLikelihood > 0 : ",np.min(signal[mask,0]))
        
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
        
        maskSig1, maskBkg1 = (signal.dijet_pt<q1),                                  (realData.dijet_pt<q1)
        maskSig2, maskBkg2 = ((signal.dijet_pt>=q1) & (signal.dijet_pt<q2)),        ((realData.dijet_pt>=q1) & (realData.dijet_pt<q2))
        maskSig3, maskBkg3 = ((signal.dijet_pt>=q2) & (signal.dijet_pt<q3)),        ((realData.dijet_pt>=q2) & (realData.dijet_pt<q3))
        maskSig4, maskBkg4 = (signal.dijet_pt>=q3),                                 ((realData.dijet_pt>=q3))
        
        signal1, realData1 = signal[maskSig1],  realData[maskBkg1]
        signal2, realData2 = signal[maskSig2],  realData[maskBkg2]
        signal3, realData3 = signal[maskSig3],  realData[maskBkg3]
        signal4, realData4 = signal[maskSig4],  realData[maskBkg4]
        print(signal1.sf.sum().compute(), realData1.sf.sum().compute())
        print(signal2.sf.sum().compute(), realData2.sf.sum().compute())
        print(signal3.sf.sum().compute(), realData3.sf.sum().compute())
        print(signal4.sf.sum().compute(), realData4.sf.sum().compute())


        

        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet1_pt", 20, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet2_pt", 20, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet2_btagDeepFlavB", .2, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet1_btagDeepFlavB", .2, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet1_qgl", .25, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "jet2_qgl", .25, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "dijet_twist", .8, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "dijet_dPhi", 2.6, None)
        signal1, realData1 = cutDaskDataFrame(signal1, realData1, "dijet_dR", 0, 4)
        

    # second class
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet1_pt", 20, None)
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet2_pt", 20, None)
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "dijet_dPhi", 2.3, None)            #  24
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "dijet_dEta", 0, 2.7)
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet1_btagDeepFlavB", 0.25, None)    
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet2_btagDeepFlavB", 0.25, None)    
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet1_qgl", 0.2, None)
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "jet2_qgl", 0.2, None)
        signal2, realData2 = cutDaskDataFrame(signal2, realData2, "dijet_dR", 0, 4)
    
    # third class
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet1_pt", 20, 120)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet2_pt", 20, 80)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "dijet_twist", 0.5, None)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet1_btagDeepFlavB", 0.2, None)    # 6
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet2_btagDeepFlavB", 0.3, None)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "dijet_dPhi", 2.0, None) 
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "dijet_dR", 0, 3.5)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet1_qgl", 0.2, None)
        signal3, realData3 = cutDaskDataFrame(signal3, realData3, "jet2_qgl", 0.2, None)

    # fourth class
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet1_pt", 20, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet2_pt", 20, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "dijet_twist", 0.2, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "ht", 200, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "dijet_dPhi", 0.6, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "dijet_dEta", 0., 2.8) 
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet1_qgl", 0.2, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet2_qgl", 0.2, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet1_btagDeepFlavB", 0.2, None)
        signal4, realData4 = cutDaskDataFrame(signal4, realData4, "jet2_btagDeepFlavB", 0.2, None)







        signal      = dd.multi.concat([signal1, signal2, signal3, signal4])
        realData    = dd.multi.concat([realData1, realData2, realData3, realData4])

        print("After cuts percentage of each class")
        print("%.1f%% %.1f%%"%(100*signal1.sf.sum().compute()/signal.sf.sum().compute(), 100*realData1.sf.sum().compute()/realData.sf.sum().compute()))
        print("%.1f%% %.1f%%"%(100*signal2.sf.sum().compute()/signal.sf.sum().compute(), 100*realData2.sf.sum().compute()/realData.sf.sum().compute()))
        print("%.1f%% %.1f%%"%(100*signal3.sf.sum().compute()/signal.sf.sum().compute(), 100*realData3.sf.sum().compute()/realData.sf.sum().compute()))
        print("%.1f%% %.1f%%"%(100*signal4.sf.sum().compute()/signal.sf.sum().compute(), 100*realData4.sf.sum().compute()/realData.sf.sum().compute()))

        
        finalSignalLength, finalBkgLength = len(signal), len(realData)
        print("Final Length of signal:\n%d = %.3f%%\nFinal Length of bkg:\n%d = %.4f%%\n\n"%(finalSignalLength, finalSignalLength/initialSignalLength*100, finalBkgLength, finalBkgLength/initialBkgLength*100))

        try:
            if os.path.exists(signalCutFileName):
                os.remove(signalCutFileName)
            if os.path.exists(bkgCutFileName):
                os.remove(bkgCutFileName)
            print("Saving signal in trivcat...")
            signal.to_parquet(signalCutFileName)
            print("Saving bkg    in trivcat...")
            realData.to_parquet(bkgCutFileName)
        except:
            print("saving in scratch...")
            signal.to_parquet("/scratch/signalCut.parquet")
            realData.to_parquet("/scratch/realDataCut.parquet")
        

    # Load the files just saved or the ones previously saved    
    signal=dd.read_parquet(signalCutFileName)
    realData=dd.read_parquet(bkgCutFileName)
    maskSig1, maskBkg1 = (signal.dijet_pt<q1),                                  (realData.dijet_pt<q1)
    maskSig2, maskBkg2 = ((signal.dijet_pt>=q1) & (signal.dijet_pt<q2)),        ((realData.dijet_pt>=q1) & (realData.dijet_pt<q2))
    maskSig3, maskBkg3 = ((signal.dijet_pt>=q2) & (signal.dijet_pt<q3)),        ((realData.dijet_pt>=q2) & (realData.dijet_pt<q3))
    maskSig4, maskBkg4 = (signal.dijet_pt>=q3),                                 ((realData.dijet_pt>=q3))

    signal1, realData1 = signal[maskSig1],  realData[maskBkg1]
    signal2, realData2 = signal[maskSig2],  realData[maskBkg2]
    signal3, realData3 = signal[maskSig3],  realData[maskBkg3]
    signal4, realData4 = signal[maskSig4],  realData[maskBkg4]
    
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
    plotDijetpTDifferential(realFiles=realFiles, afterCut=True)
    sys.exit("End of program")
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

