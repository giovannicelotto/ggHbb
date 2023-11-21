import numpy as np
import matplotlib.pyplot as plt
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import loadData, getFeaturesBScoreBased, getXSectionBR
from fuzzywuzzy import process
from plotFeaturesBscoreBased4 import plotNormalizedFeatures

def cutArray(array, featureID, min, max, featureNames = None):
    '''Perform a cut on a feature of the array. The selected features is cut to be >min and <max
    The featureID is either the number of the features or the name.
    If the name is provided the featureNames argument must be given.
    The array is passed as a (N_evts, n_features) numpy array'''

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
def main():
    # Loading files
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

    signal, realData = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=-1)

    # Correction factors and counters
    #N_mini_BPH = np.load("/t3home/gcelotto/ggHbb/outputs/N_mini.npy")
    #N_nano_ggH = np.load("/t3home/gcelotto/ggHbb/outputs/N_BPH_Nano.npy")
    #correctionData = N_nano_ggH/len(realData)
    labels = getFeaturesBScoreBased()
    initialSignalLength, initialBkgLength = len(signal), len(realData)
    print("Initial Length of signal:\n%d\nInitial Length of bkg:\n%d\n\n"%(len(signal), len(realData)))

    print("SIGNAL\n")
    mask=(signal[:,8]>0)
    print(np.min(signal[mask,0]))
    signal, realData = cutSignalAndBackground(signal, realData, "Jet1Pt", 20, None, labels)                 # 0
    signal, realData = cutSignalAndBackground(signal, realData, "Jet1Mass", 6, None, labels)                # 3
    signal, realData = cutSignalAndBackground(signal, realData, "Jet1_btagDeepFlavB", 0.2, None, labels)    # 6
    signal, realData = cutSignalAndBackground(signal, realData, "Jet2Pt", 20, None, labels)                 # 9
    signal, realData = cutSignalAndBackground(signal, realData, "Jet2Eta", -2, 2, labels)                   # 10
    signal, realData = cutSignalAndBackground(signal, realData, "twist_Dijet", 0.76, None, labels)          # 26
    signal, realData = cutSignalAndBackground(signal, realData, "dEta_Dijet", 0, 1.4, labels)               # 23
    signal, realData = cutSignalAndBackground(signal, realData, "Jet2_btagDeepFlavB", 0.35, 1, labels)      # 15
    signal, realData = cutSignalAndBackground(signal, realData, "dPhi_Dijet", 1.4, None, labels)            #  24
    signal, realData = cutSignalAndBackground(signal, realData, "Jet1_QGl", 0.25, None, labels)             # 8
    signal, realData = cutSignalAndBackground(signal, realData, "Jet2_QGl", 0.25, None, labels)             # 17
    signal, realData = cutSignalAndBackground(signal, realData, "ht", 180, None, labels)                    # 27
    signal, realData = cutSignalAndBackground(signal, realData, "Jet1Pt", 45, 180, labels)                  # repeat for process understanding
    #signal, realData = cutSignalAndBackground(signal, realData, "Jet1Eta", -5+2.46, 5-2.46, labels)
    #signal, realData = cutSignalAndBackground(signal, realData, "Jet2Pt", 20, None, labels) 
    #signal, realData = cutSignalAndBackground(signal, realData, "DijetMass", 60, 200, labels)
    # Save Data after cuts
    try:
        np.save("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/afterCutWithMoreFeatures/signalCut.npy", signal)
        np.save("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/afterCutWithMoreFeatures/realDataCut.npy", realData)
    except:
        pass
        np.save("/t3home/gcelotto/ggHbb/outputs/signalCut.npy", signal)
        np.save("/t3home/gcelotto/ggHbb/outputs/realDataCut.npy", realData)
    toKeep = [ 0, 3, 6,
                9, 10, 15,
                23, 24, 26,
                8, 17, 27]
    plotNormalizedFeatures(signal=signal, realData=realData, outFile = "/t3home/gcelotto/ggHbb/outputs/plots/normalizedFeatures_cuts.png", toKeep=toKeep)
    print("\n\n\n")
    

    #Try new cuts
    #for i in np.linspace(0, 400, 200):
    #    Newsignal, NewrealData = cutSignalAndBackground(signal, realData, "ht", i, None, labels)
    



    finalSignalLength, finalBkgLength = len(signal), len(realData)
    print("Final Length of signal:\n%d = %.3f%%\nFinal Length of bkg:\n%d = %.4f%%\n\n"%(finalSignalLength, finalSignalLength/initialSignalLength*100, finalBkgLength, finalBkgLength/initialBkgLength*100))

if __name__=="__main__":
    main()

