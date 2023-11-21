import matplotlib.pyplot as plt
import numpy as np
from plotScripts.utilsForPlot import loadData, loadDataOnlyFeatures, getXSectionBR
import glob
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

signalFileNames = glob.glob(signalPath+"/*bScoreBased4_*.npy")
realDataFileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")

for i in range(len(realDataFileNames)):
    try:
        realData = np.load(realDataFileNames[i])[:,[0, 1]]
    except:
        print("Not able to open %s"%realDataFileNames[i])
    mask = realData[:,0]<5
    l=np.sum(mask)
    print("%d/%d   L=%d"%(i, len(realDataFileNames), l))
    if l>1:
        print(realDataFileNames[i])
    