import matplotlib.pyplot as plt
import numpy as np
from plotScripts.utilsForPlot import loadData, loadDataOnlyFeatures, getXSectionBR
import glob
import pandas as pd

def main():
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
    corrupted=[]
    signalFileNames = glob.glob(signalPath+"/*.parquet")
    realDataFileNames = glob.glob(realDataPath+"/*.parquet")
    for i in range(len(realDataFileNames)):
        try:
            realData = pd.read_parquet(realDataFileNames[i], columns=['dijet_mass'])
        except:
            print("Not able to open %s"%realDataFileNames[i])
        mask = realData.iloc[:,0]<5
        l=np.sum(mask)
        print("%d/%d   L=%d"%(i+1, len(realDataFileNames), l))
        if l>1:
            print(realDataFileNames[i])
            corrupted.append(realDataFileNames[i])
    print("List of corrupted files:")
    print(corrupted)
if __name__=="__main__" :
    main()