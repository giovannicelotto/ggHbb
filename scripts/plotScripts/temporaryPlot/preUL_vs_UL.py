import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, glob
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/")
from plotFeatures import plotNormalizedFeatures
def cut(data, feature, min, max):
            newData = []
            for df in data:
                if min is not None:
                    df = df[df[feature] > min]
                if max is not None:
                    df = df[df[feature] < max]
                newData.append(df)
            return newData
def main():
    fileNames_preUL    = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_old/ggH2023Dec06_nonUL/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/**/*.parquet", recursive=True)
    fileNames_UL       = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**/*.parquet", recursive=True)
    df_preUL           = pd.read_parquet(fileNames_preUL)
    df_UL              = pd.read_parquet(fileNames_UL)
    
    print(len(df_preUL))
    print(len(df_UL))
    df_preUL, df_UL = cut([df_preUL, df_UL], 'jet1_pt', 20, None)
    df_preUL, df_UL = cut([df_preUL, df_UL], 'jet2_pt', 20, None)
    plotNormalizedFeatures(data=[df_preUL, df_UL], 
                           outFile="/t3home/gcelotto/ggHbb/outputs/plots/features/preUL_vs_UL.png",
                           legendLabels=['preUL', 'UL'],
                           colors=['blue', 'red'],
                           histtypes=[u'step', u'step'],
                           figsize=(15, 30))
    


    return


if __name__ == "__main__":
    main()