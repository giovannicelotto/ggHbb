import sys
import uproot
import awkward as ak
import numpy as np
import pandas as pd

def getPU_sfs(PV_npvs):
    df_PU = pd.read_csv("/t3home/gcelotto/ggHbb/PU_reweighting/output/pu_sfs.csv")
    indexes = np.digitize(PV_npvs, df_PU['bins_left'].values)
    PU_SFs = df_PU['PU_SFs'][indexes-1].values
    return PU_SFs

def hw():
    print("HW")
    return 