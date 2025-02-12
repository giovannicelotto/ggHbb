# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet_v2, loadMultiParquet_Data, getDfProcesses, sortPredictions, cut
import argparse
from helpersABCD.loadDataFrames import loadDataFrames

# %%
nReal = 963
nMC = -1
parser = argparse.ArgumentParser(description="Script.")

# Define arguments
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
except:
    print("Interactive mode")
    modelName = "Jan08_250p0"


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_%s"%modelName
columns = ['dijet_pt',           'dijet_mass', 
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'dijet_cs',   'leptonClass',          
          'PU_SF', 'sf', 
          ]

dfs, isMCList, dfProcesses, nReal = loadDataFrames(nReal=nReal, nMC=nMC, predictionsPath=predictionsPath, columns=columns)
#for idx, df in enumerate(dfs):
#    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)

# %%
# save a copy of the dataframes before applying any cut
dfs_precut = dfs.copy()
# %%
dfs = dfs_precut.copy()
# 0.2783 WP for medium btagID
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.2783, max=None)
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.2783, max=None)
# %%
# Save the list of dataframes as pickle
import pickle
name = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes_%s.pkl'%modelName
import os
try:
    with open(name, 'wb') as f:
        pickle.dump(dfs, f)
except:
    os.remove(name)
    with open(name, 'wb') as f:
        pickle.dump(dfs, f)

#fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=101, nReal=nReal, log=True)
#fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/dataMC_stacked.png")
# %%
#with open('/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes.pkl', 'rb') as f:
#    dfs = pickle.load(f)
# %%
print(len(dfs[0]))
# %%
