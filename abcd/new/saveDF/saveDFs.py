# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet_v2, loadMultiParquet_Data, getDfProcesses, sortPredictions, cut

from helpersABCD.loadDataFrames import loadDataFrames
from helpersABCD.getZdf import getZdf
from helpersABCD.createRootHists import createRootHists
from helpersABCD.abcd_maker import ABCD

sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from hist import Hist
import hist

# %%
nReal = 1000
nMC = -1
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
columns = ['dijet_pt',           'dijet_mass', 
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'dijet_cs',   'leptonClass',          
          'PU_SF', 'sf', 
          ]

dfs, isMCList, dfProcesses, nReal = loadDataFrames(nReal=nReal, nMC=nMC, predictionsPath=predictionsPath, columns=columns)
for idx, df in enumerate(dfs):
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)

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
with open('/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/oldDf/dataframes.pkl', 'wb') as f:
    pickle.dump(dfs, f)

#fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=101, nReal=nReal, log=True)
#fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/dataMC_stacked.png")
# %%
#with open('/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes.pkl', 'rb') as f:
#    dfs = pickle.load(f)
# %%
print(len(dfs[0]))
# %%
