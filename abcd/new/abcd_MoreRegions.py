# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet, getDfProcesses, sortPredictions, cut

from helpersABCD.loadDataFrames import loadDataFrames
from helpersABCD.getZdf import getZdf
from helpersABCD.plot import plotQCDhists_SR_CR, controlPlotABCD, plotQCDClosure, plotQCDPlusSM
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
columns = [ 'sf',           'dijet_mass',       'dijet_pt',             'jet1_pt',
          'jet2_pt',        'jet1_mass',        'jet2_mass',            'jet1_eta',
          'jet2_eta',       'dijet_dR',         'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'jet3_mass',      'Pileup_nTrueInt',  'dijet_cs',             'PU_SF',
          'leptonClass']

dfs, isMCList, dfProcesses = loadDataFrames(nReal=nReal, nMC=nMC, predictionsPath=predictionsPath, columns=columns)
for idx, df in enumerate(dfs):
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)
# %%
# save a copy of the dataframes before applying any cut
dfs_precut = dfs.copy()
# %%
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.3, max=None)

fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=101, log=True)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/dataMC_stacked.png")

# %%
# ABCD
# Variables for ABCD x1, x2 and variable for binning
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
xx = 'dijet_mass'
t1=0.6
t2 =0.4


# ABCD Start here
# Define 4 histograms for mjj, one for each region
bins = np.linspace(40, 300, 7)

dfs_lc1 = cut (data=dfs, feature='leptonClass', min=None, max=1.1)
dfs_lc2 = cut (data=dfs, feature='leptonClass', min=1.1, max=2.1)
dfs_lc3 = cut (data=dfs, feature='leptonClass', min=2.1, max=3.1)

dfs_lc1_cs0 = cut (data=dfs_lc1, feature='dijet_cs_abs', min=None, max=0.5)
dfs_lc2_cs0 = cut (data=dfs_lc2, feature='dijet_cs_abs', min=None, max=0.5)
dfs_lc3_cs0 = cut (data=dfs_lc3, feature='dijet_cs_abs', min=None, max=0.5)

dfs_lc1_cs1 = cut (data=dfs_lc1, feature='dijet_cs_abs', min=0.5, max=1)
dfs_lc2_cs1 = cut (data=dfs_lc2, feature='dijet_cs_abs', min=0.5, max=1)
dfs_lc3_cs1 = cut (data=dfs_lc3, feature='dijet_cs_abs', min=0.5, max=1)

ABCD(dfs_lc1_cs0, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc1_cs0')
ABCD(dfs_lc2_cs0, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc2_cs0')
ABCD(dfs_lc3_cs0, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc3_cs0')

ABCD(dfs_lc1_cs1, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc1_cs1')
ABCD(dfs_lc2_cs1, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc2_cs1')
ABCD(dfs_lc3_cs1, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc3_cs1')
# %%
