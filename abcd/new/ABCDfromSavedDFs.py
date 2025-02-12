# %%
import pickle
import pandas as pd
from plotDfs import plotDfs
from functions import getDfProcesses, cut
import numpy as np
from helpersABCD.abcd_maker import ABCD
import argparse
# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
except:
    print("Interactive mode")
with open('/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes_%s.pkl'%modelName, 'rb') as f:
    dfs = pickle.load(f)
# %%
nReal = 963
isMCList = [0, 
                1,
                2,
                3, 4, 5,
                6,7,8,9,10,11,
                12,13,14,
                15,16,17,18,19,
                20, 21, 22, 23, 36,
                #39    # Data2A
    ]
if isMCList[-1]==39:
    nReal = nReal *2
    print("Duplicating nReal")
dfProcesses = getDfProcesses()
#fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses, nbin=101, nReal=nReal, log=True, blindPar=(True, 125, 20))
# %%


bin_edges = np.quantile(dfs[0].dijet_mass.values, np.linspace(0, 1, 7))

# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
xx = 'dijet_mass'
# tight WP for btag 0.7100
t1 = 0.71
t2 = 0.4
bins = np.load("/t3home/gcelotto/ggHbb/PNN/results/%s/mass_bins.npy"%modelName)
ABCD(dfs, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='inclusive_old_%s'%modelName, blindPar=(False, 125, 20))

# %%
dfs_lcPass = cut (data=dfs, feature='leptonClass', min=None, max=1.1)
dfs_lcFail = cut (data=dfs, feature='leptonClass', min=1.1, max=None)
ABCD(dfs_lcPass, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc1_%s'%modelName, blindPar=(False, 125, 20))
ABCD(dfs_lcFail, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lc0_%s'%modelName, blindPar=(False, 125, 20))
## %%
#dfs_lcFail_njets30mwp0 = cut (data=dfs_lcFail, feature='nJets_pt30_btag0p2', min=1.1, max=None)
#dfs_lcFail_njets30mwp2 = cut (data=dfs_lcFail, feature='nJets_pt30_btag0p2', min=None, max=1.1)
## %%
#dfs_lcFail_njets30mwp0_btagMedium = cut (data=dfs_lcFail_njets30mwp0, feature='jet2_btagDeepFlavB', min=0.71, max=None)
#dfs_lcFail_njets30mwp0_btagTight = cut (data=dfs_lcFail_njets30mwp0, feature='jet2_btagDeepFlavB', min=None, max=0.71)
#dfs_lcFail_njets30mwp2_btagMedium = cut (data=dfs_lcFail_njets30mwp2, feature='jet2_btagDeepFlavB', min=0.71, max=None)
#dfs_lcFail_njets30mwp2_btagTight = cut (data=dfs_lcFail_njets30mwp2, feature='jet2_btagDeepFlavB', min=None, max=0.71)
## %%
#ABCD(dfs_lcFail_njets30mwp0_btagMedium, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat0', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp0_btagTight, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat1', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp2_btagMedium, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat2', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp2_btagTight, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat3', blindPar=(True, 125, 20))
## %%
##dfs_lcFail_b2Tight = cut (data=dfs_lcFail, feature='jet2_btagDeepFlavB', min=0.71, max=None)
#dfs_lcFail_b2Med = cut (data=dfs_lcFail, feature='jet2_btagDeepFlavB', min=None, max=0.71)
# %%
#dfs_lcFail_b2Med_pt0to50 = cut (data=dfs_lcFail_b2Med, feature='dijet_pt', min=None, max=50)
#dfs_lcFail_b2Med_pt50to120 = cut (data=dfs_lcFail_b2Med, feature='dijet_pt', min=50, max=120)
#dfs_lcFail_b2Med_pt120toInf = cut (data=dfs_lcFail_b2Med, feature='dijet_pt', min=120, max=None)

#dfs_lcFail_b2Tight_pt0to50 = cut (data=dfs_lcFail_b2Tight, feature='dijet_pt', min=None, max=50)
#dfs_lcFail_b2Tight_pt50to120 = cut (data=dfs_lcFail_b2Tight, feature='dijet_pt', min=50, max=120)
#dfs_lcFail_b2Tight_pt120toInf = cut (data=dfs_lcFail_b2Tight, feature='dijet_pt', min=120, max=None)
# %%
#bins = np.linspace(40, 300, 8)
#ABCD(dfs_lcPass, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcPass',  blindPar=(True, 125, 20))

#bins = np.linspace(40, 300, 15)
#ABCD(dfs_lcFail_b2Med, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med')
# %%
#bins = np.linspace(40, 300, 25)
#ABCD(dfs_lcFail_b2Tight, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight')
# %%
#bins = np.linspace(40, 300, 15)
#ABCD(dfs_lcFail_b2Med_pt0to50, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_0to50')
#ABCD(dfs_lcFail_b2Med_pt50to120, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_50to120')
#ABCD(dfs_lcFail_b2Med_pt120toInf, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_120toInf')
#
#ABCD(dfs_lcFail_b2Tight_pt0to50, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_0to50')
#ABCD(dfs_lcFail_b2Tight_pt50to120, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_50to120')
#ABCD(dfs_lcFail_b2Tight_pt120toInf, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_120toInf')
# %%
#len(dfs[0][(dfs[0].jet1_btagDeepFlavB>0.71) & (dfs[0].PNN>0.4)])
# %%
#len(dfs[0])
# %%
#dfs[0].PNN.iloc[0]
# %%
