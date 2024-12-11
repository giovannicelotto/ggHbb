# %%
import pickle
import pandas as pd
from plotDfs import plotDfs
from functions import getDfProcesses, cut, getDfProcesses_v2
import numpy as np
from helpersABCD.abcd_maker_v2 import ABCD
# %%
dfs = []
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
dfsMC = []
isMCList = [0,
            1, 
            2,3, 4,
            5,6,7,8, 9,10,
            11,12,13,
            14,15,16,17,18,
            19, 20,21, 22, 35]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/df_%s.parquet"%p)
    dfsMC.append(df)
# %%
dfsData = []
isDataList = [0,
            #1, 
            #2
            ]

lumis = []
for idx, p in enumerate(dfProcessesData.process):
    if idx not in isDataList:
        continue
    df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/dataframes%s.parquet"%p)
    dfsData.append(df)
    lumi = np.load("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/lumi%s.npy"%p)
    lumis.append(lumi)
lumi = np.sum(lumis)
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight =dfsMC[idx].weight*lumi

# %%
fig = plotDfs(dfsData=dfsData, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=lumi, log=True, blindPar=(True, 125, 20))
# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
xx = 'dijet_mass'
# tight WP for btag 0.7100
t1 = 0.7100
t2 = 0.4

bins=np.linspace(40, 300, 25)
ABCD(dfsData,dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='inclusive', blindPar=(True, 120, 20))

# %%
dfsData_lcPass = cut (data=dfsData, feature='leptonClass', min=None, max=1.1)
dfsMC_lcPass = cut (data=dfsMC, feature='leptonClass', min=None, max=1.1)
dfsData_lcFail = cut (data=dfsData, feature='leptonClass', min=1.1, max=None)
dfsMC_lcFail = cut (data=dfsMC, feature='leptonClass', min=1.1, max=None)

bins = np.linspace(40, 300, 25)
ABCD(dfsData_lcPass,dfsMC_lcPass,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='lcPass', blindPar=(True, 120, 20))
# %%

dfsData_lcFail_btagT = cut (data=dfsData, feature='jet2_btagDeepFlavB', min=0.71, max=None)
dfsData_lcFail_btagM = cut (data=dfsData, feature='jet2_btagDeepFlavB', min=None, max=0.71)
dfsMC_lcFail_btagT = cut (data=dfsMC, feature='jet2_btagDeepFlavB', min=0.71, max=None)
dfsMC_lcFail_btagM = cut (data=dfsMC, feature='jet2_btagDeepFlavB', min=None, max=0.71)

bins = np.linspace(40, 300, 20)
ABCD(dfsData_lcFail_btagT,dfsMC_lcFail_btagT,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='lcFail_btag2T', blindPar=(True, 120, 20))
bins = np.linspace(40, 300, 15)
ABCD(dfsData_lcFail_btagM,dfsMC_lcFail_btagM,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='lcFail_btag2M', blindPar=(True, 120, 20))
# %%
len(dfsData[0][(dfsData[0].jet1_btagDeepFlavB>0.71) & (dfsData[0].PNN>0.4)])
# %%
dfsData[0].PNN.iloc[0]









#dfs_lcFail_njets30mwp0_btagMedium = cut (data=dfs_lcFail_njets30mwp0, feature='jet2_btagDeepFlavB', min=0.71, max=None)
#dfs_lcFail_njets30mwp0_btagTight = cut (data=dfs_lcFail_njets30mwp0, feature='jet2_btagDeepFlavB', min=None, max=0.71)
#dfs_lcFail_njets30mwp2_btagMedium = cut (data=dfs_lcFail_njets30mwp2, feature='jet2_btagDeepFlavB', min=0.71, max=None)
#dfs_lcFail_njets30mwp2_btagTight = cut (data=dfs_lcFail_njets30mwp2, feature='jet2_btagDeepFlavB', min=None, max=0.71)
# %%
#ABCD(dfs_lcFail_njets30mwp0_btagMedium, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat0', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp0_btagTight, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat1', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp2_btagMedium, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat2', blindPar=(True, 125, 20))
#ABCD(dfs_lcFail_njets30mwp2_btagTight, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='cat3', blindPar=(True, 125, 20))
# %%
#dfs_lcFail_b2Tight = cut (data=dfs_lcFail, feature='jet2_btagDeepFlavB', min=0.71, max=None)
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
bins = np.linspace(40, 300, 15)
ABCD(dfs_lcFail_b2Med_pt0to50, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_0to50')
ABCD(dfs_lcFail_b2Med_pt50to120, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_50to120')
ABCD(dfs_lcFail_b2Med_pt120toInf, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Med_120toInf')

ABCD(dfs_lcFail_b2Tight_pt0to50, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_0to50')
ABCD(dfs_lcFail_b2Tight_pt50to120, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_50to120')
ABCD(dfs_lcFail_b2Tight_pt120toInf, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix='lcFail_b2Tight_120toInf')
# %%
