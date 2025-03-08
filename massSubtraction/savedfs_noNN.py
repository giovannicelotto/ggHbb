# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re, os
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, getDfProcesses_v2, cut, getCommonFilters
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass

# %%
boosted = 22
modelName = "Lost_%d"%boosted
columns_ = ['dijet_mass', 'dijet_pt',
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'leptonClass',          
          'PU_SF', 'sf']
columns = columns_.copy()
dfProcessesMC, dfProcessesData = getDfProcesses_v2()

df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
if not os.path.exists(df_folder):
    os.makedirs(df_folder)
    
# Load data first
# %%
DataTakingList = [
            #0,  #1A
            1,   #2A
            2,  #1D
            3,
            #4,
            #5,
            #6
            ]
nReals = [
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1
]
lumi_tot = 0
processesData = dfProcessesData.process[DataTakingList].values
for dataTakingIdx, dataTakingName in zip(DataTakingList, processesData):
    print(dataTakingIdx+1,"/",len(DataTakingList))
    paths = list(dfProcessesData.flatPath[[dataTakingIdx]])
    if dataTakingName=='Data1A':
        paths[dataTakingIdx]=paths[dataTakingIdx]+"/others"


    dfs, lumi, fileNumberList = loadMultiParquet_Data_new(dataTaking=[dataTakingIdx], nReals=[nReals[dataTakingIdx]], columns=columns, returnFileNumberList=True, filters=getCommonFilters(btagTight=False))

    dfs=cut(dfs, 'dijet_pt', 160, None)
    dfs[0] = dfs[0][~((dfs[0].jet1_btagDeepFlavB>0.71) & (dfs[0].jet2_btagDeepFlavB>0.71))]
    lumi_tot = lumi_tot + lumi
    df = preprocessMultiClass(dfs=dfs)[0].copy()
    print("Length dfs0", len(df))
    del dfs

    print(df.columns)
    df.loc[:, 'weight'] = 1

    print("Saving ", dataTakingName)
    dfName = df_folder + "/dataframes_%s_%s.parquet"%(dataTakingName, modelName)
    lumiName = df_folder + "/lumi_%s_%s.npy"%(dataTakingName, modelName)
    try:
        df.to_parquet(dfName)
        print("SAved ", dfName)
    except:
        os.remove(dfName)
        df.to_parquet(dfName)
        print("SAved ", dfName)
    try:
        np.save(lumiName, lumi)
    except:
        os.remove(lumiName)
        np.save(lumiName, lumi)
    print("Luminosity Saved is ", lumi)



# %%
columns = columns_.copy()
isMCList = [#0,
            1, 
            ##2,
            3, 4,
            ##5,6,7,8, 9,10,
            ##11,12,13,
            ##14,15,16,17,18,
            19,20,21, 22,
            ##35,
            36,
            37
            ]
nMCs = [
    -1,
    -1,
    -1, -1, -1,
    -1, -1, -1, -1, -1, -1, 
    -1, -1, -1, 
    -1, -1, -1, -1, -1, 
    -1, -1, -1, -1, -1,
    -1
]
columns = columns + ['genWeight']
processesMC = dfProcessesMC.process[isMCList].values
for idx, (isMC, processMC) in enumerate(zip(isMCList, processesMC)):
    if nMCs[idx]==0:
        continue
    
    dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=nMCs[idx], columns=columns,
                                                             returnNumEventsTotal=True, returnFileNumberList=True)

    dfs=cut(dfs, 'dijet_pt', 160, None)
    dfs[0] = dfs[0][~((dfs[0].jet1_btagDeepFlavB>0.71) & (dfs[0].jet2_btagDeepFlavB>0.71))]
    df = preprocessMultiClass(dfs=dfs)[0].copy()
    print(df.columns)

    print("Process ", dfProcessesMC.process[isMC], " PNN assigned")
    print("Process ", dfProcessesMC.process[isMC])
    print("Xsection ", dfProcessesMC.xsection[isMC])
    df['weight'] = df.genWeight*df.PU_SF*df.sf*dfProcessesMC.xsection[isMC] * 1000/numEventsList[0]

    dataFrameName = df_folder + "/df_%s_%s.parquet"%(processMC, modelName)
    try:
        df.to_parquet(dataFrameName)
        print("Saved ", dataFrameName)
    except:
        os.remove(dataFrameName)
        df.to_parquet(dataFrameName)
        print("Saved ", dataFrameName)
# %%
