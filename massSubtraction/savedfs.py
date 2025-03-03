# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re, os
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, getDfProcesses_v2, sortPredictions, cut, getCommonFilters
from helpersABCD.loadDataFrames import getPredictionNamesNumbers, loadPredictions
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from hist import Hist

# %%
boosted = 2
modelName = "Feb27_%d_0p0"%boosted
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/mjjDiscoPred_%s"%modelName
columns = ['dijet_mass', 'dijet_pt',
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'leptonClass',          
          'PU_SF', 'sf']
dfProcessesMC, dfProcessesData = getDfProcesses_v2()

df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
if not os.path.exists(df_folder):
    os.makedirs(df_folder)
    
# Load data first
# %%
DataTakingList = [
            #0,  #1A
            #1,  #2A
            2,  #1D
            #3,
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

    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([dataTakingName],[dataTakingIdx], predictionsPath)
    # return ALL the available predictions fileNames and Numbers.
    # Predictions are ordered in increasing number
    # predictionsFileNumbers includes also training if not properly separated in a different folder.
    # I suppose training will be separated when matching the flattuple
    paths = list(dfProcessesData.flatPath[[dataTakingIdx]])
    if dataTakingName=='Data1A':
        paths[dataTakingIdx]=paths[dataTakingIdx]+"/training"


    dfs, lumi, fileNumberList = loadMultiParquet_Data_new(dataTaking=[dataTakingIdx], nReals=[nReals[dataTakingIdx]], columns=columns,
                                                          selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True, filters=getCommonFilters(btagTight=False))
    if boosted==1:
        dfs=cut(dfs, 'dijet_pt', 100, 160)
    elif boosted==2:
        dfs=cut(dfs, 'dijet_pt', 160, None)
    lumi_tot = lumi_tot + lumi
    predsData = loadPredictions(processesData, [dataTakingIdx], predictionsFileNames, fileNumberList)[0]
    df = preprocessMultiClass(dfs=dfs)[0].copy()
    print("Length dfs0", len(df))
    del dfs

    print(df.columns)
    df.loc[:, 'PNN'] = np.array(predsData.PNN)
    df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.71, max=None)[0].copy()
    df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.71, max=None)[0].copy()
    #print("Process ", dfProcessesData.process[isMC], " NN assigned")
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
isMCList = [0,
            1, 
            #2,
            3, 4,
            #5,6,7,8, 9,10,
            #11,12,13,
            #14,15,16,17,18,
            19,20,21, 22,
            #35,
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
    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
    
    dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=nMCs[idx], columns=columns,
                                                             returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                             returnFileNumberList=True)
    if boosted==1:
        dfs=cut(dfs, 'dijet_pt', 100, 160)
    elif boosted==2:
        dfs=cut(dfs, 'dijet_pt', 160, None)
    predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
    df = preprocessMultiClass(dfs=dfs)[0].copy()


        
    df['PNN'] = np.array(predsMC.PNN)
    print("Process ", dfProcessesMC.process[isMC], " PNN assigned")
    print("Process ", dfProcessesMC.process[isMC])
    print("Xsection ", dfProcessesMC.xsection[isMC])
    df['weight'] = df.genWeight*df.PU_SF*df.sf*dfProcessesMC.xsection[isMC] * 1000/numEventsList[0]


# save a copy of the dataframes before applying any cut
#dfs_precut = dfs.copy()

#dfs = dfs_precut.copy()
# 0.2783 WP for medium btagID
    df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.71, max=None)[0].copy()
    df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.71, max=None)[0].copy()
    dataFrameName = df_folder + "/df_%s_%s.parquet"%(processMC, modelName)
    try:
        df.to_parquet(dataFrameName)
        print("Saved ", dataFrameName)
    except:
        os.remove(dataFrameName)
        df.to_parquet(dataFrameName)
        print("Saved ", dataFrameName)
# %%
