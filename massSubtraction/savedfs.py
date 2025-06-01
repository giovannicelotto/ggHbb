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
boosted = 1
modelName = "Mar21_%d_0p0"%boosted
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/mjjDiscoPred_%s"%modelName
columns_ = ['dijet_mass', 'dijet_pt',
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB']
columns = columns_.copy()
dfProcessesMC, dfProcessesData, dfProcessMC_JEC = getDfProcesses_v2()

df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
if not os.path.exists(df_folder):
    os.makedirs(df_folder)
    
# Load data first
# %%
DataDict = {
0 : 0,         
1 : 0,         2 : 0,         3 : 0,         4 : 0,         5 : 0,
6 : 0,         7 : 0,         8 : 0,         9 : 0,         10 : 0,        11 : 0,
12 : 0,        13 : 0,        14 : 0,        15 : 0,        16 : 0,        17 : 0, 
18 : 0, 
}

DataTakingList = list(DataDict.keys())
nReals = list(DataDict.values())

lumi_tot = 0
processesData = dfProcessesData.process[DataTakingList].values
for dataTakingIdx, dataTakingName in zip(DataTakingList, processesData):
    print(dataTakingIdx+1,"/",len(DataTakingList))
    if nReals[dataTakingIdx]==0:
        continue

    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([dataTakingName],[dataTakingIdx], predictionsPath)
    # return ALL the available predictions fileNames and Numbers.
    # Predictions are ordered in increasing number
    # predictionsFileNumbers includes also training if not properly separated in a different folder.
    # I suppose training will be separated when matching the flattuple
    paths = list(dfProcessesData.flatPath[[dataTakingIdx]])
    if dataTakingName=='Data1A':
        paths[dataTakingIdx]=paths[dataTakingIdx]+"/training"


    dfs, lumi, fileNumberList = loadMultiParquet_Data_new(dataTaking=[dataTakingIdx], nReals=nReals[dataTakingIdx], columns=columns,
                                                          selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True, filters=getCommonFilters(btagTight=False))
    if boosted==1:
        dfs=cut(dfs, 'dijet_pt', 100, 160)
    elif boosted==2:
        dfs=cut(dfs, 'dijet_pt', 160, None)
    elif boosted==60:
        dfs=cut(dfs, 'dijet_pt', 60, 100)
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
columns = columns_.copy()
MC_dict = {
    #Nominal
    0:0, 1:0, 3:0, 4:0, 19:0, 20:0, 21:0, 22:0, 35:0,
    36:0, 37:0,43:0,
    #JER Down
    44:0, 45:0, 46:0, 47:0, 48:0, 49:0, 50:0, 51:0, 52:0, 53:0,54:-1, 55:0,
    #JER Up
    56:0, 57:0, 58:0, 59:0, 60:0, 61:0, 62:0, 63:0, 64:0, 65:0, 66:-1, 67:0
}
isMCList = list(MC_dict.keys())
nMCs = list(MC_dict.values())


columns = columns + ['genWeight', 'PU_SF', 'sf',
                     'jet1_pt', #'jet1_eta', 'jet1_phi', 'jet1_mass',
                     'jet2_pt', #'jet2_eta', 'jet2_phi', 'jet2_mass',
                     'btag_central', 'btag_up', 'btag_down',
                        ]

processesMC = dfProcessesMC.process[isMCList].values
for idx, (isMC, processMC) in enumerate(zip(isMCList, processesMC)):
    if nMCs[idx]==0:
        continue
    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
    
    dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=nMCs[idx], columns=columns,
                                                             returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                             returnFileNumberList=True)
    print(numEventsList[0])
    if boosted==1:
        dfs=cut(dfs, 'dijet_pt', 100, 160)
    elif boosted==2:
        dfs=cut(dfs, 'dijet_pt', 160, None)
    elif boosted==60:
        dfs=cut(dfs, 'dijet_pt', 60, 100)
    predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
    df = preprocessMultiClass(dfs=dfs)[0].copy()


        
    df['PNN'] = np.array(predsMC.PNN)
    print("Process ", dfProcessesMC.process[isMC], " PNN assigned")
    print("Process ", dfProcessesMC.process[isMC])
    print("Xsection ", dfProcessesMC.xsection[isMC])
    df['weight'] = df.genWeight * df.PU_SF * df.sf * df.btag_central * dfProcessesMC.xsection[isMC] * 1000/numEventsList[0]


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




##
##      JEC Varied dataset
##


columns = columns_.copy()
MC_JEC_keys = [_ for _ in range(0, 0)]
nRealsMC_JEC_values = [int(-1) for _ in range(len(MC_JEC_keys))]
isMCJECList = list(MC_JEC_keys)
nMCs = list(nRealsMC_JEC_values)



columns = columns + ['genWeight', 'PU_SF', 'sf',
                     'jet1_pt', 
                     'jet2_pt', 
                     'btag_central', 'btag_up', 'btag_down',
                        ]

processesMC = dfProcessMC_JEC.process[isMCJECList].values
for idx, (isMC, processMC) in enumerate(zip(isMCJECList, processesMC)):
    print("\n\n",idx, "/", len(isMCJECList))
    #if "Data" not in processMC:
    #    continue
    try:
        if nMCs[idx]==0:
            continue
        predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
        dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=nMCs[idx], columns=columns,
                                                                returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                                returnFileNumberList=True, isJEC=1)
        print(numEventsList[0])
        if boosted==1:
            dfs=cut(dfs, 'dijet_pt', 100, 160)
        elif boosted==2:
            dfs=cut(dfs, 'dijet_pt', 160, None)
        elif boosted==60:
            dfs=cut(dfs, 'dijet_pt', 60, 100)
        predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
        df = preprocessMultiClass(dfs=dfs)[0].copy()



        df['PNN'] = np.array(predsMC.PNN)
        print("Process ", dfProcessMC_JEC.process[isMC], " PNN assigned")
        print("Process ", dfProcessMC_JEC.process[isMC])
        print("Xsection ", dfProcessMC_JEC.xsection[isMC])
        df['weight'] = df.genWeight * df.PU_SF * df.sf * df.btag_central * dfProcessMC_JEC.xsection[isMC] * 1000/numEventsList[0]


    # save a copy of the dataframes before applying any cut
    #dfs_precut = dfs.copy()

    #dfs = dfs_precut.copy()
    # 0.2783 WP for medium btagID
        df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.71, max=None)[0].copy()
        df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.71, max=None)[0].copy()
        dataFrameName = df_folder + "/df_%s_%s.parquet"%(processMC, modelName)
        try:
            df.to_parquet(dataFrameName)
            print("Saved ", dataFrameName, "\n\n\n")
        except:
            os.remove(dataFrameName)
            df.to_parquet(dataFrameName)
            print("Saved ", dataFrameName, "\n\n\n")
    except:
        continue
# %%
