# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re, os
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, getDfProcesses_v2, sortPredictions, cut

from helpersABCD.loadDataFrames import getPredictionNamesNumbers, loadPredictions
from helpersABCD.getZdf import getZdf
from helpersABCD.createRootHists import createRootHists
from helpersABCD.abcd_maker import ABCD

sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from hist import Hist
import hist
import pickle
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
    modelName = "Mar05_700p1"
# %%
#modelName = "Jan24_900p0"
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_%s"%modelName
columns = ['dijet_mass', 
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'leptonClass',          
          'PU_SF', 'sf', 
          'muon_dxySig',
          'muon_pt',
          'dijet_pt',
          'dijet_eta',
          'ht',
          'jet1_nTightMuons',
          #'Muon_fired_HLT_Mu9_IP6',
          #'Muon_fired_HLT_Mu12_IP6'
          ]
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
if not os.path.exists(df_folder):
    os.makedirs(df_folder)


# Load data first
# %%
DataTakingList = [
            # 0, #1A
            1,   #2A
            2,   #1D
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
    #-1,
    #-1,
    #-1
]
lumi_tot = 0
processesData = dfProcessesData.process[DataTakingList].values
for idx, (dataTakingIdx, dataTakingName) in enumerate(zip(DataTakingList, processesData)):
    print(dataTakingIdx+1,"/",len(DataTakingList))

    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([dataTakingName],[dataTakingIdx], predictionsPath)
    # return ALL the available predictions fileNames and Numbers.
    # Predictions are ordered in increasing number
    # predictionsFileNumbers includes also training if not properly separated in a different folder.
    # I suppose training will be separated when matching the flattuple
    paths = list(dfProcessesData.flatPath[[dataTakingIdx]])
    if dataTakingName=='Data1A':
        paths[dataTakingIdx]=paths[dataTakingIdx]+"/others"


    dfs, lumi, fileNumberList = loadMultiParquet_Data_new(dataTaking=[dataTakingIdx], nReals=[nReals[idx]], columns=columns,
                                                          selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)
    dfs=cut(dfs, 'dijet_pt', None, 100)
    lumi_tot = lumi_tot + lumi
    predsData = loadPredictions(processesData, [dataTakingIdx], predictionsFileNames, fileNumberList)[0]
    df = preprocessMultiClass(dfs=dfs)[0].copy()
    print("Length dfs0", len(df))
    del dfs

    print(df.columns)
    df.loc[:, 'PNN1'] = np.array(predsData.PNN1)
    df.loc[:, 'PNN2'] = np.array(predsData.PNN2)
    df.loc[:, 'weight'] = 1

    print("Saving ", dataTakingName)
    dfName = df_folder+'/dataframes_dd_%s_%s.parquet'%(dataTakingName, modelName)
    lumiName = df_folder+"/lumi_dd_%s_%s.npy"%(dataTakingName, modelName)
    try:
        df.to_parquet(dfName)
    except:
        os.remove(dfName)
        df.to_parquet(dfName)
    try:
        np.save(lumiName, lumi)
    except:
        os.remove(lumiName)
        np.save(lumiName, lumi)
    print("Luminosity Saved is ", lumi)



# %%
isMCList = [
            #0,
            #1, 
            #2,3, 4,
            #5,6,7,8, 9,10,
            #11,12,13,
            #14,
            #15,16,17,18,
            #19,20,21, 22,
            ##23, 24, 25, 26, 27, 28, 
            ##29, 30, 31, 32, 33, 34,
            35,
            36,
            37
    #41
            ]
nMCs = [
    -1,
    -1,
    -1, -1, -1,
    -1, -1, -1, -1, -1, -1, 
    300, 300, 300, 
    -1, -1, -1, -1, -1, 
    -1, -1, -1, -1, -1,

# QCD
    #-1, -1,-1, -1, -1, -1, 
    #-1, -1, -1, -1, -1, -1,
# Z and Higgs
    -1,
    -1
#ggSpin200
    #-1
]

processesMC = dfProcessesMC.process[isMCList].values
for idx, (isMC, processMC) in enumerate(zip(isMCList, processesMC)):
    if nMCs[idx]==0:
        continue
    predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
    
    dfs, genEventSumwList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=-1, columns=columns+['genWeight'],
                                                             returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                             returnFileNumberList=True)

    dfs=cut(dfs, 'dijet_pt', None, 100)
    predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
    df = preprocessMultiClass(dfs=dfs)[0].copy()



        
    df['PNN1'] = np.array(predsMC.PNN1)
    df['PNN2'] = np.array(predsMC.PNN2)
    print("Process ", dfProcessesMC.process[isMC], " PNN assigned")
    print("Process ", dfProcessesMC.process[isMC])
    print("Xsection ", dfProcessesMC.xsection[isMC])
    df['weight'] = df.genWeight*df.PU_SF*df.sf*dfProcessesMC.xsection[isMC] * 1000/genEventSumwList[0]


# save a copy of the dataframes before applying any cut
#dfs_precut = dfs.copy()

#dfs = dfs_precut.copy()
# 0.2783 WP for medium btagID
    df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.2783, max=None)[0].copy()
    df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.2783, max=None)[0].copy()
    dataFrameName = df_folder+"/df_dd_%s_%s.parquet"%(processMC, modelName)
    try:
        print("Saving ", processMC, "\n\n")
        df.to_parquet(dataFrameName)
    except:
        os.remove(dataFrameName)
        df.to_parquet(dataFrameName)


# %%
