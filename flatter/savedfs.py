# %%
# Save the dataframes with the PNN scores assigned
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
import yaml
import argparse
# %%
cfg_file = "/t3home/gcelotto/ggHbb/flatter/savedfs.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)
# %%
parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-runMC", "--runMC", type=int, help="run MC 1 or 0", default=0)
parser.add_argument("-runData", "--runData", type=int, help="run data 1 or 0", default=0)
parser.add_argument("-period", "--period", type=str, help="Specific period to run (e.g. 1A). When this is set runData is set to 1 automatically and runMC is set to 0", default=None)


args = parser.parse_args()
if (args.runMC==0) & (args.runData==0):
    print("No option selected. Please select at least one of the options -runMC or -runData")
    sys.exit(0)
ttbar_CR = cfg["ttbar_CR"]
modelName = cfg["modelName"]
DataDict = cfg["dataPeriods"]
MCDict = cfg["MC"]
boosted = cfg["boosted"]
columns = cfg["columns"]
MConlyFeatures = cfg["MConlyFeatures"]
include_nn = cfg["include_nn"]
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/mjjDiscoPred_%s"%modelName
if args.period is not None:
    for key in DataDict:
        DataDict[key] = 0
        if key== args.period:
            DataDict[key] = -1
    print("Running only for period ", args.period)
    print(DataDict)
# %%
dfProcessesMC, dfProcessesData, dfProcessMC_JEC = getDfProcesses_v2()

if args.runMC:
    df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/%s"%modelName
if args.runData:
    df_folder = "/scratch"

if not os.path.exists(df_folder):
    os.makedirs(df_folder)
    
# Load data first
# %%

DataTakingList = list(DataDict.keys())
nReals = list(DataDict.values())
# %%
lumi_tot = 0
processesData = [dfProcessesData.index[dfProcessesData.process=="Data"+dataTakingName].values[0] for dataTakingName in DataTakingList]
# %%
if args.runData:
    for dataTakingName,dataTakingIdx in zip(DataTakingList, processesData):
        dataTakingIdx = dfProcessesData.index[dfProcessesData.process=="Data"+dataTakingName].values[0]
        print(dataTakingIdx+1,"/",len(DataTakingList), flush=True)
        if nReals[dataTakingIdx]==0:
            continue
        if include_nn:
            predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers(["Data"+dataTakingName],[dataTakingIdx], predictionsPath)
        else:
            predictionsFileNumbers=None
        # return ALL the available predictions fileNames and Numbers.
        # Predictions are ordered in increasing number
        # predictionsFileNumbers includes also training if not properly separated in a different folder.
        # I suppose training will be separated when matching the flattuple
        paths = list(dfProcessesData.flatPath[[dataTakingIdx]])
    #    if dataTakingName=='Data1A':
    #        paths[dataTakingIdx]=paths[dataTakingIdx]+"/training"

        print("ttbar_CR=", ttbar_CR)
        dfs, lumi, fileNumberList = loadMultiParquet_Data_new(  dataTaking=[dataTakingIdx],
                                                                nReals=nReals[dataTakingIdx],
                                                                columns=columns,
                                                                selectFileNumberList=predictionsFileNumbers,
                                                                returnFileNumberList=True,
                                                                filters=getCommonFilters(btagWP="L", cutDijet=True, ttbarCR=ttbar_CR))
        #if boosted==1:
        #    dfs=cut(dfs, 'dijet_pt', 100, 160)
        #elif boosted==2:
        #    dfs=cut(dfs, 'dijet_pt', 160, None)
        #elif boosted==3:
        #    dfs=cut(dfs, 'dijet_pt', 100, None)
        #elif boosted==60:
        #    dfs=cut(dfs, 'dijet_pt', 60, 100)
        dfs=cut(dfs, 'dijet_mass', 50, 300)
        lumi_tot = lumi_tot + lumi
        predsData = loadPredictions(processesData, [dataTakingIdx], predictionsFileNames, fileNumberList)[0]
        print("Predictions loaded", flush=True)
        #df = preprocessMultiClass(dfs=dfs)[0].copy()
        df = dfs[0].copy()
        #print("Length dfs0", len(df))
        del dfs

        #print(df.columns)
        df.loc[:, 'PNN'] = np.array(predsData.PNN)
        df.loc[:, 'PNN_pca'] = np.array(predsData.PNN_pca)
        df.loc[:,'PNN_qm'] = np.array(predsData.PNN_qm)
        #print("Process ", dfProcessesData.process[isMC], " NN assigned")
        df.loc[:, 'weight'] = 1

        print("Saving ", dataTakingName)
        dfName = df_folder + "/dataframes_Data%s_%s.parquet"%(dataTakingName, modelName)
        lumiName = df_folder + "/lumi_Data%s_%s.npy"%(dataTakingName, modelName)
        try:
            df.to_parquet(dfName)
            print("Saved ", dfName, flush=True)
        except:
            os.remove(dfName)
            df.to_parquet(dfName)
            print("Saved ", dfName)
        try:
            np.save(lumiName, lumi)
        except:
            os.remove(lumiName)
            np.save(lumiName, lumi)
        print("Luminosity Saved is ", lumi, flush=True) 



# %%

MC_dict = cfg["MC"]
isMCList = list(MC_dict.keys())
nMCs = list(MC_dict.values())
# %%
if args.runMC:
    print("MC starting")
    processesMC = dfProcessesMC.process[isMCList].values
    for idx, (isMC, processMC) in enumerate(zip(isMCList, processesMC)):
        print("\n ", idx+1,"/",len(processesMC), processMC, nMCs[idx])
        if nMCs[idx]==0:
            print("Skipping ", processMC)
            continue
        predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
        #if "ZJets" in processMC:
        #    columns_ = columns + ["NLO_kfactor"] 
        #else:
        #    columns_ = columns

        dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=nMCs[idx], columns=list(columns+MConlyFeatures),
                                                                returnNumEventsTotal=True, selectFileNumberList=None,
                                                                returnFileNumberList=True,
                                                                filters=getCommonFilters(btagWP="L", cutDijet=True, ttbarCR=ttbar_CR))


        dfs=cut(dfs, 'dijet_mass', 50, 300)
        
        
        
        predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
        #print(predsMC)

        df = dfs[0].copy()
        if len(df)!=len(predsMC):
            print("[ERROR] Length of df and predsMC do not match for process ", processMC)
            print("[ERROR] Length df ", len(df), " Length predsMC ", len(predsMC))
            continue
        df.loc[:,'PNN'] = np.array(predsMC.PNN)
        df.loc[:,'PNN_pca'] = np.array(predsMC.PNN_pca)
        df.loc[:,'PNN_qm'] = np.array(predsMC.PNN_qm)

        print("Process ", dfProcessesMC.process[isMC], " isMC :", isMC)
        #print("Xsection ", dfProcessesMC.xsection[isMC])
        df['weight'] = df.flat_weight * df.xsection * 1000/numEventsList[0]


        nan_values = df.isna().sum().sum()
        inf_values = np.isinf(df.weight).sum().sum()
        #print("Nan Values", nan_values)
        #print("Inf Values", inf_values)
        status = 1 if ((nan_values>0)  | (inf_values>0)) else 0
        #print("status", status)

        



    # save a copy of the dataframes before applying any cut
    #dfs_precut = dfs.copy()

    #dfs = dfs_precut.copy()

        #df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.0490, max=None)[0].copy()
        #df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.0490, max=None)[0].copy()
        dataFrameName = df_folder + "/df_%s_%s.parquet"%(processMC, modelName)
        #dataFrameName =  "/scratch/df_%s_%s.parquet"%(processMC, modelName)
        try:
            df.to_parquet(dataFrameName)
            print("Saved ", dataFrameName)
        except:
            os.remove(dataFrameName)
            df.to_parquet(dataFrameName)
            print("Saved ", dataFrameName)

