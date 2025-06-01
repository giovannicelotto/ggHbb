# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re, os
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, getDfProcesses_v2, sortPredictions, cut, getCommonFilters

from helpersABCD.loadDataFrames import getPredictionNamesNumbers, loadPredictions
from helpersABCD.getZdf import getZdf
from helpersABCD.createRootHists import createRootHists
from helpersABCD.abcd_maker import ABCD

sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
import argparse
# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    parser.add_argument("-MC", "--MC", type=int, help="MC (1) or Data (0)", default=0)
    parser.add_argument("-pN", "--processNumber", type=int, help="Number of dataTaking or process MC", default=0)
    parser.add_argument("-s", "--slurm", type=int, help="Process on slurm (1) or not (0)", default=0)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
except:
    print("Interactive mode")
    modelName = "Apr01_1000p0"
# %%

predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_%s"%modelName
columns = ['dijet_mass',    'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'muon_dxySig',    'muon_pt',              'dijet_pt']
dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
if args.slurm==0:
    df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
elif args.slurm==1:
    df_folder = "/scratch"
if not os.path.exists(df_folder):
    os.makedirs(df_folder)


# Load data first
# %%
if args.MC == 0:
    DataTakingList = [args.processNumber]
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


        dfs, lumi, fileNumberList = loadMultiParquet_Data_new(dataTaking=[dataTakingIdx], nReals=-1, columns=columns,
                                                                selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True, filters=getCommonFilters(btagTight=False))
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
        #df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.71, max=None)[0].copy()
        #df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.71, max=None)[0].copy()

        print("Saving ", dataTakingName)
        dfName = df_folder+'/dataframes_dd_%s_%s.parquet'%(dataTakingName, modelName)
        lumiName = df_folder+"/lumi_dd_%s_%s.npy"%(dataTakingName, modelName)
        if os.path.exists(dfName):
            os.remove(dfName)
        df.to_parquet(dfName)
        if os.path.exists(lumiName):
            os.remove(lumiName)
        np.save(lumiName, lumi)
        print("Luminosity Saved is ", lumi)


        if args.slurm:
            unique_id = args.processNumber
            output_txt = f"/scratch/output_path_{unique_id}.txt"
            
            with open(output_txt, "w") as f:
                f.write(dfName + "\n")
                f.write(lumiName + "\n")

elif args.MC == 1:

# %%
    isMCList = [args.processNumber]
    nMCs =[-1]


    processesMC = dfProcessesMC.process[isMCList].values
    for idx, (isMC, processMC) in enumerate(zip(isMCList, processesMC)):
        if nMCs[idx]==0:
            continue
        predictionsFileNames, predictionsFileNumbers = getPredictionNamesNumbers([processMC],[isMC], predictionsPath)
        
        dfs, genEventSumwList, fileNumberList = loadMultiParquet_v2(paths=[isMC], nMCs=-1, columns=
                                                                    columns+['genWeight', 'PU_SF', 'sf', 'btag_central', 'btag_up', 'btag_down'],
                returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                returnFileNumberList=True,
                filters=getCommonFilters(btagTight=False))

        dfs=cut(dfs, 'dijet_pt', None, 100)
        predsMC = loadPredictions([processMC], [isMC], predictionsFileNames, fileNumberList)[0]
        df = preprocessMultiClass(dfs=dfs)[0].copy()



            
        df['PNN1'] = np.array(predsMC.PNN1)
        df['PNN2'] = np.array(predsMC.PNN2)
        print("Process ", dfProcessesMC.process[isMC], " PNN assigned")
        print("Process ", dfProcessesMC.process[isMC])
        print("Xsection ", dfProcessesMC.xsection[isMC])
        df['weight'] = df.genWeight*df.PU_SF*df.sf* df.btag_central * dfProcessesMC.xsection[isMC] * 1000/genEventSumwList[0]


    # save a copy of the dataframes before applying any cut
    # 0.2783 WP for medium btagID
        #df = cut (data=[df], feature='jet2_btagDeepFlavB', min=0.71, max=None)[0].copy()
        #df = cut (data=[df], feature='jet1_btagDeepFlavB', min=0.71, max=None)[0].copy()

        dataFrameName = df_folder+"/df_dd_%s_%s.parquet"%(processMC, modelName)
        if os.path.exists(dataFrameName):
            os.remove(dataFrameName)
        print("Saving ", processMC, "\n\n")
        df.to_parquet(dataFrameName)

        if args.slurm:
            unique_id = args.processNumber
            output_txt = f"/scratch/output_path_{unique_id}.txt"
            
            with open(output_txt, "w") as f:
                f.write(dataFrameName + "\n")


# %%
