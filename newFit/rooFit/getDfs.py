import yaml
from functions import getDfProcesses_v2,  cut_advanced
import pandas as pd
import ROOT
import numpy as np
from array import array
import sys
def getDfs(idx):

    # Open config file and extract values
# Open config file and extract values
    config_path = "/t3home/gcelotto/ggHbb/WSFit/Configs/cat%d_bkg.yml"%(int(idx))
    # Open config file and extract values
    config_path_cuts = "/t3home/gcelotto/ggHbb/WSFit/Configs/cat%d.yml"%(int(idx))
    print("Opening config file ", config_path, "...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    x1 = config["x1"]
    x2 = config["x2"]
    key = config["key"]
    nbins = config["nbins"]
    t0 = config["t0"]
    t1 = config["t1"]
    t2 = config["t2"]
    t3 = config["t3"]
    isDataList = config["isDataList"]
    modelName = config["modelName"]
    plotFolder = config["plotFolder"]
    MCList_Z = config["MCList_Z"]
    MCList_H = config["MCList_H"]
    MCList_Z_sD = config["MCList_Z_sD"]
    MCList_H_sD = config["MCList_H_sD"]
    MCList_Z_sU = config["MCList_Z_sU"]
    MCList_H_sU = config["MCList_H_sU"]
    params = config["Bkg_params"][key]
    paramsLimits = config["Bkg_paramsLimits"]
    output_file = config["output_file"]
    fitZSystematics = config["fitZSystematics"]
    fitHSystematics = config["fitHSystematics"]

    with open(config_path_cuts, 'r') as f2:
        config_cuts = yaml.safe_load(f2)


    # Call the dataframes with names, xsections, paths of processes
    dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
    dfProcessesData = dfProcessesData.iloc[isDataList]


    # %%
    # Open Data
    dfsData = []
    lumi_tot = 0.
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
    for processName in dfProcessesData.process.values:
        print("Opening ", processName)
        df = pd.read_parquet(path+"/dataframes_%s_%s.parquet"%(processName, modelName))
        dfsData.append(df)
        lumi_tot = lumi_tot + np.load(path+"/lumi_%s_%s.npy"%(processName, modelName))

    # Open the MC

    dfsMC_Z = []
    for processName in dfProcessesMC.iloc[MCList_Z].process.values:
        print("Opening ", processName)
        df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
        dfsMC_Z.append(df)


    dfsMC_H = []
    for processName in dfProcessesMC.iloc[MCList_H].process.values:
        print("Opening ", processName)
        df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
        dfsMC_H.append(df)

    # %%
    # Normalize the MC to the luminosity
    for idx, df in enumerate(dfsMC_H):
        dfsMC_H[idx].weight=dfsMC_H[idx].weight*lumi_tot

    for idx, df in enumerate(dfsMC_Z):
        dfsMC_Z[idx].weight=dfsMC_Z[idx].weight*lumi_tot


    # %%
    # Apply cuts
    dfsData = cut_advanced(dfsData, config_cuts["cuts_string"])
    dfsMC_Z = cut_advanced(dfsMC_Z, config_cuts["cuts_string"])
    dfsMC_H = cut_advanced(dfsMC_H, config_cuts["cuts_string"])

    
    # Add label process in dfsMC to distinguish VBF and ggF contribution
    for idx, df in enumerate(dfsMC_H):
        dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
    for idx, df in enumerate(dfsMC_Z):
        dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process

    # Concatenate all the subprocesses
    dfMC_Z = pd.concat(dfsMC_Z)
    dfMC_H = pd.concat(dfsMC_H)
    df = pd.concat(dfsData)

    # %%



    return dfMC_Z, dfMC_H, df, nbins, x1, x2
