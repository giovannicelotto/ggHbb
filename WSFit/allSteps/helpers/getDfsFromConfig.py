import yaml
from functions import getDfProcesses_v2,  cut_advanced
import pandas as pd
import ROOT
import numpy as np
from array import array
import sys, re

def extract_pnn_edges(condition: str):
    pattern = r"PNN\s*([<>]=?)\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, condition)

    lower, upper = None, None
    for op, val in matches:
        val = float(val)
        if ">" in op:
            lower = val
        elif "<" in op:
            upper = val

    if upper is None:
        upper = 1.0  # default upper bound

    return lower, upper

def getDfsFromConfig(idx, return_nn=False, return_lumi=False):

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
    nbins = config["nbins"]
    isDataList = config["isDataList"]
    modelName = config["modelName"]
    MCList_Z = config["MCList_Z"]
    MCList_H = config["MCList_H"]

    with open(config_path_cuts, 'r') as f2:
        config_cuts = yaml.safe_load(f2)


    # Call the dataframes with names, xsections, paths of processes
    dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
    dfProcessesData = dfProcessesData.iloc[isDataList]


    # %%
    # Open Data
    dfsData = []
    lumi_tot = 0.
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/%s"%modelName
    process_names = list(dfProcessesData.process.values)

    print("\n\nOpening Data:", ", ".join(process_names))
    for processName in dfProcessesData.process.values:
        #print("Opening ", processName)
        df = pd.read_parquet(path+"/dataframes_%s_%s.parquet"%(processName, modelName), columns=[  "dijet_mass", "weight", "PNN","jet1_pt_uncor", "jet2_pt_uncor","dR_jet3_dijet",
                                                                                                 "jet2_pt", "jet1_eta", "jet2_eta", "jet1_muon_pt", "jet1_muon_eta", "dijet_pt", "jet1_btagDeepFlavB", "jet2_btagDeepFlavB"]) 
        dfsData.append(df)
        lumi_tot = lumi_tot + np.load(path+"/lumi_%s_%s.npy"%(processName, modelName))

    # Open the MC

    dfsMC_Z = []
    process_names = list(dfProcessesMC.iloc[MCList_Z].process.values)
    print("Opening Zbb:", ", ".join(process_names))
    for processName in dfProcessesMC.iloc[MCList_Z].process.values:
        #print("Opening ", processName)
        df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
        dfsMC_Z.append(df)


    dfsMC_H = []
    process_names = list(dfProcessesMC.iloc[MCList_H].process.values)
    print("Opening Hbb:", ", ".join(process_names), "\n\n")
    for processName in dfProcessesMC.iloc[MCList_H].process.values:
        #print("Opening ", processName)
        df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
        dfsMC_H.append(df)

    # %%
    # Normalize the MC to the luminosity
    for idx, df in enumerate(dfsMC_H):
        dfsMC_H[idx].weight=dfsMC_H[idx].weight*lumi_tot
        #if (0 in MCList_H) & (37 in MCList_H):
        #    print("GluGluPowheg and GluGLuMINLO both present, dividing their weights by 2 to avoid double counting")
        #    if ((MCList_H[idx]==0) | (MCList_H[idx]==37)):
        #        print("Weight sum before weight division for idx ", idx, " is ", dfsMC_H[idx].weight.sum())
        #        dfsMC_H[idx].weight=dfsMC_H[idx].weight/2
        #        print("Weight sum after weight division for idx ", idx, " is ", dfsMC_H[idx].weight.sum())
     # 13.7        

    for idx, df in enumerate(dfsMC_Z):
        dfsMC_Z[idx].weight=dfsMC_Z[idx].weight*lumi_tot


    # %%
    # Apply cuts
    dfsData = cut_advanced(dfsData, config_cuts["cuts_string"])
    dfsMC_Z = cut_advanced(dfsMC_Z, config_cuts["cuts_string"])
    dfsMC_H = cut_advanced(dfsMC_H, config_cuts["cuts_string"])
    dfsData = cut_advanced(dfsData, "(%d < dijet_mass) & (dijet_mass <= %d)"%(x1,x2))
    dfsMC_Z = cut_advanced(dfsMC_Z, "(%d < dijet_mass) & (dijet_mass <= %d)"%(x1,x2))
    dfsMC_H = cut_advanced(dfsMC_H, "(%d < dijet_mass) & (dijet_mass <= %d)"%(x1,x2))
    

    lower_NN, upper_NN = extract_pnn_edges(config_cuts["cuts_string"])

    
    # Add label process in dfsMC to distinguish VBF and ggF contribution
    for idx, df in enumerate(dfsMC_H):
        dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
    for idx, df in enumerate(dfsMC_Z):
        dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process

    # Concatenate all the subprocesses
    dfMC_Z = pd.concat(dfsMC_Z)
    dfMC_H = pd.concat(dfsMC_H)
    df = pd.concat(dfsData)

    if return_nn:
        return dfMC_Z, dfMC_H, df, nbins, x1, x2, lower_NN, upper_NN
    if return_lumi:
        return dfMC_Z, dfMC_H, df, nbins, x1, x2, lumi_tot
    else:
        return dfMC_Z, dfMC_H, df, nbins, x1, x2
