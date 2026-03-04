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
sys.path.append("/t3home/gcelotto/ggHbb/WSFit/allSteps/helpers")
from getDfsFromConfig import extract_pnn_edges



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', type=str, default="0", help='Category')
args = parser.parse_args([]) if hasattr(sys, 'ps1') or not sys.argv[1:] else parser.parse_args()


config_path_cuts = "/t3home/gcelotto/ggHbb/WSFit/Configs/cat%d.yml"%(int(args.category))
with open(config_path_cuts, 'r') as f2:
    config_cuts = yaml.safe_load(f2)
lower_NN, upper_NN = extract_pnn_edges(config_cuts["cuts_string"])


# %%
cfg_file = "/t3home/gcelotto/ggHbb/tt_CR/plot_tt_from_df.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)
# %%

modelName = cfg["modelName"]
DataDict = cfg["dataPeriods"]
MCDict = cfg["MC"]
boosted = cfg["boosted"]
columns = cfg["columns"]
MConlyFeatures = cfg["MConlyFeatures"]
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/mjjDiscoPred_%s"%modelName

# %%
dfProcessesMC, dfProcessesData, dfProcessMC_JEC = getDfProcesses_v2()

df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/%s"%modelName
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

# %%

MC_dict = cfg["MC"]
isMCList = list(MC_dict.keys())
nMCs = list(MC_dict.values())

# %%
processesMC = dfProcessesMC.process[isMCList].values


# %%
dfsData, dfsMC = [], []
lumis = []
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in np.array(isMCList)[np.array(nMCs)!=0]:
        continue
    print("[DEBUG] Trying to open %s"%(df_folder+"/df_%s_%s.parquet"%(p, modelName)))
    df = pd.read_parquet(df_folder+"/df_%s_%s.parquet"%(p, modelName), columns=columns+MConlyFeatures)
    dfsMC.append(df)
# %%
for idx, p in enumerate(dfProcessesData.process):
    if idx not in np.array(processesData)[np.array(nReals)!=0]:
        continue
    df = pd.read_parquet(df_folder+"/dataframes_%s_%s.parquet"%(p, modelName), columns=columns)
    print("/dataframes_%s_%s.parquet"%(p, modelName))

    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi_%s_%s.npy"%(p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['weight'] =dfsMC[idx].weight*lumi
print("Lumi total is %.2f fb-1"%lumi)
# %%
process_dict = {
    "ST": [5, 6, 7, 8, 9, 10],
    "tt(ll)": [11],
    "tt(others)": [12, 13],
    "diboson": [2, 3, 4],
    "V+Jets": [14,15,16,17,18, 19,20,21,22,35,49],
    "QCD": [23,24,25,26,27,28,29,30,31,32,33,34],
    "ggH(bb)": [37],
    #"WJets":[],
    #"DY":[],
}

id_to_process = {}
for proc, ids in process_dict.items():
    for i in ids:
        id_to_process[i] = proc
        
#%%
for idx, (mc_id, df) in enumerate(
    zip(np.array(isMCList)[np.array(nMCs) != 0], dfsMC)
):
    #print(idx, mc_id)
    dfsMC[idx]["process"] = id_to_process.get(mc_id, "other")
    #print(id_to_process.get(mc_id, "other"))

dfData = pd.concat(dfsData, ignore_index=True)
dfMC   = pd.concat(dfsMC, ignore_index=True)





# %%
import ROOT
import numpy as np

# output file
fout = ROOT.TFile(f"/t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/histograms_{args.category}.root", "RECREATE")

# histogram settings (adjust if needed)
n_bins = 5
x_min = lower_NN
x_max = upper_NN

processes = dfMC["process"].unique()

hists = {}

for proc in processes:
    if (proc == "ggH(bb)") | (proc == "other") | (proc == "V+Jets"):
        continue

    hist_name = f"h_{proc}"
    hist = ROOT.TH1F(hist_name, hist_name, n_bins, x_min, x_max)
    hist.Sumw2()  # VERY important for weighted histograms

    df_proc = dfMC[(dfMC["process"] == proc) & (dfMC.is_ttbar_CR == 1) & (dfMC.PNN_qm>=x_min) & (dfMC.PNN_qm<x_max)]

    values = df_proc["PNN_qm"].values
    weights = df_proc["weight"].values

    for x, w in zip(values, weights):
        hist.Fill(float(x), float(w))

    hists[proc] = hist

    hist.Write()
# --- Fill observed data histogram ---
hist_data = ROOT.TH1F("data_obs", "data_obs", n_bins, x_min, x_max)
# data is unweighted → just count events
values_data = dfData[(dfData.is_ttbar_CR==1) & (dfData.PNN_qm>=x_min) & (dfData.PNN_qm<x_max)]["PNN_qm"].values

for x in values_data:
    hist_data.Fill(float(x))

hist_data.Write()
fout.Close()

# %%
