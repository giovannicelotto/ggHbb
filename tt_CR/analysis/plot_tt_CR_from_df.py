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
# %%
cfg_file = "/t3home/gcelotto/ggHbb/tt_CR/analysis/plot_tt_from_df.yaml"
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
print(columns)
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
    df['dijet_dEta'] =df.jet1_eta - df.jet2_eta

    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi_%s_%s.npy"%(p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['weight'] =dfsMC[idx].weight*lumi
    dfsMC[idx]['dijet_dEta'] =dfsMC[idx].jet1_eta - dfsMC[idx].jet2_eta
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

#%%

from binning_per_variable import plot_vars
# %%
def plot_data_mc_stack_ratio(dfData, dfMC, var, bins, xlabel):
    fig, (ax, rax) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # -----------------
    # Main (Data + MC)
    # -----------------
    mask_data = dfData.is_ttbar_CR == 1
    data_vals = dfData.loc[mask_data, var]
    data_counts, _ = np.histogram(data_vals, bins=bins)

    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        data_counts,
        yerr=np.sqrt(data_counts),
        marker="o",
        linestyle="none",
        color="black",
        label="Data",
    )

    mask_mc = dfMC.is_ttbar_CR == 1
    processes = list(process_dict.keys())

    mc_arrays = []
    mc_weights = []
    mc_processes = []

    for proc in processes:
        if proc == "ggH(bb)":
            continue
        sel = (mask_mc) & (dfMC.process == proc)
        mc_arrays.append(dfMC.loc[sel, var])
        mc_weights.append(dfMC.loc[sel, "weight"])
        mc_processes.append(proc)

    mc_counts = np.histogram(
        np.concatenate(mc_arrays),
        bins=bins,
        weights=np.concatenate(mc_weights),
    )[0]

    ax.hist(
        mc_arrays,
        bins=bins,
        weights=mc_weights,
        stacked=True,
        label=processes,
    )
    if "PNN" in var:
        var_ggH = "PNN"
    else:
        var_ggH = var
    ax.hist(
        dfMC[dfMC.process=="ggH(bb)"][var_ggH],
        bins=bins,
        weights=dfMC.weight[dfMC.process=="ggH(bb)"],
        histtype="step",
        color='red',
        linewidth=3,
        label="ggH(bb)",
    )
    ax.set_ylabel("Events")
    ax.set_ylim(0, 1.4 * max(data_counts.max(), mc_counts.max()))
    ax.legend(loc='best')

    # -----------------
    # Ratio
    # -----------------
    ratio = np.divide(
        data_counts,
        mc_counts,
        out=np.zeros_like(data_counts, dtype=float),
        where=mc_counts > 0,
    )

    ratio_err = np.divide(
        np.sqrt(data_counts),
        mc_counts,
        out=np.zeros_like(data_counts, dtype=float),
        where=mc_counts > 0,
    )

    centers = (bins[1:] + bins[:-1]) / 2

    rax.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        marker="o",
        linestyle="none",
        color="black",
    )
    mc_sumw2 = np.histogram(
    np.concatenate(mc_arrays),
    bins=bins,
    weights=np.concatenate(mc_weights) ** 2,
    )[0]

    mc_unc = np.sqrt(mc_sumw2)
    ratio_mc_unc = np.divide(   mc_unc, mc_counts,
                                out=np.zeros_like(mc_unc, dtype=float), where=mc_counts > 0)
    y_low = 1 - ratio_mc_unc
    y_high = 1 + ratio_mc_unc

    # Extend arrays to match bin edges
    y_low = np.r_[y_low, y_low[-1]]
    y_high = np.r_[y_high, y_high[-1]]

    rax.fill_between(
                        bins, y_low, y_high,
                        step="post",color="gray",alpha=0.4,linewidth=0,
    )
    rax.axhline(1.0, linestyle="--", linewidth=1)
    rax.set_ylim(0.5, 1.5)
    ax.set_xlim(bins[0], bins[-1])
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(xlabel)

    # Align y-labels
    fig.align_ylabels([ax, rax])
    hep.cms.label(lumi=np.round(lumi,2), ax=ax)

    return fig, (ax, rax), mc_arrays, mc_weights



# %%
for var, cfg in plot_vars.items():
    if var in columns or var == "dijet_dEta":
        pass
    else:
        #print("[WARNING] Variable %s not found in dataframe columns, skipping..."%var)
        continue
    print(var," plotting...")
    fig, (ax, rax), mc_arrays, mc_weights = plot_data_mc_stack_ratio(
        dfData,
        dfMC,
        var=var,
        bins=cfg["bins"],
        xlabel=cfg["xlabel"],
    )
    fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/ttbar_CR/%s.png"%(var), bbox_inches="tight")
    plt.close('all')
    del fig, ax, rax


# %%
