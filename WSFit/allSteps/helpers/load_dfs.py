import yaml
import pandas as pd
import numpy as np
import re
from functions import getDfProcesses_v2, cut_advanced

BASE_CONFIG_PATH = "/t3home/gcelotto/ggHbb/WSFit/Configs"
BASE_DATA_PATH = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN"
COLUMNS = [
            "dijet_mass", "weight", "PNN",
            "jet1_pt_uncor", "jet2_pt_uncor", "dR_jet3_dijet",
            "jet2_pt", "jet1_eta", "jet2_eta",
            "jet1_muon_pt", "jet1_muon_eta",
            "dijet_pt", "jet1_btagDeepFlavB", "jet2_btagDeepFlavB"
        ]
def load_dfs(process_names, path, modelName, columns=None, prefix="df"):
    dfs = []
    for name in process_names:
        file_path = f"{path}/{prefix}_{name}_{modelName}.parquet"
        df = pd.read_parquet(file_path, columns=columns)
        dfs.append(df)
    return dfs
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dfs(process_names, path, model_name, columns=None, prefix="df"):
    dfs = []
    for name in process_names:
        file_path = f"{path}/{prefix}_{name}_{model_name}.parquet"
        df = pd.read_parquet(file_path, columns=columns)
        dfs.append(df)
    return dfs


def compute_lumi(process_names, path, model_name):
    lumi = 0.0
    for name in process_names:
        lumi += np.load(f"{path}/lumi_{name}_{model_name}.npy")
    return lumi


def normalize_mc(dfs, lumi):
    for df in dfs:
        df["weight"] *= lumi
    return dfs


def apply_common_cuts(dfs, cuts_string, x1, x2):
    dfs = cut_advanced(dfs, cuts_string)
    mass_cut = f"({x1} < dijet_mass) & (dijet_mass <= {x2})"
    dfs = cut_advanced(dfs, mass_cut)
    return dfs


def label_processes(dfs, process_names):
    for df, name in zip(dfs, process_names):
        df["process"] = name
    return dfs

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

import yaml
import pandas as pd
import numpy as np
import re
from functions import getDfProcesses_v2, cut_advanced

# -----------------------------
# Utilities
# -----------------------------

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

    return lower, upper if upper is not None else 1.0


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dfs(process_names, path, model_name, columns=None, prefix="df"):
    dfs = []
    for name in process_names:
        file_path = f"{path}/{prefix}_{name}_{model_name}.parquet"
        df = pd.read_parquet(file_path, columns=columns)
        dfs.append(df)
    return dfs


def compute_lumi(process_names, path, model_name):
    lumi = 0.0
    for name in process_names:
        lumi += np.load(f"{path}/lumi_{name}_{model_name}.npy")
    return lumi


def normalize_mc(dfs, lumi):
    for df in dfs:
        df["weight"] *= lumi
    return dfs


def apply_common_cuts(dfs, cuts_string, x1, x2):
    dfs = cut_advanced(dfs, cuts_string)
    mass_cut = f"({x1} < dijet_mass) & (dijet_mass <= {x2})"
    dfs = cut_advanced(dfs, mass_cut)
    return dfs


def label_processes(dfs, process_names):
    for df, name in zip(dfs, process_names):
        df["process"] = name
    return dfs


# -----------------------------
# Main function
# -----------------------------

def getDfsFromConfig(cat_idx, return_nn=False, return_lumi=False):

    # ---- Paths ----

    config_path = f"{BASE_CONFIG_PATH}/cat{int(cat_idx)}_bkg.yml"
    config_cuts_path = f"{BASE_CONFIG_PATH}/cat{int(cat_idx)}.yml"

    print(f"Opening config file {config_path}...")

    config = load_yaml(config_path)
    config_cuts = load_yaml(config_cuts_path)

    # ---- Extract config values ----
    x1, x2 = config["x1"], config["x2"]
    nbins = config["nbins"]
    nbins_MC = config["nbins_MC"]
    model_name = config["modelName"]

    isDataList = config["isDataList"]
    MCList_Z = config["MCList_Z"]
    MCList_H = config["MCList_H"]

    cuts_string = config_cuts["cuts_string"]

    # ---- Load process metadata ----
    dfMC_proc, dfData_proc, _ = getDfProcesses_v2()
    dfData_proc = dfData_proc.iloc[isDataList]

    data_names = list(dfData_proc.process.values)
    z_names = list(dfMC_proc.iloc[MCList_Z].process.values)
    h_names = list(dfMC_proc.iloc[MCList_H].process.values)

    path = f"{BASE_DATA_PATH}/{model_name}"

    print("Opening Data:", ", ".join(data_names))
    dfsData = load_dfs(
        data_names,
        path,
        model_name,
        columns=COLUMNS,
        prefix="dataframes"
    )

    lumi_tot = compute_lumi(data_names, path, model_name)

    print("Opening Zbb:", ", ".join(z_names))
    dfsMC_Z = load_dfs(z_names, path, model_name)

    print("Opening Hbb:", ", ".join(h_names))
    dfsMC_H = load_dfs(h_names, path, model_name)

    # ---- Normalize MC ----
    dfsMC_Z = normalize_mc(dfsMC_Z, lumi_tot)
    dfsMC_H = normalize_mc(dfsMC_H, lumi_tot)

    # ---- Apply cuts ----
    dfsData = apply_common_cuts(dfsData, cuts_string, x1, x2)
    dfsMC_Z = apply_common_cuts(dfsMC_Z, cuts_string, x1, x2)
    dfsMC_H = apply_common_cuts(dfsMC_H, cuts_string, x1, x2)

    # ---- Extract NN region ----
    lower_NN, upper_NN = extract_pnn_edges(cuts_string)

    # ---- Label processes ----
    dfsMC_Z = label_processes(dfsMC_Z, z_names)
    dfsMC_H = label_processes(dfsMC_H, h_names)

    # ---- Concatenate ----
    dfMC_Z = pd.concat(dfsMC_Z)
    dfMC_H = pd.concat(dfsMC_H)
    dfData = pd.concat(dfsData)

    # ---- Returns ----
    if return_nn:
        return dfMC_Z, dfMC_H, dfData, nbins, nbins_MC, x1, x2, lower_NN, upper_NN
    if return_lumi:
        return dfMC_Z, dfMC_H, dfData, nbins, nbins_MC, x1, x2, lumi_tot

    return dfMC_Z, dfMC_H, dfData, nbins, nbins_MC, x1, x2