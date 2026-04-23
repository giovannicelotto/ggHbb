from functions import getDfProcesses_v2, getCommonFilters
import numpy as np
import pandas as pd
from functions import getCommonFilters
FILTERS = [('is_ttbar_CR', '==', 1),
           ('dijet_mass','>',50),
            ('dijet_mass','<',300)
            ] 
def load_all_data(config):
    dfProcessesMC, dfProcessesData, _ = getDfProcesses_v2()

    modelName = config["modelName"]

    df_folder = config["df_folder"]
    columns = config["columns"]
    MConlyFeatures = config["MConlyFeatures"]

    # --- MC ---
    isMCList = []
    nMCs = []
    processes_groups = []
    for group_name, process in config["MC"].items():
        for k, v in process.items():
            isMCList.append(k)
            nMCs.append(v)
            processes_groups.append(group_name)



    dfsMC = []
    valid_mc = np.array(isMCList)[np.array(nMCs) != 0]
    for idx, p in zip(dfProcessesMC.index, dfProcessesMC.process):
        if idx not in valid_mc:
            continue
        print("Loading process %s..." % p)
        df = pd.read_parquet(
            f"{df_folder}/df_{p}_{modelName}.parquet",
            columns=columns + MConlyFeatures,
            filters=FILTERS if "HtoBB" not in p else None
        )
        dfsMC.append(df)
        dfsMC[-1]['process'] = processes_groups[isMCList.index(idx)]
        #print("Process idx %d, group %s"%(idx, processes_groups[isMCList.index(idx)]))

    # --- Data ---
    DataTakingList = list(config["dataPeriods"].keys())
    nReals = list(config["dataPeriods"].values())

    processesData = [
        dfProcessesData.index[dfProcessesData.process == "Data" + name].values[0]
        for name in DataTakingList
    ]

    dfsData, lumis = [], []

    valid_data = np.array(processesData)[np.array(nReals) != 0]

    for idx, p in enumerate(dfProcessesData.process):
        if idx not in valid_data:
            continue
        print("Loading data-taking period %s..." % p)
        df = pd.read_parquet(
            f"{df_folder}/dataframes_{p}_{modelName}.parquet",
            columns=columns,
            filters=FILTERS if "HtoBB" not in p else None
        )
        dfsData.append(df)

        lumi = np.load(f"{df_folder}/lumi_{p}_{modelName}.npy")
        lumis.append(lumi)

    lumi = np.sum(lumis)

    dfMC = pd.concat(dfsMC, ignore_index=True)
    dfData = pd.concat(dfsData, ignore_index=True)

    return dfMC, dfData, lumi