import numpy as np
import  pandas as pd
def getDataFrames(dfProcessesMC, dfProcessesData, isMCList, isDataList, modelName, df_folder, dd):
    # Load MC
    dfsMC = []
    for idx, p in enumerate(dfProcessesMC.process):
        if idx not in isMCList:
            continue
        df = pd.read_parquet(df_folder+"/df_dd_%s_%s.parquet"%(p, modelName))
        dfsMC.append(df)



    # Load Data
    dfsData = []
    lumis = []
    for idx, p in enumerate(dfProcessesData.process):
        if idx not in isDataList:
            continue
        df = pd.read_parquet(df_folder+"/dataframes%s%s_%s.parquet"%("_dd_" if dd else "", p, modelName))
        dfsData.append(df)
        lumi = np.load(df_folder+"/lumi%s%s_%s.npy"%("_dd_", p, modelName))
        lumis.append(lumi)
    lumi = np.sum(lumis)
    for idx, df in enumerate(dfsMC):
        dfsMC[idx].weight =dfsMC[idx].weight*lumi

    return dfsMC, dfsData, lumi
    