import pandas as pd
import glob
from functions import loadMultiParquet
import re

def loadDataAndPredictions(isMCList, predictionsPath, nReal, nMC):
    dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    processes = dfProcesses.process[isMCList].values

    # Get predictions names path for both datasets
    predictionsFileNames = []
    for p in processes:
        print(p)
        predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


    # %%
    predictionsFileNumbers = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l = []
        for fileName in predictionsFileNames[idx]:
            fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
            l.append(int(fn))

        predictionsFileNumbers.append(l)
    # %%
    paths = list(dfProcesses.flatPath[isMCList])
    dfs= []
    print(predictionsFileNumbers)
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                        columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt',
                                                                'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                                'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR',
                                                                    'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt',
                                                                'jet2_btagDeepFlavB', 'dijet_cs','leptonClass',
                                                                'jet1_btagDeepFlavB'],
                                                                returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                                returnFileNumberList=True)
    
    if isMCList[-1]==39:
        nReal = nReal *2
        print("Duplicating nReal")
    # %%
    preds = []
    predictionsFileNamesNew = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l =[]
        for fileName in predictionsFileNames[idx]:
            fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
            if fn in fileNumberList[idx]:
                l.append(fileName)
        predictionsFileNamesNew.append(l)
        
        print(len(predictionsFileNamesNew[idx]), " files for process")
        df = pd.read_parquet(predictionsFileNamesNew[idx])
        preds.append(df)


    return dfs, numEventsList, preds, dfProcesses
