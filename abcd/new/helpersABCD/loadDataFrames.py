import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
import os
from functions import loadMultiParquet, getDfProcesses_v2, sortPredictions, loadMultiParquet_v2, getDfProcesses
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
def loadPredictions(processes, isMCList, predictionsFileNames, fileNumberList):
    preds = []
    predictionsFileNamesNew = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l =[]
        for fileName in predictionsFileNames[idx]:
            fn = int(re.search(r'FN(\d+)\.parquet', fileName).group(1))
            if fn in fileNumberList[idx]:
                l.append(fileName)
        predictionsFileNamesNew.append(l)

        print(len(predictionsFileNamesNew[idx]), " files for process")
        try:
            df = pd.read_parquet(predictionsFileNamesNew[idx])
        except:
            print("Error opening a file")
            for f in predictionsFileNamesNew[idx]:
                try:
                    df = pd.read_parquet(f)
                except:
                    os.remove(f)
            df = pd.read_parquet(predictionsFileNamesNew[idx])
        preds.append(df)
    return preds


def getPredictionNamesNumbers(processes,isMCList, predictionsPath):
    # Get predictions names path for all the datasets
    predictionsFileNames = []
    for p in processes:
        print("Process ", p)
        tempFileNames = glob.glob(predictionsPath+"/%s/*.parquet"%p)
        print("Predictions in ", predictionsPath+"/%s/*.parquet"%p)
        print(len(tempFileNames), " predictions found")
        #split the predictions fileNames based on the string yProcessName_ e.g. yData1A_
        #take the last part of the split. Will return FN%d.parquet
        # filter only the digits inside the string
        # convert the digits as a unique integer and use it to sort ascending
        sortedFileNames = sorted(tempFileNames, key=lambda x: int(''.join(filter(str.isdigit, x.split("y%s_"%p)[-1]))))
        predictionsFileNames.append(sortedFileNames)
        if len(sortedFileNames)==0:
            print("*"*10)
            print("No Files found for process ", p)
            print("*"*10)
    predictionsFileNumbers = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l = []
        for fileName in predictionsFileNames[idx]:
            fn = re.search(r'FN(\d+)\.parquet', fileName).group(1)
            l.append(int(fn))
        predictionsFileNumbers.append(l)

    return predictionsFileNames, predictionsFileNumbers




def loadDataFrames(nReal, nMC, predictionsPath, columns):
    # Define number of Data Files, MC files per process, predictionsPath, list of MC processes

    isMCList = [0, 1,
                2,
                3, 4, 5,
                6,7,8,9,10,11,
                12,13,14,
                15,16,17,18,19,
                20, 21, 22, 23, 36,
                #39    # Data2A
    ]

    # Take the DataFrame with processes, path, xsection. Filter the needed rows (processes)
    dfProcesses = getDfProcesses()
    processes = dfProcesses.process[isMCList].values

    # Put all predictions used for training in the proper folder. They will not be used here
    #sortPredictions(predictionsPath=predictionsPath)

    # Get predictions names path for all the datasets
    predictionsFileNames = []
    for p in processes:
        print(p)
        tempFileNames = glob.glob(predictionsPath+"/%s/others/*.parquet"%p)
        sortedFileNames = sorted(tempFileNames, key=lambda x: int(''.join(filter(str.isdigit, x))))
        print(len(sortedFileNames), "predictions available")
        predictionsFileNames.append(sortedFileNames)
        if len(predictionsFileNames)==0:
            print("*"*10)
            print("No Files found for process ", p)
            print("*"*10)
    print(sortedFileNames[:10])
    # For each fileNumber extract the fileNumber
    predictionsFileNumbers = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l = []
        for fileName in predictionsFileNames[idx]:
            print
            fn = re.search(r'FN(\d+)\.parquet', fileName).group(1)
            l.append(int(fn))
        predictionsFileNumbers.append(l)
    print("*"*50)
    print("FileNumbers", predictionsFileNumbers)
    print("*"*50)
    # Load flattuple for fileNumbers matching
    paths = list(dfProcesses.flatPath[isMCList])
    dfs= []
    print(predictionsFileNumbers)
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                          columns=columns,
                                                          returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                        returnFileNumberList=True)
    print("Lenght of dfs0", len(dfs[0]))
    if isMCList[-1]==39:
        nReal = nReal *2
        print("Duplicating nReal")

    preds = []
    predictionsFileNamesNew = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l =[]
        for fileName in predictionsFileNames[idx]:
            #print(fileName)
            fn = int(re.search(r'FN(\d+)\.parquet', fileName).group(1))
            if fn in fileNumberList[idx]:
                l.append(fileName)
        predictionsFileNamesNew.append(l)

        print(len(predictionsFileNamesNew[idx]), " files for process")
        df = pd.read_parquet(predictionsFileNamesNew[idx])
        preds.append(df)



    # preprocess for feeding the NN in order to have same cuts
    dfs = preprocessMultiClass(dfs=dfs)
    print("Length of dfs0 post process",len(dfs[0]))
    for idx, df in enumerate(dfs):
        print(idx)
        dfs[idx]['PNN'] = np.array(preds[idx])


    for idx, df in enumerate(dfs):
        isMC = isMCList[idx]
        print("isMC ", isMC)
        print("Process ", dfProcesses.process[isMC])
        print("Xsection ", dfProcesses.xsection[isMC])
        dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
     # make uinque data columns
    if isMCList[-1]==39:
        dfs[0]=pd.concat([dfs[0], dfs[-1]])
    # remove the last element (data2a)
        dfs = dfs[:-1]
    #set to 1 weights of data
    dfs[0]['weight'] = np.ones(len(dfs[0]))


    return dfs, isMCList, dfProcesses, nReal




