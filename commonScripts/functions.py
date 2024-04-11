import sys, os, re
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import glob
def cut(data, feature, min, max):
            newData = []
            for df in data:
                if min is not None:
                    df = df[df[feature] > min]
                if max is not None:
                    df = df[df[feature] < max]
                newData.append(df)
            return newData
def loadParquet(signalPath, realDataPath, nSignalFiles=-1, nRealDataFiles=1, columns=None, returnNumEventsTotal=False):

    signalFileNames = glob.glob(signalPath+"/*.parquet", recursive=True)
    realDataFileNames = glob.glob(realDataPath+"/*.parquet", recursive=True)
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    
    signal = pd.read_parquet(signalFileNames, columns=columns)
    realData = pd.read_parquet(realDataFileNames, columns=columns)
    if returnNumEventsTotal:
        numEventsTotal=0
        df = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
        for fileName in signalFileNames:
            filename = os.path.splitext(os.path.basename(fileName))[0]
            process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
            fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
            numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
        return signal, realData, numEventsTotal
    else:
        return signal, realData
    

def loadMultiParquet(paths, nReal=1, nMC=1, columns=None, returnNumEventsTotal=False, returnFileNumberList=False):
    '''
    paths = array of string ordered with the first position occupied by the realData
    nReal = how many realdata files load  (-1=all the files, -2 no realData in the array)
    nMC = how many MC files load
    columns = which columns to read

    return
    dfs
    numEventsTotal = array of sum num events generated
    '''
    dfs = []
    numEventsList = []
    if returnFileNumberList:
        fileNumberList = []
    for path in paths:
        if returnFileNumberList:
            fileNumberListProcess = []
        fileNames = glob.glob(path+"/*.parquet", recursive=True)
        if paths.index(path)==0:
            if nReal>0:
                fileNames = fileNames[:nReal]
            if nReal == -2:
                fileNames = fileNames[:nMC]
            elif nReal == -1:
                pass
        else:
            fileNames = fileNames[:nMC] if nMC!=-1 else fileNames

        print("%d files for process %d" %(len(fileNames), paths.index(path)))
        
    
        df = pd.read_parquet(fileNames, columns=columns)
        dfs.append(df)
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Mar.csv")
            for fileName in fileNames:
                filename = os.path.splitext(os.path.basename(fileName))[0]
                try:
                    process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    if returnFileNumberList:
                        fileNumberListProcess.append(fileNumber)
                    if process == "BParkingDataRun20181A":
                        continue
                    numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsPassed.iloc[0]
                except:
                    process = '_'.join(filename.split('_')[:2])  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    print(process, fileNumber)
                    if returnFileNumberList and fileNumberListProcess[-1]!=fileNumber:
                        fileNumberListProcess.append(fileNumber)
                    try:
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsPassed.iloc[0]
                    except:
                        print(fileNumber, process)
                        # execute compute mini to recompute the df into csv
                        # /t3home/gcelotto/ggHbb/computeMini.py
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsPassed.iloc[0]

                
            numEventsList.append(numEventsTotal)
        if returnFileNumberList:
            fileNumberList.append(fileNumberListProcess)

    if (returnNumEventsTotal) & (not returnFileNumberList):
        return dfs, numEventsList
    elif returnFileNumberList:
        return dfs, numEventsList, fileNumberList
    else:
        return dfs
    

def getPU_sfs(PV_npvs):
    df_PU = pd.read_csv("/t3home/gcelotto/ggHbb/PU_reweighting/output/pu_sfs.csv")
    indexes = np.digitize(PV_npvs, df_PU['bins_left'].values)
    PU_SFs = df_PU['PU_SFs'][indexes-1].values
    return PU_SFs

def getXSectionBR():
    xSectionGGH = 48.52 # pb
    br = 0.5801
    return xSectionGGH*br
