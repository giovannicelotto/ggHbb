import os, re
import pandas as pd
import glob, json
def load_mapping_dict(file_path):
    with open(file_path, 'r') as file:
        mapping_dict = json.load(file)
        # Convert keys back to integers if needed (they should already be integers)
        mapping_dict = {int(k): v for k, v in mapping_dict.items()}

    return mapping_dict

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
    realDataFileNames = glob.glob(realDataPath+"/*.parquet", recursive=False)
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
    

def loadMultiParquet(paths, nReal=1, nMC=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False):
    '''
    paths = array of string ordered with the first position occupied by the realData
    nReal = how many realdata files load  (-1=all the files, -2 no realData in the array)
    nMC = how many MC files load
    columns = which columns to read

    returnf
    dfs
    numEventsTotal = array of sum num events generated
    '''
    if 'Pileup_nTrueInt' not in columns:
        columns = list(columns)
        columns.append('Pileup_nTrueInt')
    dfs = []
    numEventsList = []
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    for path in paths: 
        # loop over processes
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)

        #if selectFileNumberList is not None then keep only strings where there is a match (want keep only files when i have predictions)
        if selectFileNumberList is not None:
            print("Looking for a specific list of ", len(selectFileNumberList[paths.index(path)]), " files expected")
            fileNamesSelectedProcess = []
            for fileName in fileNames:
                match = re.search(r'_(\d+).parquet', fileName)
                if match:
                    fn = match.group(1)
                if int(fn) in selectFileNumberList[paths.index(path)]:
                    fileNamesSelectedProcess.append(fileName)
                else:
                    pass
                    #print("remove", fileName)
            
            fileNamesSelected.append(fileNamesSelectedProcess)
            fileNames=fileNamesSelectedProcess
        print("Found %d files for process %d"%(len(fileNames), paths.index(path)))
        if paths.index(path)==0:
            if nReal>0:
                fileNames = fileNames[:nReal]
            if nReal == -2:
                fileNames = fileNames[:nMC] if nMC!=-1 else fileNames
            elif nReal == -1:
                pass
        else:
            fileNames = fileNames[:nMC] if nMC!=-1 else fileNames

        print("%d files for process %d" %(len(fileNames), paths.index(path)))
        #print("\n")
        
        df = pd.read_parquet(fileNames, columns=columns)
        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Sep.csv")
            for fileName in fileNames:
                filename = os.path.splitext(os.path.basename(fileName))[0]
                try:
                    process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    if returnFileNumberList:
                        fileNumberListProcess.append(fileNumber)
                    if process == "Data":
                        continue
                    numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsPassed.iloc[0]
                except:
                    process = '_'.join(filename.split('_')[:-1])  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    #print(process, fileNumber)
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
    PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
    for df in dfs:
        #print(df['Pileup_nTrueInt'])
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0
    

    if (returnNumEventsTotal) & (not returnFileNumberList):
        print("lenght of elements returned in fileNumberList")
        print(numEventsList)
        return dfs, numEventsList
    elif returnFileNumberList:
        print("lenght of elements returned in fileNumberList")
        print([len(el) for el in fileNumberList])
        return dfs, numEventsList, fileNumberList
    else:
        return dfs
    


def getXSectionBR():
    xSectionGGH = 48.61 # pb
    br = 0.5801
    return xSectionGGH*br
