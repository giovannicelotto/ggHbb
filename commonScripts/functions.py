import os, re
import pandas as pd
import glob, json

import os
import glob
import shutil

def getDfProcesses_v2 (reset=False):
    if reset:
        # rerun the script to have the csv file with new table
        from processes import getProcessesDataFrame
        getProcessesDataFrame()
    dfProcessesMC = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv")
    dfProcessesData = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
    return dfProcessesMC, dfProcessesData
def getCommonFilters():
    filters = [
                ('jet1_pt', '>',  20),
                ('jet2_pt', '>',  20),
                
                ('jet1_mass', '>', 0),
                ('jet2_mass', '>', 0),
                ('jet3_mass', '>',  0),
                
                ('jet1_eta', '>', -2.5),
                ('jet2_eta', '>', -2.5),
                ('jet1_eta', '<',  2.5),
                ('jet2_eta', '<',  2.5)]
    return filters

def loadMultiParquet_v2(paths, nMCs=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters()):
    '''
    paths = array of string 
            or list of numbers (integers) with the isMC number
    nMC = how many MC files load (:int --> nFiles per process ) (:list --> one number per process)
    columns = which columns to read

    return
    dfs = list of dfs of MC
    numEventsTotal = array of sum num events generated e.g. [1302, 5403, ...]
    fileNumberList = list of filenumbers for each process e.g.[[1, 2, 3], [1, 2, 3, 4], ...]
    '''
    #if columns is not None:
    #    if 'Pileup_nTrueInt' not in columns:
    #        columns = list(columns)
    #        columns.append('Pileup_nTrueInt')
    dfs = []
    numEventsList = []
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    # Check types are as expected
    if isinstance(paths[0], str):
        pass
    elif isinstance(paths[0], int):
        paths = paths.copy()
        for idx, isMCnumber in enumerate(paths):
            df_processesMC = getDfProcesses_v2()[0]
            paths[idx] = df_processesMC.flatPath[isMCnumber]


    if isinstance(nMCs, int):
        nMCs = [nMCs for i in range(len(paths))]
    elif isinstance(nMCs, list):
        if len(nMCs)==len(paths):
            pass
        else:
            print("Error, number of nMCs requested is different from the number of processes. Setting nMCs equal to the first instance")
            nMCs = [nMCs[0] for i in range(len(paths))]

    for nMC, path in zip(nMCs, paths): 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
        # loop over processes
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)

        #if selectFileNumberList is not None then keep only strings where there is a match (want keep only files when i have predictions)
        if selectFileNumberList is not None:
            print("Looking for a specific list of ", len(selectFileNumberList[paths.index(path)]), " files for which predictions is available")
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
        fileNames = fileNames[:nMC] if nMC!=-1 else fileNames
        eg = os.path.basename(fileNames[0])
        match = re.match(r'(.+)_([0-9]+)\.parquet', eg)
        process = match.group(1)

        print("Found %d files for process %s"%(len(fileNames), process))
        del match, eg
        

        print("%d files for process %s" %(len(fileNames), process))
        #print("\n")
        
        df = pd.read_parquet(fileNames, columns=columns,
                                 engine='pyarrow',
                                 filters= filters        )

        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Oct_new.csv")
            for fileName in fileNames:
                #remove the path to file: file_name_322.parquet
                filename = os.path.basename(fileName)
                try:
                    #print(filename)

                    match = re.match(r'(.+)_([0-9]+)\.parquet', filename)
                    #(.+)_([0-9]+)\.parquet: The regular expression captures everything up to the last underscore as the file name ((.+)), followed by a series of digits (([0-9]+)) and ending with .parquet
                    process = match.group(1)
                    fileNumber = int(match.group(2))
                    #print("Process ", process)
                    #print("fileNumber ", fileNumber)
                    #process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    #fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    if returnFileNumberList:
                        fileNumberListProcess.append(fileNumber)
                    numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
                except:
                    process = '_'.join(filename.split('_')[:-1])  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    #print(process, fileNumber)
                    if returnFileNumberList and fileNumberListProcess[-1]!=fileNumber:
                        fileNumberListProcess.append(fileNumber)
                    try:
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
                    except:
                        print(fileNumber, process)
                        # execute compute mini to recompute the df into csv
                        # /t3home/gcelotto/ggHbb/computeMini.py
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]

                
            numEventsList.append(numEventsTotal)
        if returnFileNumberList:
            fileNumberList.append(fileNumberListProcess)


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
    


def loadMultiParquet_Data(paths, nReals=1, columns=None, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters()):
    '''
    paths = array of string 
            or list of numbers (integers) with the process number
    nReals = how many files load (:int --> nFiles per data taking ) (:list --> one number per data taking)
    columns = which columns to read

    return
    dfs = list of dfs of MC
    fileNumberList = list of filenumbers for each process e.g.[[1, 2, 3], [1, 2, 3, 4], ...]
    '''

    dfs = []
    df_processesData = getDfProcesses_v2()[1]
    lumi_tot = 0
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    # Check types are as expected
    if isinstance(paths[0], str):
        pass
    elif isinstance(paths[0], int):
        for idx, isMCnumber in enumerate(paths):
            paths[idx] = df_processesData.flatPath[isMCnumber]


    if isinstance(nReals, int):
        nReals = [nReals for i in range(len(paths))]
    elif isinstance(nReals, list):
        if len(nReals)==len(paths):
            pass
        else:
            print("Error, number of nReals requested is different from the number of processes. Setting nReals equal to the first instance")
            nReals = [nReals[0] for i in range(len(paths))]

    for nReal, path, lumi, nFiles in zip(nReals, paths, df_processesData.lumi, df_processesData.nFiles): 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
        # loop over processes
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)
        nReal = len(fileNames) if nReal == -1 else nReal

        #if selectFileNumberList is not None then keep only strings where there is a match (want keep only files when i have predictions)
        if selectFileNumberList is not None:
            if nReal >= len(selectFileNumberList[paths.index(path)]):
                nReal = len(selectFileNumberList[paths.index(path)])
            elif nReal == -1:
                nReal = len(selectFileNumberList[paths.index(path)])
            else:
                #print(len(selectFileNumberList[paths.index(path)]), " length before")
                selectFileNumberList[paths.index(path)] = selectFileNumberList[paths.index(path)][:nReal]
                #print(len(selectFileNumberList[paths.index(path)]), " length after")
            print("Looking for a specific list of ", len(selectFileNumberList[paths.index(path)]), " files for which predictions is available")
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
        fileNames = fileNames[:nReal] if nReal!=-1 else fileNames
        lumi_tot = lumi_tot + nReal*lumi/nFiles
        try:
            eg = os.path.basename(fileNames[0])
        except:
            print(fileNames[0])
        match = re.match(r'(.+)_([0-9]+)\.parquet', eg)
        process = match.group(1)

        print("Found %d files for process %s"%(len(fileNames), process))
        del match, eg
        

        print("%d files for process %s" %(len(fileNames), process))
        #print("\n")
        
        df = pd.read_parquet(fileNames, columns=columns,
                                 engine='pyarrow',
                                 filters= filters        )

        dfs.append(df)
        print("Lenght dfs0", len(dfs[0]))
    #return dfs
        if returnFileNumberList:
            fileNumberListProcess = []
            for fileName in fileNames:
                #remove the path to file: file_name_322.parquet
                filename = os.path.basename(fileName)
                try:
                    match = re.match(r'(.+)_([0-9]+)\.parquet', filename)
                    fileNumber = int(match.group(2))
                    fileNumberListProcess.append(fileNumber)
                except:
                    print("Something wrong here")
            fileNumberList.append(fileNumberListProcess)




    if returnFileNumberList:
        print("lenght of elements returned in fileNumberList")
        print([len(el) for el in fileNumberList])
        return dfs, lumi_tot, fileNumberList
    else:
        return dfs, lumi_tot
    
def sortFlatFiles():
    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"

    subfolders = {
        "Data1A": {
            "others": list(range(10, 1018)),
            "training": list(range(1, 10))
        },
        "GluGluHToBB": {
            "others": list(range(46, 200)),
            "training": list(range(1, 46))
        }
    }

    for category, folders in subfolders.items():
        for folder, expected_files in folders.items():
            folder_path = os.path.join(flatPath, category, folder)
            
            # Ensure the folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Get existing files in the folder
            files_in_folder = glob.glob(f"{folder_path}/*.parquet")
            
            # Extract the numeric parts of filenames
            present_numbers = set(
                int(os.path.basename(file).split('_')[-1].split('.')[0])
                for file in files_in_folder
            )
            
            # Check for missing files
            expected_set = set(expected_files)
            missing_files = expected_set - present_numbers
            
            if missing_files:
                print(f"Missing files in {folder_path}: {sorted(missing_files)}")
            
            # Check for files in the wrong folder and move them
            other_folder = "training" if folder == "others" else "others"
            other_folder_path = os.path.join(flatPath, category, other_folder)
            files_to_move = [
                f for f in glob.glob(f"{other_folder_path}/*.parquet")
                if int(os.path.basename(f).split('_')[-1].split('.')[0]) in expected_files
            ]
            
            for file_path in files_to_move:
                shutil.move(file_path, folder_path)
                print(f"Moved {os.path.basename(file_path)} to {folder_path}")

import os
import glob
import shutil

def sortPredictions():
    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"

    subfolders = {
        "Data": {
            "others": list(range(10, 1018)),
            "training": list(range(1, 10))
        },
        "GluGluHToBB": {
            "others": list(range(46, 200)),
            "training": list(range(1, 46))
        }
    }

    for category, folders in subfolders.items():
        for folder, expected_files in folders.items():
            folder_path = os.path.join(flatPath, category, folder)
            
            # Ensure the folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Get existing files in the folder
            files_in_folder = glob.glob(f"{folder_path}/*.parquet")
            
            # Extract the numeric parts of filenames
            present_numbers = set(
                int(os.path.basename(file).split('_fn')[-1].split('.')[0])
                for file in files_in_folder
            )
            
            # Check for missing files
            expected_set = set(expected_files)
            missing_files = expected_set - present_numbers
            
            if missing_files:
                print(f"Missing files in {folder_path}: {sorted(missing_files)}")
            
            # Check for files in the wrong folder and move them
            other_folder = "training" if folder == "others" else "others"
            other_folder_path = os.path.join(flatPath, category, other_folder)
            files_to_move = [
                f for f in glob.glob(f"{other_folder_path}/*.parquet")
                if int(os.path.basename(f).split('_fn')[-1].split('.')[0]) in expected_files
            ]
            
            for file_path in files_to_move:
                # Destination file path
                destination_file = os.path.join(folder_path, os.path.basename(file_path))
                
                # If the file already exists, remove it
                if os.path.exists(destination_file):
                    os.remove(destination_file)
                    print(f"Overwriting existing file: {destination_file}")
                
                # Move the file
                shutil.move(file_path, folder_path)
                print(f"Moved {os.path.basename(file_path)} to {folder_path}")



def getDfProcesses(reset=False):
    if reset:
        # rerun the script to have the csv file with new table
        from processes import getProcessesDataFrame
        getProcessesDataFrame()
    dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    return dfProcesses


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
    

def loadMultiParquet(paths, nReal=1, nMC=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False):
    '''
    paths = array of string ordered with the first position occupied by the realData
            or list of numbers (integers) with the isMC number
    nReal = how many realdata files load  (-1=all the files, -2 no realData in the array)
    nMC = how many MC files load
    columns = which columns to read

    returnf
    dfs
    numEventsTotal = array of sum num events generated
    '''
    #if columns is not None:
    #    if 'Pileup_nTrueInt' not in columns:
    #        columns = list(columns)
    #        columns.append('Pileup_nTrueInt')
    dfs = []
    numEventsList = []
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    if isinstance(paths[0], str):
        pass
    elif isinstance(paths[0], int):
        for idx, isMCnumber in enumerate(paths):
            df_processes = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
            paths[idx] = df_processes.flatPath[isMCnumber]

    for path in paths: 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
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
            fileNames = sorted(fileNames, key=lambda x: int(''.join(filter(str.isdigit, x))))

        eg = os.path.basename(fileNames[0])
        match = re.match(r'(.+)_([0-9]+)\.parquet', eg)
        process = match.group(1)

        print("Found %d files for process %s"%(len(fileNames), process))
        del match, eg
        if 'Data' in process:
            if nReal>0:
                fileNames = fileNames[:nReal]
            if nReal == -2:
                fileNames = fileNames[:nMC] if nMC!=-1 else fileNames
            elif nReal == -1:
                pass
        else:
            fileNames = fileNames[:nMC] if nMC!=-1 else fileNames

        print("%d files for process %s" %(len(fileNames), process))
        #print("\n")
        
        df = pd.read_parquet(fileNames, columns=columns,
                             engine='pyarrow',
                             filters= [
                                ('jet1_pt', '>',  20),
                                ('jet2_pt', '>',  20),
                                
                                ('jet1_mass', '>', 0),
                                ('jet2_mass', '>', 0),
                                ('jet3_mass', '>',  0),

#
                                ('jet1_eta', '>', -2.5),
                                ('jet2_eta', '>', -2.5),
                                ('jet1_eta', '<',  2.5),
                                ('jet2_eta', '<',  2.5),

                                ('jet1_btagDeepFlavB', '>',  0.2783),
                                ('jet2_btagDeepFlavB', '>',  0.2783),
                                      ]
        )

        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Oct_new.csv")
            for fileName in fileNames:
                #remove the path to file: file_name_322.parquet
                filename = os.path.basename(fileName)
                try:
                    #print(filename)

                    match = re.match(r'(.+)_([0-9]+)\.parquet', filename)
                    #(.+)_([0-9]+)\.parquet: The regular expression captures everything up to the last underscore as the file name ((.+)), followed by a series of digits (([0-9]+)) and ending with .parquet
                    process = match.group(1)
                    fileNumber = int(match.group(2))
                    #print("Process ", process)
                    #print("fileNumber ", fileNumber)
                    #process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    #fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    if returnFileNumberList:
                        fileNumberListProcess.append(fileNumber)
                    if "Data" in process:
                        continue
                    #print(process, fileNumber)
                    numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
                except:
                    process = '_'.join(filename.split('_')[:-1])  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    #print(process, fileNumber)
                    if returnFileNumberList and fileNumberListProcess[-1]!=fileNumber:
                        fileNumberListProcess.append(fileNumber)
                    try:
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
                    except:
                        print(fileNumber, process)
                        # execute compute mini to recompute the df into csv
                        # /t3home/gcelotto/ggHbb/computeMini.py
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]

                
            numEventsList.append(numEventsTotal)
        if returnFileNumberList:
            fileNumberList.append(fileNumberListProcess)
    #PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
    #for df in dfs:
    #    #print(df['Pileup_nTrueInt'])
    #    df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
    #    df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0
    

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

def getZXsections(EWK=False):
    if EWK:
        # first cross section is EWKZJets
        l = [9.8, 5261, 1012.0, 114.2, 25.34, 12.99]
    else:
        l = [5261, 1012.0, 114.2, 25.34, 12.99]
    return l
