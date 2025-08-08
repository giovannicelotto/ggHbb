import os, re
import pandas as pd
import glob, json
import numpy as np
import os
import glob
import shutil
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
import sys
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from utilsForPlot import getBins
def extract_numbers_from_filenames(filenames):
    """
    Extracts integers from a list of strings that end with '_number.parquet'.
    
    Args:
        filenames (list): A list of strings, each ending with '_number.parquet'.
        
    Returns:
        list: A list of integers extracted from the filenames.
    """
    numbers = []
    for filename in filenames:
        # Split on '_' and take the last part before '.parquet'
        number_str = filename.split('_')[-1].replace('.parquet', '')
        numbers.append(int(number_str))  # Convert to integer and append
    return numbers


def create_intersection_mask(list1, list2):
    """
    Creates a mask for the intersection of two lists, keeping only the common elements.
    
    Args:
        list1 (list): First list of numbers.
        list2 (list): Second list of numbers.
    
    Returns:
        list: A boolean mask of the same length as list1 indicating common elements.
    """
    # Convert list2 to a set for efficient lookup
    set2 = set(list2)
    # Create the mask by checking if each element in list1 is in set2
    mask = [item in set2 for item in list1]
    return mask

def getDfProcesses_v2 (reset=False):
    if reset:
        # rerun the script to have the csv file with new table
        from processes import getProcessesDataFrame
        getProcessesDataFrame()
    dfProcessesMC = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv")
    dfProcessesData = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
    dfProcessesMC_JEC = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC_JEC.csv")
    return dfProcessesMC, dfProcessesData, dfProcessesMC_JEC

def getCommonFilters(btagTight=False, btagWP=None):
    '''
    btagTight True for Tight, False for Medium
    btagWP overwrites the btagTight argument (L, M, T)
    '''
    btag = 0.2783 if btagTight is False else 0.71
    if btagWP=="L":
        btag = 0.049
    elif btagWP=="M":
        btag = 0.2783
    elif btagWP=="T":
        btag = 0.71
    elif btagWP==None:
        "No WP provided, you are using the btagTight argument"
    else:
        assert False

    if btagTight:
        print("Setting btag cut to 0.71 for both jets")
    filters = [
            [   ('jet1_pt', '>',  20),
                ('jet2_pt', '>',  20),
                
                ('jet1_mass', '>', 0),
                ('jet2_mass', '>', 0),
                #('jet3_mass', '>',  0),
                
                ('jet1_eta', '>', -2.5),
                ('jet2_eta', '>', -2.5),
                ('jet1_eta', '<',  2.5),
                ('jet2_eta', '<',  2.5),
                ('jet1_btagDeepFlavB', '>',  btag),
                ('jet2_btagDeepFlavB', '>',  btag),
                ('muon_pt', '>=',  9.0),
                ('muon_eta', '>=',  -1.5),
                ('muon_eta', '<=',  1.5),
                ('muon_dxySig', '>=', 6.0)
                ],
                  
                  # OR Condition

            [   ('jet1_pt', '>',  20),
                ('jet2_pt', '>',  20),
                
                ('jet1_mass', '>', 0),
                ('jet2_mass', '>', 0),
                #('jet3_mass', '>',  0),
                
                ('jet1_eta', '>', -2.5),
                ('jet2_eta', '>', -2.5),
                ('jet1_eta', '<',  2.5),
                ('jet2_eta', '<',  2.5),
                ('jet1_btagDeepFlavB', '>',  btag),
                ('jet2_btagDeepFlavB', '>',  btag),
                ('muon_pt', '>=',  9.0),
                ('muon_eta', '>=',  -1.5),
                ('muon_eta', '<=',  1.5),
                ('muon_dxySig', '<=', -6.0)
                ]

    ]
    return filters
def loadMultiParquet_v2(paths, nMCs=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters(), training=False, isJEC=0):
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
    genEventSumwList = []
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    # Check types are as expected
    if isinstance(paths[0], str):
        print("Path is string")
        pass
    elif isinstance(paths[0], int):
        print("Path is integer")
        paths = paths.copy()
        #for idx, isMCnumber in enumerate(paths):
        if isJEC==0:
            df_processesMC = getDfProcesses_v2()[0]
        elif isJEC==1:
            df_processesMC = getDfProcesses_v2()[2]
        df_processesMC = df_processesMC.iloc[paths]
        paths = list(df_processesMC.flatPath)
    else:
        
        assert False, "path is of unreco type %s"%type(paths[0])


    if isinstance(nMCs, int):
        nMCs = [nMCs for i in range(len(paths))]
    elif isinstance(nMCs, list):
        if len(nMCs)==len(paths):
            pass
        else:
            print("Error, number of nMCs requested is different from the number of processes. Setting nMCs equal to the first instance")
            nMCs = [nMCs[0] for i in range(len(paths))]
    
    
    print(nMCs)
    for nMC, path,processName in zip(nMCs, paths, df_processesMC.process): 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
        # loop over processes
        if training:
            if path == "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others":
                path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training"
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)
        fileNames = sorted(
                                fileNames,
                                key=lambda x: int(''.join(filter(str.isdigit, x.split("%s_"%processName)[-1])))
        )

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

        print("Found %d flattuple for process %s"%(len(fileNames), process))
        del match, eg

        
        df = pd.read_parquet(fileNames, columns=columns,
                                 engine='pyarrow',
                                 filters= filters        )

        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            genEventSumw=0
            if isJEC:
                processName = '_'.join(processName.split('_')[:-2])

                # Handle special case of GluGLu where two dataframes are used one for train and one for test
                #if processName == "GluGluHToBB":
                    # Read both CSV files
                #    df1 = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_GluGluHToBB_tr.csv")
                #    df2 = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_GluGluHToBB.csv")
                
                    # Concatenate the two DataFrames row-wise
                #    df = pd.concat([df1, df2], ignore_index=True)
                
                    # Set the 'process' column to a fixed value
                #    df['process'] = 'GluGluHToBB'
                #else:
                df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_%s.csv"%processName)
            else:
                df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_%s.csv"%processName)
            for fileName in fileNames:
                #remove the path to file: file_name_322.parquet
                filename = os.path.basename(fileName)
                try:
                    match = re.match(r'(.+)_([0-9]+)\.parquet', filename)
                    #(.+)_([0-9]+)\.parquet: The regular expression captures everything up to the last underscore as the file name ((.+)), followed by a series of digits (([0-9]+)) and ending with .parquet
                    #process = match.group(1)
                    fileNumber = int(match.group(2))
                    #print("Process ", process)
                    #print("fileNumber ", fileNumber)
                    #process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    #fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    if returnFileNumberList:
                        fileNumberListProcess.append(fileNumber)
                    try:
                        genEventSumw = genEventSumw + df[(df.process==processName) & (df.fileNumber==fileNumber)].genEventSumw.iloc[0]
                    except:
                        print(df[df.process==processName])
                        print(df[df.fileNumber==fileNumber])
                        print("Error finding the fileNumber", processName, fileNumber)
                        assert False, "Error finding the fileNumber"
                except:
                    process = '_'.join(filename.split('_')[:-1])  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
                    fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
                    #print(process, fileNumber)
                    if returnFileNumberList and fileNumberListProcess[-1]!=fileNumber:
                        fileNumberListProcess.append(fileNumber)
                    try:
                        genEventSumw = genEventSumw + df[(df.process==processName) & (df.fileNumber==fileNumber)].genEventSumw.iloc[0]
                    except:
                        print(fileNumber, process)
                        # execute compute mini to recompute the df into csv
                        # /t3home/gcelotto/ggHbb/computeMini.py
                        genEventSumw = genEventSumw + df[(df.process==processName) & (df.fileNumber==fileNumber)].genEventSumw.iloc[0]

                
            genEventSumwList.append(genEventSumw)
        if returnFileNumberList:
            fileNumberList.append(fileNumberListProcess)


    if (returnNumEventsTotal) & (not returnFileNumberList):
        return dfs, genEventSumwList
    elif returnFileNumberList:
        print("lenght of elements returned in fileNumberList")
        print([len(el) for el in fileNumberList])
        return dfs, genEventSumwList, fileNumberList
    else:
        return dfs
    


def loadMultiParquet_v2_dep(paths, nMCs=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters()):
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

    for nMC, path,processName in zip(nMCs, paths, df_processesMC.process): 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
        # loop over processes
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)
        fileNames = sorted(
                                fileNames,
                                key=lambda x: int(''.join(filter(str.isdigit, x.split("%s_"%processName)[-1])))
        )

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

        print("Found %d flattuple for process %s"%(len(fileNames), process))
        del match, eg

        
        df = pd.read_parquet(fileNames, columns=columns,
                                 engine='pyarrow',
                                 filters= filters        )

        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Jan.csv")
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
                    try:
                        numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
                    except:
                        print(process, fileNumber)
                        assert False
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
    
##### new function
def loadMultiParquet_Data_new(dataTaking=[0], nReals=[1], columns=None, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters(), training=False):
    import sys
    '''
    dataTaking :list  = list of data taking idx
    nReals :list = list --> one number per data taking. ignored if selectFilenUmberslist is not none
    columns :list = which columns to read
    selectFileNumberList :list = list of list of fileNumbers
    returnFileNumberList :bool = True for returning the list of flattuple read

    return
    dfs = list of dfs of MC
    fileNumberList = list of filenumbers for each process e.g.[[1, 2, 3], [1, 2, 3, 4], ...]
    '''
    
    print("LoadMultiParquet Data v.Jan25")
    # Check on the arguments
    if isinstance(nReals, int):
        print("nReal is integer setting the same")
        nReals = [nReals for i in range(len(dataTaking))]
    elif isinstance(nReals, list):
        if len(nReals)==len(dataTaking):
            pass
        else:
            print("Error, number of nReals requested is different from the number of processes. Setting nReals to -1 always")
            nReals = [-1 for i in range(len(dataTaking))]
            print(nReals)
    assert len(dataTaking)==len(nReals)
    if selectFileNumberList is not None:
        assert len(dataTaking)==len(selectFileNumberList), "mismatch between len data taking and len selectFileNumberList"
        assert isinstance(selectFileNumberList[0], list), "selectFileNumberList needs to be a list of list if not set to None"
    

    dfs = []
    df_processesData = getDfProcesses_v2()[1].iloc[dataTaking]
    flatPaths = list(df_processesData.flatPath)
    lumi_tot = 0
    
    # In case want the fileNumbers returned, create a list where to store the numbers
    if returnFileNumberList:
        fileNumberList = []



    for nReal, path, lumi, nFiles, processName, dataTakingIdx in zip(nReals, flatPaths, df_processesData.lumi, df_processesData.nFiles, df_processesData.process, dataTaking): 

        print("\n\n", processName, " | ", lumi, " | ", nReal, " requested", " | ", nFiles, " avail. at nano", )
        if dataTakingIdx == 17:
            assert processName=='Data1D'
            if training:
                print("Files used for training are called")
                flatPaths[flatPaths.index(path)] = path+"/training"    
                path = path+"/training"
            else:
                flatPaths[flatPaths.index(path)] = path+"/others"
                path = path+"/others"
        # loop over processes
        print("PATH : ", path)

        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)
        print("Files sorted per FileNumber")
        fileNames = sorted(
                                fileNames,
                                key=lambda x: int(''.join(filter(str.isdigit, x.split("%s_"%processName)[-1])))
        )

        print(len(fileNames), " files avail at flat")
        nReal = len(fileNames) if nReal == -1 else nReal
        #if selectFileNumberList is not None then keep only strings where there is a match (want keep only files when i have predictions)
        if selectFileNumberList is not None:
            print(len(selectFileNumberList[flatPaths.index(path)]), " predictions")
            flattuple_numbers = extract_numbers_from_filenames(fileNames)
            nIntersections = sum(create_intersection_mask(flattuple_numbers, selectFileNumberList[flatPaths.index(path)]))
            print(nIntersections, " files for intersection between predictions and flat")
            if nReal < nIntersections:
                print("nReals < nIntersections. Keeping less in the predictions")
                flattuple_numbers = flattuple_numbers[:nReal]
            else:
                nReal =  nIntersections if nReal > nIntersections else nReal
            intersectionMaskFlattuple = create_intersection_mask(flattuple_numbers, selectFileNumberList[flatPaths.index(path)])
            intersectionMaskPredictions = create_intersection_mask(selectFileNumberList[flatPaths.index(path)], flattuple_numbers)
            print("Keeping FileNumber of predictions which are in the intersections")
            selectFileNumberList[flatPaths.index(path)] = list(np.array(selectFileNumberList[flatPaths.index(path)])[intersectionMaskPredictions])
            print(len(selectFileNumberList[flatPaths.index(path)]), " predictions")

            print("Looking for a specific list of ", len(selectFileNumberList[flatPaths.index(path)]), " in the flattuple files for which predictions is available")
            fileNamesSelectedProcess = []
            for fileName in fileNames:
                match = re.search(r'_(\d+).parquet', fileName)
                if match:
                    #print("match")
                    fn = match.group(1)
                if int(fn) in selectFileNumberList[flatPaths.index(path)]:
                    #print("append")
                    fileNamesSelectedProcess.append(fileName)
                else:
                    pass
                    #print("remove", int(fn), selectFileNumberList[paths.index(path)])
            fileNames = fileNamesSelectedProcess
        else:
            fileNames = fileNames[:nReal]
        currentLumi = len(fileNames)*lumi/nFiles
        print(currentLumi, " fb -1")
        lumi_tot = lumi_tot + currentLumi

        print("%d files for process %s" %(len(fileNames), processName))

        df = pd.read_parquet(fileNames, columns=columns,
                                    engine='pyarrow',
                                    filters= filters        )

        dfs.append(df)
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
            #print(selectFileNumberList[:10])
            #print(fileNumberList[:10])




    if returnFileNumberList:
        print("lenght of elements returned in fileNumberList")
        print([len(el) for el in fileNumberList])
        return dfs, lumi_tot, fileNumberList
    else:
        return dfs, lumi_tot
    





    ######

def loadMultiParquet_Data(paths, nReals=1, columns=None, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters()):
    assert False
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
    print(df_processesData.lumi)
    lumi_tot = 0
    if returnFileNumberList:
        fileNumberList = []
    fileNamesSelected=[]
    # Check types are as expected
    if isinstance(paths[0], str):
        pass
    elif isinstance(paths[0], int):
        print("Conversion of paths from int (processNumber) to string (folderPath)")
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

    for nReal, path, lumi, nFiles, processName in zip(nReals, paths, df_processesData.lumi, df_processesData.nFiles, df_processesData.process): 
        assert isinstance(path, str), "Paths do not contains strings: %s"%str(path)
        print(processName, " | ", lumi)
        # loop over processes
        print("PATH : ", path)
        fileNames = glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True)
        fileNames = sorted(
                                fileNames,
                                key=lambda x: int(''.join(filter(str.isdigit, x.split("%s_"%processName)[-1])))
        )
        print(len(fileNames), " files in flattuple")
        nReal = len(fileNames) if nReal == -1 else nReal
        #if selectFileNumberList is not None then keep only strings where there is a match (want keep only files when i have predictions)
        if selectFileNumberList is not None:
            flattuple_numbers = extract_numbers_from_filenames(fileNames)
            nIntersections = sum(create_intersection_mask(flattuple_numbers, selectFileNumberList[paths.index(path)]))
            print(nIntersections, " files for intersection")
            print("Setting nReal to nIntersections")
            nReal =  nIntersections if nReal > nIntersections else nReal

            intersectionMaskFlattuple = create_intersection_mask(flattuple_numbers, selectFileNumberList[paths.index(path)])
            intersectionMaskPredictions = create_intersection_mask(selectFileNumberList[paths.index(path)], flattuple_numbers)
            print(len(selectFileNumberList[0]), " predictions")
            print("Cropping the list of predictions to the last %d elements"%nReal)
            selectFileNumberList[paths.index(path)] = list(np.array(selectFileNumberList[paths.index(path)])[intersectionMaskPredictions])
            print(len(selectFileNumberList[0]), " predictions")

            print("Looking for a specific list of ", len(selectFileNumberList[paths.index(path)]), " files for which predictions is available")
            fileNamesSelectedProcess = []
            for fileName in fileNames:
                match = re.search(r'_(\d+).parquet', fileName)
                if match:
                    #print("match")
                    fn = match.group(1)
                if int(fn) in selectFileNumberList[paths.index(path)]:
                    #print("append")
                    fileNamesSelectedProcess.append(fileName)
                else:
                    pass
                    #print("remove", int(fn), selectFileNumberList[paths.index(path)])
            fileNamesSelected.append(fileNamesSelectedProcess)
            fileNames=fileNamesSelectedProcess
        fileNames = fileNames[:nReal] if nReal!=-1 else fileNames
        lumi_tot = lumi_tot + nReal*lumi/nFiles
        try:
            eg = os.path.basename(fileNames[0])
        except:
            print(fileNames)


            assert False
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

def sortPredictions(predictionsPath="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"):

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
            folder_path = os.path.join(predictionsPath, category, folder)
            
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
            other_folder_path = os.path.join(predictionsPath, category, other_folder)
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
                    df = df[df[feature] >= min]
                if max is not None:
                    df = df[df[feature] < max]
                newData.append(df)
            return newData

def cut_advanced(data, feature, condition):
    """
    Filters a list of DataFrames based on a specified condition for a feature.

    Parameters:
        data (list of pd.DataFrame): List of DataFrames to filter.
        feature (str): The column/feature name to apply the condition on.
        condition (str): A string representing the condition, e.g., 
                         '-2.5 < dijet_eta <= 2.5', '|dijet_eta| > 2.5'.

    Returns:
        list of pd.DataFrame: List of filtered DataFrames.
    """
    newData = []
    for df in data:
        # Replace the feature placeholder in the condition with the actual column
        condition_with_feature = condition.replace(feature, f"df['{feature}']")
        
        # Evaluate the condition safely
        filtered_df = df[eval(condition_with_feature)]
        newData.append(filtered_df)
    
    return newData
    

def loadMultiParquet(paths, nReal=1, nMC=1, columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters()):
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
        if len(fileNames)==0:
            print("Length was zero in ", path)
            fileNames = glob.glob(path+"/*.parquet")


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
            columnsToRead = [f for f in columns if ('gen' not in f) & ('btag_central' not in f) ]
            if nReal>0:
                fileNames = fileNames[:nReal]
            if nReal == -2:
                fileNames = fileNames[:nMC] if nMC!=-1 else fileNames
            elif nReal == -1:
                pass
        else:
            fileNames = fileNames[:nMC] if nMC!=-1 else fileNames
            columnsToRead = columns

        print("%d files for process %s" %(len(fileNames), process))
        #print("\n")
        
        df = pd.read_parquet(fileNames, columns=columnsToRead,
                             engine='pyarrow',
                             filters= filters 
                                      
        )

        dfs.append(df)
        if returnFileNumberList:
            fileNumberListProcess = []
    #return dfs
        if returnNumEventsTotal:
            numEventsTotal=0
            df = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Jan.csv")
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


def plotNormalizedFeatures(data, outFile, legendLabels, colors, histtypes=None, alphas=None, figsize=None, autobins=False, weights=None, error=True):
    '''
    plot normalized features of signal and background
    data= list of dataframes one for each process'''
    # Find common columns
    common_columns = set(data[0].columns) 
    for df in data[1:]:
        common_columns.intersection_update(df.columns)  # Update with the intersection of columns
    ordered_common_columns = [col for col in data[0].columns if col in common_columns]

    # Retain only the common columns in their original order for each DataFrame
    data = [df[ordered_common_columns] for df in data]


    xlims = getBins(dictFormat=True)
    nRow, nCol = int(len(data[0].columns)/6+int(bool(len(data[0].columns)%6))), 6
    fig, ax = plt.subplots(nRow, nCol, figsize=(20, 15) if figsize==None else figsize, constrained_layout=True)
    fig.align_ylabels(ax[:,0])
    
    for i in range(nRow):
        fig.align_xlabels(ax[i,:])
        for j in range(nCol):
            if i*nCol+j>=len(data[0].columns):
                break
            featureName = data[0].columns[i*nCol+j]

            print("="*30)
            print(featureName)
            fig.align_ylabels(ax[:,j])
            if featureName not in xlims.columns:
                bins = np.linspace(data[0][featureName].min(), data[0][featureName].max(), 20)
                print("Feature %s not found. Binning Automatically defined"%featureName)
            else:
                bins = np.linspace(xlims[featureName][1], xlims[featureName][2], int(xlims[featureName][0])+1)
            if autobins:
                try:
                    xmin, xmax = data[0][featureName].quantile(0.02), data[0][featureName].quantile(0.98)
                    for idx in range(len(data)):
                        if data[idx][featureName].quantile(0.1) < xmin:
                            xmin =data[idx][featureName].quantile(0.1)
                        if data[idx][featureName].quantile(0.9) > xmax:
                            xmax = data[idx][featureName].quantile(0.9)
                    bins = np.linspace(xmin, xmax, 20)
                    if featureName=='sf':
                        bins=np.linspace(0, 1, 20)
                except:
                    bins = np.linspace(data[1][featureName].min(), data[1][featureName].max(), 20)
            dataIdx = 0
            for idx, df in enumerate(data):
                
                if weights is None:
                    weightsDf=df.sf*df.PU_SF
                else:
                    weightsDf = weights[idx]
                counts = np.zeros(len(bins)-1)
                
                feature_data = df[featureName].astype(int) if df[featureName].dtype == bool else df[featureName]
                counts = np.histogram(np.clip(feature_data, bins[0], bins[-1]),weights = weightsDf if featureName!='sf' else None, bins=bins)[0]
                              
                
                if ((counts<0).any()):
                    print("Negative counts in ", featureName)
                    #print(counts)
                if error:
                    countsErr = np.sqrt(np.histogram(np.clip(feature_data, bins[0], bins[-1]),weights = weightsDf**2 if featureName!='sf' else None, bins=bins)[0])

                
                # Normalize the counts to 1 so also the errors undergo the same operation. Do first the errors, otherwise you lose the info on the signal
                    countsErr = countsErr/np.sum(counts)
                counts = counts/np.sum(counts)
                

                ax[i, j].hist(bins[:-1], bins=bins, weights=counts, label=legendLabels[dataIdx], histtype=u'step' if histtypes==None else histtypes[dataIdx], 
                            alpha=1 if alphas==None else alphas[dataIdx], color=colors[dataIdx], )[:2]
                
                ax[i, j].set_xlabel(featureName, fontsize=18)
                ax[i, j].set_xlim(bins[0], bins[-1])
                ax[i, j].set_ylabel("Probability", fontsize=18)

                # Some subplots in log scale
                if any(substring in df.columns[i * nCol + j] for substring in ['nMuons', 'nElectrons','nTightMuons' ]):
                    ax[i, j].set_yscale('log')
                if featureName == 'sf':
                    ax[i, j].legend(fontsize=18)


                ax[i, j].tick_params(which='major', length=8)
                ax[i, j].xaxis.set_minor_locator(AutoMinorLocator())
                ax[i, j].tick_params(which='minor', length=4)

                if error:
                    for idx in range(len(bins)-1):
                        rect = patches.Rectangle((bins[idx], counts[idx] - countsErr[idx]),
                            bins[idx+1]-bins[idx], 2 *  countsErr[idx],
                            linewidth=0, edgecolor=colors[dataIdx], facecolor='none', hatch='///')
                        ax[i, j].add_patch(rect)
                dataIdx = dataIdx + 1

    
    fig.savefig(outFile, bbox_inches='tight')
    print("Saving in %s"%outFile)
    plt.close('all')