import uproot
import sys
import numpy as np
import ROOT
import glob
import time
import os
import pandas as pd
import dask.dataframe as dd
def loadDask(signalPath, realDataPath, nSignalFiles, nRealDataFiles):
    
    signalFileNames = glob.glob(signalPath+"/*.parquet")[:nSignalFiles]
    realDataFileNames = glob.glob(realDataPath+"/*.parquet")[:nRealDataFiles]

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    
    try:    
        signal = dd.read_parquet(signalFileNames)
        realData = dd.read_parquet(realDataFileNames)
        return signal, realData
    except:
        print("Some of the files might be corrupted. Here is the list:\n")
        for fileName in signalFileNames:
            try:
                df=pd.read_parquet(fileName)
            except:
                print(fileName)
        for fileName in realDataFileNames:
            try:
                df=pd.read_parquet(fileName)
            except:
                print(fileName)
        sys.exit("Exiting the program due to a corrupted files.")




def loadData(signalPath, realDataPath, nSignalFiles, nRealDataFiles):
    print("Loading Data...")
    signalFileNames = glob.glob(signalPath+"/*bScoreBased4_*.npy")
    realDataFileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    def load_data_generator(fileNames):
        for fileName in fileNames:
            sys.stdout.write('\r')
            sys.stdout.write("   %d/%d   "%(fileNames.index(fileName)+1, len(fileNames)))
            sys.stdout.flush()
            yield np.array(np.load(fileName, mmap_mode='r')[:, :], dtype=np.float32)

    signal = load_data_generator(signalFileNames)
    bscore4 = load_data_generator(realDataFileNames)
    

    # In Python, a generator is a special type of iterator that allows you to iterate over a potentially large 
    # sequence of data without loading the entire sequence into memory at once. It generates values on-the-fly 
    # as you iterate through it, making it memory-efficient.


    return signal, bscore4

def checkZombie():
    path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
    fileNames = glob.glob(path+"/BParking*.root")
    for fileName in fileNames:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(fileNames.index(fileName)+1, len(fileNames)))
        sys.stdout.flush()
        try:
            root_file = ROOT.TFile.Open(fileName)
        except:
            print(fileName)
            os.remove(fileName)
        if root_file.IsZombie():
            print(fileName)
            os.remove(fileName)
def checkIterators():
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatData"
    def load_data_generator(fileNames):
        for fileName in fileNames:
            yield np.array(np.load(fileName, mmap_mode='r')[:, :], dtype=np.float32)

    realDataFileNames = glob.glob(realDataPath+"/*.npy")[:35]
    realData = load_data_generator(realDataFileNames)
    
    start_time = time.time()
    c=None
    for sig in realData:
        ## do some operations
        #c=c+len(sig)
        mask = (sig[:,0]>20) & (abs(sig[:,1])<2.5)
        if c is None:
            c=np.histogram(sig[mask,21], bins=np.linspace(0, 500, 10))[0]
        else:
            c=c+np.histogram(sig[mask,21], bins=np.linspace(0, 500, 10))[0]
        print(c/35)
        # get the result for the current file only
    
    elapsed_time_rdf = time.time() - start_time
    print(f"Time taken for numpy: {elapsed_time_rdf} seconds")
    

def uprootGenerator():
    def root_file_generator(file_path):
        for fileName in file_path:
            with uproot.open(fileName) as file:
                tree = file["Events"]
                arrays = tree.arrays(library="np")
                #values = arrays.values()
                #nparray=np.array(list(values), dtype=np.float32).T

                yield arrays

    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
    fileNames = glob.glob(signalPath+"/BParking*.root")[:35]
    print("%d files"%len(fileNames))
    
    start_time = time.time()
    c=None
    for file in root_file_generator(fileNames[:]):
        mask = (file['jet1_pt']>20) & (abs(file['jet1_eta'])<2.5)
        print(type(file))
        if c is None:
            c=np.histogram(file['dijet_M'][mask], bins=np.linspace(0, 500, 10))[0]
        else:
            c=c+np.histogram(file['dijet_M'][mask], bins=np.linspace(0, 500, 10))[0]
        print(c/35)

        

    
    elapsed_time_rdf = time.time() - start_time
    print(f"Time taken for uprootGenerator: {elapsed_time_rdf} seconds")
def readParquet():
    def yieldDF():
        path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
        fileNames = glob.glob(path+"/*.parquet")
        df = pd.DataFrame()
        for fileName in fileNames:
            sys.stdout.write('\r')
            sys.stdout.write("   %d/%d   "%(fileNames.index(fileName)+1, len(fileNames)))
            sys.stdout.flush()
            try:
                df_ = pd.read_parquet(fileName)
                yield df_
                #if len(df)==0:
                #    df=df_
                #else:
                #    df = pd.concat((df, df_))
            except:
                print(fileName)
        #return df.iloc[0,21].mean()
    dfs = yieldDF()
    for df in dfs:
        pass
        #print(df.iloc[0,21].mean())

def rdf():
    ROOT.EnableImplicitMT()
    chain = ROOT.TChain("Events")
    path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
    fileNames = glob.glob(path+"/BParking*.root")
    print("%d files"%len(fileNames))
    start_time = time.time()
    for fileName in fileNames[:]:
        chain.Add(fileName)
    rdf = ROOT.RDataFrame(chain)
    leaves_order = [column for column in rdf.GetColumnNames()]
    npy = rdf.AsNumpy(columns=leaves_order)
    array = np.column_stack(list(npy.values()))
    elapsed_time_rdf = time.time() - start_time
    print(f"Time taken for rdf: {elapsed_time_rdf} seconds")
    print("\n", array.shape,"\n")

    return array

def main():
    path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatDataRoot"
    fileNames = glob.glob(path+"/BParking*.root")
    print(len(fileNames), " files to be used")
    arList=[]
    for fileName in fileNames[:]:
        print(fileName)
        
        f=uproot.open(fileName)
        tree = f['Events']
        
        
        branches = tree.arrays(library="numpy")
        branches=np.array(list(branches.values())).T
        arList.append(branches)

    ar = np.vstack(arList)
    
    print(ar.shape)

    return


if __name__=="__main__":
    #rdf()
    #uprootGenerator()
    #checkIterators()
    #readParquet()
    loadDask()
    
    