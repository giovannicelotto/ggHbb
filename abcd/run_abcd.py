import numpy as np
import pandas as pd
import glob, sys
import random
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
import time
hep.style.use("CMS")



def closure(nFilesData, nFilesMC):
    # Define name of the process, folder for the files and xsections

    df.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    
    currentLumi = nFilesData * 0.774 / 1017
    np.save('/t3home/gcelotto/ggHbb/bkgEstimation/output/currentLumi.npy', currentLumi)
    # choose binning for HT
    bins=np.linspace(0, 800, 20)
    toSkip = df.index
    np.save('/t3home/gcelotto/ggHbb/bkgEstimation/output/binsForHT.npy', bins)
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        print("Starting process ", process)
        if process in list(toSkip):
            print("skipping process ....", process)
            continue
        print(process, list(toSkip))
        time.sleep(50)
        nFiles=nFilesData if process=='Data' else nFilesMC
        
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        if nFiles > len(fileNames) :
            nFiles = len(fileNames)
        
        if process!='Data':
            np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/mini_%s.npy"%process, np.float32(0))
        counts = np.zeros(len(bins)-1) # counts of MC sample
        np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/counts_%s.npy"%process, counts)

        print("Process : ", process, "     %d files"%nFiles)
        for fileName in fileNames[:nFiles]:
            subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 4), '/t3home/gcelotto/ggHbb/bkgEstimation/scripts/process_job.sh', fileName, process])
       
    return 

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData = int(nFilesData)
    nFilesMC = int(nFilesMC)

    closure(nFilesData, nFilesMC)