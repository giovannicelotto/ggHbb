import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time



def main(nFiles):
    # Define name of the process, folder for the files and xsections
    
    df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")


    nanoPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Oct09/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/241009_135255/0000"
    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
    if not os.path.exists(flatPath):
        print("Creting flathPath ...", flatPath)
        os.makedirs(flatPath)
    process = "GluGluHToBB"
    nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
    flatFileNames = glob.glob(flatPath+"/**/*.parquet", recursive=True)
    print(process, len(flatFileNames), "/", len(nanoFileNames))
    if len(flatFileNames)==len(nanoFileNames):
        return
        
    time.sleep(1)

    
    nFiles = nFiles if nFiles != -1 else len(nanoFileNames)
    if nFiles > len(nanoFileNames) :
        nFiles = len(nanoFileNames)
    #nFiles to be done
    doneFiles = 0
    for nanoFileName in nanoFileNames:
        if doneFiles==nFiles:
            break
        try:
            #print(nanoFileName)
            fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1))
        except:
            sys.exit("FileNumber not found")

        filePattern = flatPath+"/**/"+process+"_GenMatched_"+str(fileNumber)+".parquet"
        matching_files = glob.glob(filePattern, recursive=True)


        if matching_files:

            continue
        subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 500), '/t3home/gcelotto/ggHbb/genMatching/job.sh', nanoFileName, process, str(fileNumber), flatPath])
        doneFiles = doneFiles+1
    return 

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    main(nFiles=nFiles)