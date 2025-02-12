import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
from functions import getDfProcesses_v2


def main(isMC, nFiles):
    # Define name of the process, folder for the files and xsections
    
    df=getDfProcesses_v2()[0]

    nanoPath = df.nanoPath[isMC]
    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
    process = df.process[isMC]
    if not os.path.exists(flatPath):
        print("Creting flathPath ...", flatPath)
        os.makedirs(flatPath)
    if not os.path.exists(flatPath+"/"+process):
        os.makedirs(flatPath+"/"+process)
    nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
    flatFileNames = glob.glob(flatPath+"/%s/*.parquet"%process, recursive=True)
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
        subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 500), '/t3home/gcelotto/ggHbb/genMatching/genFlatterNu/job.sh', nanoFileName, process, str(fileNumber), flatPath+"/"+process])
        doneFiles = doneFiles+1
    return 

if __name__ == "__main__":
    isMC   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    nFiles = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    main(isMC=isMC, nFiles=nFiles)