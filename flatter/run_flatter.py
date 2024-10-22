import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time



def main(isMC, nFiles, maxEntries, maxJet):
    # Define name of the process, folder for the files and xsections
    
    df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    nanoPath = list(df.nanoPath)[isMC]
    flatPath = list(df.flatPath)[isMC]
    if not os.path.exists(flatPath):
        print("Creting flathPath ...", flatPath)
        os.makedirs(flatPath)
    process = list(df.process)[isMC]
    nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
    print("Look for ", nanoPath+"/**/*.root")
    flatFileNames = glob.glob(flatPath+"/**/*.parquet", recursive=True)
    print(process, len(flatFileNames), "/", len(nanoFileNames))
    if len(flatFileNames)==len(nanoFileNames):
        return
        
    time.sleep(1)
    #sys.exit("exit")
    
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

        filePattern = flatPath+"/**/"+process+"_"+str(fileNumber)+".parquet"
        matching_files = glob.glob(filePattern, recursive=True)
        #print("Checking for ", flatPath+"/**/"+process+"_"+fileNumber+".parquet")

        if matching_files:
            #print(process+"_"+fileNumber+".parquet present. Skipped")
            continue
            #pass
        #print(process, fileNumber, str(isMC))
        #print(flatPath)
        subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 500), '/t3home/gcelotto/ggHbb/flatter/job.sh', nanoFileName, str(maxEntries), str(maxJet), str(isMC), process, str(fileNumber), flatPath])
        doneFiles = doneFiles+1
    return 

if __name__ == "__main__":
    isMC        = int(sys.argv[1]) if len(sys.argv) > 1 else 1 # 0 = Data , 1 = ggHbb, 2 = ZJets
    nFiles      = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    # optional
    maxEntries  = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    maxJet      = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print("max jet", maxJet)
    main(isMC, nFiles, maxEntries, maxJet)