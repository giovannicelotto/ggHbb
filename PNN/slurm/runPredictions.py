import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
import os

def main(isMC, nFiles):
    # Define name of the process, folder for the files and xsections
    
    df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")

    flatPath = list(df.flatPath)[isMC]
    # temp
    #if isMC==0:
    #    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/flat_old/Data1A"
    #elif isMC==1:
    #    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/flat_old/GluGluHToBB"
    process = list(df.process)[isMC]
    print("NN predictions for %s"%process)
    

    files = glob.glob("/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/*.out")
    for f in files:
        os.remove(f)

    flatFiles = glob.glob(flatPath+"/**/*.parquet", recursive=True)
    if nFiles == -1:
        nFiles = len(flatFiles)


    doneFiles = 0
    for fileName in flatFiles:
        if doneFiles == nFiles:
            print("%d done"%nFiles)
            sys.exit()

        match = re.search(r'_(\d+)\.parquet$', fileName)
        if match:
            fileNumber = int(match.group(1))
        else:
            print("No match found")
        #print(fileName)


        # check if the predictions was already done:
        if not os.path.exists("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18/%s/others"%process):
            os.makedirs("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18/%s/others"%process)
        pattern = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18/%s/**/yMC%d_fn%d.parquet" % (process, isMC, fileNumber)
        matching_files = glob.glob(pattern, recursive=True)

        if not matching_files:  # No files match the pattern
            print("Launching the job soon")
            print(fileName, str(isMC))
            subprocess.run(['sbatch', '-J', "y%d_%d"%(isMC, random.randint(1, 20)), '/t3home/gcelotto/ggHbb/PNN/slurm/predict.sh', fileName, str(isMC), process])
            doneFiles = doneFiles + 1
        else:
            print("..")

                     
        

if __name__ == "__main__":
    isMC = int(sys.argv[1])
    nFiles = int(sys.argv[2])
    main(isMC=isMC, nFiles=nFiles)