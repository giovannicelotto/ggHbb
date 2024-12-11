import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
import argparse
import os

def main(isMC, nFiles, modelName):
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
        if not os.path.exists("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_%s/%s/others"%(modelName, process)):
            os.makedirs("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_%s/%s/others"%(modelName, process))
        pattern = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_%s/%s/**/yMC%d_fn%d.parquet" % (modelName, process, isMC, fileNumber)
        matching_files = glob.glob(pattern, recursive=True)

        if not matching_files:  # No files match the pattern
            print("Launching the job soon")
            print(fileName, str(isMC))
            subprocess.run(['sbatch', '-J', "y%d_%d"%(isMC, random.randint(1, 20)), '/t3home/gcelotto/ggHbb/PNN/slurm/predict.sh', fileName, str(isMC), process, modelName])
            doneFiles = doneFiles + 1
        else:
            print("..")

                     
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script.")

    # Define arguments
    parser.add_argument("-MC", "--MC", type=int, help="number of isMC code", default=1)
    parser.add_argument("-m", "--modelName", type=str, help="suffix of the model", default="dec10")
    parser.add_argument("-n", "--nFiles", type=int, help="number of files", default=1)

    args = parser.parse_args()

    isMC = args.MC
    nFiles = args.nFiles
    modelName = args.modelName
    main(isMC=isMC, nFiles=nFiles, modelName=modelName)