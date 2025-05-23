import pandas as pd
import glob, sys, re, os
import random
import subprocess
import argparse
import os
from functions import getDfProcesses_v2
def main(isMC, processNumber, nFiles, modelName):
    # Define name of the process, folder for the files and xsections
    
    df=getDfProcesses_v2()[0] if isMC else getDfProcesses_v2()[1]

    flatPath = list(df.flatPath)[processNumber]
    # temp
    #if isMC==0:
    #    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/flat_old/Data1A"
    #elif isMC==1:
    #    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/flat_old/GluGluHToBB"
    process = list(df.process)[processNumber]
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
        pattern = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_%s/%s/**/y%s_FN%d.parquet" % (modelName, process, process, fileNumber)
        matching_files = glob.glob(pattern, recursive=True)

        if not matching_files:  # No files match the pattern
            print("Launching the job soon")
            print(fileName, str(processNumber))
            subprocess.run(['sbatch', '-J', "y%s_%d"%(process, random.randint(1, 300)), '/t3home/gcelotto/ggHbb/PNN/slurm/predict.sh', fileName, str(processNumber), process, modelName])
            doneFiles = doneFiles + 1
        else:
            print("..")

                     
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script.")

    # Define arguments
    parser.add_argument("-MC", "--isMC", type=int, help="isMC True or False", default=0)
    parser.add_argument("-pN", "--processNumber", type=int, help="processNumber of MC or datataking", default=0)
    parser.add_argument("-m", "--modelName", type=str, help="suffix of the model", default="Jan08_250p0")
    parser.add_argument("-n", "--nFiles", type=int, help="number of files", default=1)

    args = parser.parse_args()

    isMC = args.isMC
    pN = args.processNumber
    nFiles = args.nFiles
    modelName = args.modelName
    main(isMC=isMC, processNumber=pN, nFiles=nFiles, modelName=modelName)