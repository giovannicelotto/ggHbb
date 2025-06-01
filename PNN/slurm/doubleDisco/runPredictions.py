import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
import argparse
import os
from functions import getDfProcesses_v2
def main(isMC, processNumber, nFiles, modelName, multigpu, epoch=None):
    # Define name of the process, folder for the files and xsections
    
    df=getDfProcesses_v2()[0] if isMC else getDfProcesses_v2()[1]

    flatPath = list(df.flatPath)[processNumber]
    process = list(df.process)[processNumber]
    print("NN predictions for %s"%process)
    

    files = glob.glob("/t3home/gcelotto/ggHbb/PNN/slurm/outputFiles/*.out")
    for f in files:
        os.remove(f)

    flatFiles = glob.glob(flatPath+"/**/*.parquet", recursive=True)
    if nFiles == -1:
        nFiles = len(flatFiles)


    doneFiles = 0
    dots=0
    for fileName in flatFiles:
        if doneFiles == nFiles:
            print("%d done"%nFiles)
            sys.exit()

        match = re.search(r'_(\d+)\.parquet$', fileName)
        if match:
            fileNumber = int(match.group(1))
        else:
            print("No match found")


        # check if the predictions was already done:
        if not os.path.exists("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_%s/%s/"%(modelName, process)):
            os.makedirs("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_%s/%s/"%(modelName, process))
        pattern = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NN_predictions/DoubleDiscoPred_%s/%s/**/yDD_%s_FN%d.parquet" % (modelName, process, process, fileNumber)
        matching_files = glob.glob(pattern, recursive=True)

        if not matching_files:  # No files match the pattern
            print("Launching the job soon")
            print(fileName, str(processNumber))
            subprocess.run(['sbatch', '-J', "y%s_%d"%(process, random.randint(1, 100)), '/t3home/gcelotto/ggHbb/PNN/slurm/doubleDisco/predict.sh', fileName, str(processNumber), process, modelName, str(multigpu), str(epoch)])
            doneFiles = doneFiles + 1
        else:
            print("Waiting" + "." * dots + " " * (3 - dots), end="\r", flush=True)
            dots = (dots + 1) % 4

                     
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script.")

    # Define arguments
    parser.add_argument("-MC", "--isMC", type=int, help="isMC True or False", default=0)
    parser.add_argument("-pN", "--processNumber", type=int, help="processNumber of MC or datataking", default=0)
    parser.add_argument("-m", "--modelName", type=str, help="suffix of the model", default=None)
    parser.add_argument("-n", "--nFiles", type=int, help="number of files", default=1)
    parser.add_argument("-gpu", "--gpu", type=int, help="Model trained in MultiGPU or SingleGPU", default=1)
    parser.add_argument("-e", "--epoch", type=int, help="Epoch", default=None)

    args = parser.parse_args()

    isMC = args.isMC
    pN = args.processNumber
    nFiles = args.nFiles
    modelName = args.modelName
    multigpu = args.gpu
    epoch = args.epoch
    if (multigpu) & (epoch is None):
        assert False, "Multigpu True and Epoch not specified"
    if modelName is None:
        assert False, "ModelName not specified"
    main(isMC=isMC, processNumber=pN, nFiles=nFiles, modelName=modelName, multigpu=multigpu, epoch=epoch)