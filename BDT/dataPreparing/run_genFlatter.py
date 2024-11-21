import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time



def main(nFiles, mass):
    # Define name of the process, folder for the files and xsections
    df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    if mass == 125:
        isMC = 1
    elif mass ==50:
        isMC = 40
    elif mass ==70:
        isMC = 41
    elif mass ==100:
        isMC = 42
    elif mass ==200:
        isMC = 43
    elif mass ==300:
        isMC = 44
    

    
    
    path = df.nanoPath[isMC]                                    # nanoPath from where flattening
    fileNames = glob.glob(path+'/**/*.root', recursive=True)    # fileNames from the nanoPath
    prefix=str(mass)                                            # string for organizing

    if (nFiles > len(fileNames)) | (nFiles == -1):
        nFiles=len(fileNames)
        

    print("nFiles                : ", nFiles)
    print("fileNames             : ", len(fileNames))
    doneFiles = 0
    for fileName in fileNames:
        if doneFiles==nFiles:
            print("Reached the end :) ")
            break
        finalDestinationFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/%s"%prefix

        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(finalDestinationFolder +"/%s_%s.parquet"%(prefix, fileNumber)):
            print("skip")
            # if you already saved this file skip
            #print("%s_%s.parquet already present\n"%(prefix, fileNumber))
            continue
        subprocess.run(['sbatch', '-J', "M"+str(mass)+"_%d"%random.randint(1, 40), '/t3home/gcelotto/ggHbb/BDT/dataPreparing/job.sh', fileName, prefix, finalDestinationFolder, fileNumber])
        doneFiles = doneFiles+1
    return 

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    mass                   = int(sys.argv[2])    # mass of the particle GluGluSpin0_M<> # possible values 50, 70, 100, 125, 200, 300
    main(nFiles=nFiles, mass=mass)