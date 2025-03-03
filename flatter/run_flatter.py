import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-MC", "--isMC", type=int, help="isMC True or False", default=None)
parser.add_argument("-pN", "--processNumber", type=int, help="process Number for MC or datataking for Data", default=None)
parser.add_argument("-n", "--nFiles", type=int, help="number of files to flatten per process", default=-1)
parser.add_argument("-mE", "--maxEntries", type=int, help="max number of Entries", default=-1)
parser.add_argument("-mJ", "--maxJet", type=int, help="max number of jet", default=4)
args = parser.parse_args()

isMC            = True if args.isMC==1 else False
pN              = args.processNumber
nFiles          = args.nFiles
maxEntries      = args.maxEntries
maxJet          = args.maxJet
print("pN =",pN)
print("max jet", maxJet)
print("*"*20)


# Define name of the process, folder for the files and xsections
print("isMC = ", isMC)
df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv") if isMC else pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
nanoPath = list(df.nanoPath)[pN]
flatPath = list(df.flatPath)[pN]
if not os.path.exists(flatPath):
    print("Creting flathPath ...", flatPath)
    os.makedirs(flatPath)
process = list(df.process)[pN]
nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
print("Look for ", nanoPath+"/**/*.root")
flatFileNames = glob.glob(flatPath+"/**/*.parquet", recursive=True)
print(process, len(flatFileNames), "/", len(nanoFileNames))
if len(flatFileNames)==len(nanoFileNames):
    sys.exit("Ended here")
    
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

    filePattern = flatPath+"/**/"+process+"_"+str(fileNumber)+".parquet"
    matching_files = glob.glob(filePattern, recursive=True)

    if matching_files:
        continue
    subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 500), '/t3home/gcelotto/ggHbb/flatter/job.sh', nanoFileName, str(maxEntries), str(maxJet), str(pN), process, str(fileNumber), flatPath])
    doneFiles = doneFiles+1
