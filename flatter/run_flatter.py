import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time
import argparse
from functions import getDfProcesses_v2

parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-MC", "--isMC", type=int, help="isMC True or False", default=None)
parser.add_argument("-JEC", "--isJEC", type=int, help="iIs a JEC variation", default=0)
parser.add_argument("-pN", "--processNumber", type=int, help="process Number for MC or datataking for Data", default=None)
parser.add_argument("-n", "--nFiles", type=int, help="number of files to flatten per process", default=-1)
parser.add_argument("-mE", "--maxEntries", type=int, help="max number of Entries", default=-1)
parser.add_argument("-mJ", "--maxJet", type=int, help="max number of jet", default=4)
parser.add_argument("-m", "--method", type=int, help="method of selecting jets", default=-1)

args = parser.parse_args()

isMC            = True if args.isMC==1 else False
pN              = args.processNumber
isJEC              = args.isJEC
nFiles          = args.nFiles
maxEntries      = args.maxEntries
maxJet          = args.maxJet
method          = args.method

if method==1:
    maxJet=999
assert method!=-1, "Select a method -m 0 or 1"
print("pN =",pN)
print("max jet", maxJet)
print("*"*20)


# Define name of the process, folder for the files and xsections
print("isMC = ", isMC)
if isMC:
    if isJEC:
        df=getDfProcesses_v2()[2]
    else:
        df=getDfProcesses_v2()[0]
else:
    df=getDfProcesses_v2()[1]
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



# Delete all *.out files
#target_path = "/t3home/gcelotto/slurm/output/flat"
#for file_path in glob.glob(target_path+"/*.out"):
#    if os.path.isfile(file_path):
#        os.remove(file_path)

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
    subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 100), '/t3home/gcelotto/ggHbb/flatter/job.sh', nanoFileName, str(maxEntries), str(maxJet), str(pN), process, str(fileNumber), flatPath, str(method), str(isJEC)])
    doneFiles = doneFiles+1
