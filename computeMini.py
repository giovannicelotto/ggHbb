# %%
import uproot
import numpy as np
import pandas as pd
import glob, re, sys
import argparse
from functions import getDfProcesses_v2
# %%
try:
    parser = argparse.ArgumentParser(description="Script.")
    parser.add_argument("-pN", "--processNumber", type=int, help="e.g. Number of the MC process", default=0)
    args = parser.parse_args()
    pN = args.processNumber
except:
    pN = 0
# %%
df = getDfProcesses_v2()[0].iloc[pN]
# %%
miniDf = {'process' :   [],
            'fileNumber': [],
            #'numEventsPassed':       [],
            'genEventCount':       [],
            'genEventSumw':[],
            'genEventSumw2':[]}
process, nanoPath, xsection = df.process, df.nanoPath, df.xsection
# %%
print(process)
# %%


nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
print("Searching for", nanoPath+"/**/*.root ... %d files found"%len(nanoFileNames))
# %%
for idx, nanoFileName in enumerate(nanoFileNames):
    print(idx+1, " / ", len(nanoFileNames))
    try:
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1)
    except:
        sys.exit(1)
    f = uproot.open(nanoFileName)
    #lumiBlocks = f['LuminosityBlocks']
    Runs = f['Runs']
    miniDf['process'].append(process)
    miniDf['fileNumber'].append(fileNumber)
    miniDf['genEventCount'].append(Runs.arrays()['genEventCount'][0])
    miniDf['genEventSumw'].append(Runs.arrays()['genEventSumw'][0])
    miniDf['genEventSumw2'].append(Runs.arrays()['genEventSumw2'][0])
# %%
miniPandasDf = pd.DataFrame(miniDf)
miniPandasDf.to_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_%s.csv"%process)

# %%
