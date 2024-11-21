import uproot
import numpy as np
import pandas as pd
import glob, re, sys

def computeMini():
    df = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")

    miniDf = {'process' :   [],
              'fileNumber': [],
              'numEventsPassed':       [],
              'numEventsTotal':       []}
    for (process, nanoPath, xsection) in zip(df.process, df.nanoPath, df.xsection):
        print(process)
        if 'Data' in process:
            continue
        nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
        print("Searching for", nanoPath+"/**/*.root ... %d files found"%len(nanoFileNames))

        for nanoFileName in nanoFileNames:
            try:
                fileNumber = re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1)
            except:
                sys.exit(1)
            f = uproot.open(nanoFileName)
            lumiBlocks = f['LuminosityBlocks']
            numEventsPassed = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])
            numEventsTotal = np.sum(lumiBlocks.arrays()['GenFilter_numEventsTotal'])
            miniDf['process'].append(process)
            miniDf['fileNumber'].append(fileNumber)
            miniDf['numEventsPassed'].append(numEventsPassed)
            miniDf['numEventsTotal'].append(numEventsTotal)
        miniPandasDf = pd.DataFrame(miniDf)
        miniPandasDf.to_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Oct_new.csv")
    miniDf = pd.DataFrame(miniDf)
    miniDf.to_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Oct_new.csv")

    return

if __name__ == "__main__":
    computeMini()