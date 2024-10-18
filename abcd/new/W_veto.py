# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from plotDfs import plotDfs
from functions import cut
# %%
# Get predictionsFileNames available for each process
nReal, nMC = 50, -1

predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 1,
            2,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            39]

dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

predictionsFileNames = []
for p in processes:
    print(p)
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


# %%
# Extract predictions fileNumbers
predictionsFileNumbers = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l = []
    for fileName in predictionsFileNames[idx]:
        print
        fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
        l.append(int(fn))

    predictionsFileNumbers.append(l)
# %%
# Load corresponding flattuple for each process
paths = list(dfProcesses.flatPath[isMCList])
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                      columns=None,
                                                               returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                               returnFileNumberList=True)
if isMCList[-1]==39:
    nReal = nReal *2
    print("Duplicating nReal")
# %%
# load only predictions for which there is a corresponding flattuple
preds = []
predictionsFileNamesNew = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l =[]
    for fileName in predictionsFileNames[idx]:
        print(fileName)
        fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
        if fn in fileNumberList[idx]:
            l.append(fileName)
    predictionsFileNamesNew.append(l)
    
    print(len(predictionsFileNamesNew[idx]), " files for process")
    df = pd.read_parquet(predictionsFileNamesNew[idx])
    preds.append(df)

dfs = preprocessMultiClass(dfs=dfs)
# %%
# Add PNN score as a column
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])

# %%
for idx, df in enumerate(dfs):
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
 # make uinque data columns
dfs[0]=pd.concat([dfs[0], dfs[-1]])

dfs = dfs[:-1]
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))


# %%
plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)

# %%
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'
t11=0.1
t12=0.5
t2 =0.3
xx = 'dijet_mass'
dfs_precut = dfs.copy()
# further preprocess


dfs = cut (data=dfs, feature=x1, min=t11, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=t11, max=None)



# Data MC Control plot for dijet mass 
plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)

# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

dfW = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 15) | (isMCList[idx] == 16) | (isMCList[idx] == 17) | (isMCList[idx] == 18) | (isMCList[idx] == 19):
        dfW.append(df)
dfW=pd.concat(dfW)



# %%

sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
plotNormalizedFeatures(data=[dfZ, dfW],
                       outFile="/t3home/gcelotto/ggHbb/abcd/new/features_ZvsW.png", legendLabels=['Z', 'W'],
                       colors=['blue', 'red'], histtypes=[u'step', u'step'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(10,30), autobins=False,
                       weights=[dfZ.weight, dfW.weight], error=True)

