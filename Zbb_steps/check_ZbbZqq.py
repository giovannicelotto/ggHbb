# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import glob
from functions import cut, getDfProcesses, loadMultiParquet
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
# %%
dfProcess = getDfProcesses()
isMCList = [0,
            20, # ZJetsQQ200-400
            47, # ZJetsBB200-400
            53  # ZJetsqq200-400
            ]
flatPaths = dfProcess.flatPath[isMCList]
dfs = loadMultiParquet(flatPaths, nReal=5, nMC=-1, columns=None, returnFileNumberList=False, selectFileNumberList=None, returnNumEventsTotal=False)
# %%
dfQQ, dfBB, dfqq = dfs[1:]
dfqq, dfBB, dfQQ = cut([dfqq, dfBB, dfQQ], 'jet1_btagDeepFlavB', 0.7100, None)
dfqq, dfBB, dfQQ = cut([dfqq, dfBB, dfQQ], 'jet2_btagDeepFlavB', 0.2783, None)
dfqq, dfBB, dfQQ = cut([dfqq, dfBB, dfQQ], 'jet1_pt', 20, None)
dfqq, dfBB, dfQQ = cut([dfqq, dfBB, dfQQ], 'jet1_pt', 20, None)
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(10, 300, 101)
cqq = ax.hist(dfqq.dijet_mass, bins=bins, label='cc ss uu dd', weights=dfqq.sf*dfqq.PU_SF)[0]
cBB = ax.hist(dfBB.dijet_mass, bins=bins, label='bb stacked', bottom=cqq, weights=dfBB.sf*dfBB.PU_SF)[0]
cQQ = ax.hist(dfQQ.dijet_mass, bins=bins, histtype='step', linewidth=2, label='QQ inclusive', weights=dfQQ.sf*dfQQ.PU_SF)[0]
cBB = ax.hist(dfBB.dijet_mass, bins=bins, histtype='step', label='bb', weights=dfBB.sf*dfBB.PU_SF)[0]
ax.legend()
# %%
cQQ - cBB - cqq
# %%


nReal, nMC = -2, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
isMCList = [20]
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/others/*.parquet"%p))


# %%
import re
featuresForTraining = ['dijet_mass', 'sf', 'PU_SF']

# extract fileNumbers
predictionsFileNumbers = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l = []
    for fileName in predictionsFileNames[idx]:
        fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
        l.append(int(fn))

    predictionsFileNumbers.append(l)

# %%  


#paths = list(dfProcesses.flatPath.values[isMCList])
paths =[
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
        ]
dfs= []
print(predictionsFileNumbers)

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=np.append(featuresForTraining, ['sf', 'PU_SF']), returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)
dfs = preprocessMultiClass(dfs=dfs)


# %%
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

# %%
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])
# %%
