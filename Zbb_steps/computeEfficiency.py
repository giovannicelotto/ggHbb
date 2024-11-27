# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.getFeatures import getFeatures
from helpers.preprocessMultiClass import preprocessMultiClass
import re
from helpers.scaleUnscale import scale
from helpers.doPlots import roc, ggHscoreScan
import mplhep as hep
hep.style.use("CMS")
from functions import getXSectionBR, getZXsections
from functions import loadMultiParquet, cut
# %%

nReal, nMC = 10, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
isMCList = [0, 1, 2, 36, 20, 21, 22, 23]
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/others/*.parquet"%p))


# %%
featuresForTraining, columnsToRead = getFeatures()

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
paths =["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
        ]
dfs= []
print(predictionsFileNumbers)

# *****************************
# *  FIRST KIND OF EFFICIENCY *
# *****************************
efficiencies = []
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=np.append(featuresForTraining, ['sf', 'PU_SF', 'Muon_fired_HLT_Mu9_IP6']), returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)
for id,df in enumerate(dfs):
    if id==0:
        continue
    print((df.sf*df.PU_SF).sum()/numEventsList[id])
    efficiencies.append((df.sf*df.PU_SF).sum()/numEventsList[id])


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

# preprocess 
# *****************************
# *   2nd KIND OF EFFICIENCY  *
# *****************************
eff2nd = []
dfs = preprocessMultiClass(dfs=dfs)
for id,df in enumerate(dfs):
    if id==0:
        continue
    print((df.sf*df.PU_SF).sum()/numEventsList[id])
    eff2nd.append((df.sf*df.PU_SF).sum()/numEventsList[id])
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])


# *****************************
# *   3rd KIND OF EFFICIENCY  *
# *****************************

eff3rd = []
dfs = cut(dfs, 'jet1_btagDeepFlavB', 0.6, None)
dfs = cut(dfs, 'PNN', 0.4, None)
for id,df in enumerate(dfs):
    if id==0:
        continue
    print((df.sf*df.PU_SF).sum()/numEventsList[id])
    eff3rd.append((df.sf*df.PU_SF).sum()/numEventsList[id])
# %%
np.save("/t3home/gcelotto/ggHbb/Zbb_steps/eff1st.npy", efficiencies)
np.save("/t3home/gcelotto/ggHbb/Zbb_steps/eff2nd.npy", eff2nd)
np.save("/t3home/gcelotto/ggHbb/Zbb_steps/eff3rd.npy", eff3rd)
# %%
