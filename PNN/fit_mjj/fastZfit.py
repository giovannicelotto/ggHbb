# %%
from iminuit.cost import LeastSquares
from numba_stats import crystalball_ex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from helpers.getFeatures import getFeatures
from helpers.preprocessMultiClass import preprocessMultiClass
import re
from helpers.scaleUnscale import scale
from helpers.doPlots import roc, ggHscoreScan
import mplhep as hep
hep.style.use("CMS")
from functions import getXSectionBR, getZXsections
from functions import loadMultiParquet

# %%

nReal, nMC = 1, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 1, 36, 20, 21, 22, 23]
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


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
paths =["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
        ]
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR', 'dijet_dPhi', 'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt'], returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)


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


# given the fn load the data


# preprocess 
dfs = preprocessMultiClass(dfs=dfs)
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])
def model_with_norm(x, norm, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    return  norm * crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)


W_Z = []
for idx, df in enumerate(dfs[2:]):
    w = df.sf*getZXsections()[idx]/numEventsList[idx+2]*nReal*0.774/1017*1000
    W_Z.append(w)


# %%
dfZ = pd.concat(dfs[2:])
W_Z=np.concatenate(W_Z)

bins = np.linspace(40, 300, 51)
wp_Z  =dfZ.PNN>0.678
x = (bins[:-1] + bins[1:])/2
counts = np.histogram( dfZ.dijet_mass[wp_Z], bins=bins, weights=W_Z[wp_Z] )[0]
errors = np.sqrt(np.histogram( dfZ.dijet_mass[wp_Z], bins=bins, weights=(W_Z[wp_Z])**2 )[0])
integral = np.sum(counts * np.diff(bins))

least_squares = LeastSquares(x, counts, errors, model_with_norm)
beta_left, m_left, scale_left = 2.1, 30, 15
beta_right, m_right, scale_right = 1.4, 12, 14
loc = 91
from iminuit import Minuit
m = Minuit(least_squares, integral, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
m.migrad()  # finds minimum of least_squares function
m.hesse() 
# %%
