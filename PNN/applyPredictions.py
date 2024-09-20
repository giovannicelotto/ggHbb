# %%
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
from functions import getXSectionBR
from functions import loadMultiParquet

# %%

nReal, nMC = 300, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 1]
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
    for fileName in predictionsFileNames[isMC]:
        fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
        l.append(int(fn))

    predictionsFileNumbers.append(l)

# %%  


paths = list(dfProcesses.flatPath.values[isMCList])
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
    for fileName in predictionsFileNames[isMC]:
        fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
        if fn in fileNumberList[idx]:
            l.append(fileName)
    predictionsFileNamesNew.append(l)
    
    print(len(predictionsFileNamesNew[isMC]), " files for process")
    df = pd.read_parquet(predictionsFileNamesNew[isMC])
    preds.append(df)


# given the fn load the data


# preprocess 
dfs = preprocessMultiClass(dfs=dfs)
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])

# %%
fig, ax = plt.subplots(1, 1)
bins= np.linspace(0, 1, 31)
c = np.histogram(dfs[0].PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step')

c = np.histogram(dfs[1].PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c,  histtype=u'step')

roc(thresholds=np.linspace(0, 1, 101), signal_predictions=dfs[1].PNN, realData_predictions=dfs[0].PNN, signalTrain_predictions=None, realDataTrain_predictions=None, outName=None)



# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

bins = np.linspace(0, 300, 100)
t = [0, 0.2, 0.4, 0.6, 0.8, 0.9]


for i in range(len(t)):
        combinedMask = (dfs[0].PNN>t[i]) 
        ax.hist(dfs[0].dijet_mass[combinedMask], bins=bins, weights=dfs[0][combinedMask].sf, label='ggH score >%.1f'%t[i], histtype=u'step', density=True)[0]

ax.legend()
ax.set_title("Dijet Mass : ggH score scan")
# %%


W_H = dfs[1].sf*getXSectionBR()/numEventsList[1]*nReal*0.774/1017*1000
def sig(dfs, W_H, t):
    massWindowH_data = (dfs[0].dijet_mass>100) & (dfs[0].dijet_mass<150)
    massWindowH_H = (dfs[1].dijet_mass>100) & (dfs[1].dijet_mass<150)
    
    mask_data = dfs[0].PNN>t
    mask_H = dfs[1].PNN>t
    

    sig = np.sum(W_H[(mask_H) & (massWindowH_H)])/np.sqrt(dfs[0][(mask_data) & (massWindowH_data)].sf.sum() + 1e-7) *np.sqrt(41.6/0.774*1017/nReal)
    return sig


# %%
from bayes_opt import BayesianOptimization

maxt_qcd, maxt_H = -1, -1
    
pbounds = {
        't': (0., 0.999)}
optimizer = BayesianOptimization(
f=lambda t: sig(dfs, W_H, t),  # lambda to pass the fixed args,
pbounds=pbounds,
verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
random_state=1,
allow_duplicate_points=True
)
    
optimizer.maximize(
    init_points=10,
    n_iter=30,
)
maxt=optimizer.max["params"]["t"]

# %%
thresholds = np.linspace(0, 1, 101)
sigList = []
for t in thresholds:
    sigList.append(sig(dfs=dfs, W_H=W_H, t=t))
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, sigList, color='green')
ax.set_ylabel("S/sqrt(B)")
ax.set_xlabel("PNN threshold")
xmax = np.argmax(sigList)
ax.set_ylim(ax.get_ylim())
ax.vlines(x=thresholds[xmax], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dotted')
ax.text(x=0.4, y=1.9, s="Sig(%.2f) = %.2f"%(thresholds[xmax], sigList[xmax]))
# %%
