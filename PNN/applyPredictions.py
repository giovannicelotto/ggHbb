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
from functions import getXSectionBR, getZXsections
from functions import loadMultiParquet, cut

# %%

nReal, nMC = 1, -1


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

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=np.append(featuresForTraining, ['sf', 'PU_SF']), returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)



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
dfs = preprocessMultiClass(dfs=dfs)
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])

# %%


W_Z = []
for idx, df in enumerate(dfs[2:]):
    w = df.sf*df.PU_SF*getZXsections(EWK=True)[idx]/numEventsList[idx+2]*nReal*0.774/1017*1000
    W_Z.append(w)
dfZ = pd.concat(dfs[2:])
W_Zar=np.concatenate(W_Z)

W_H = dfs[1].sf*dfs[1].PU_SF*getXSectionBR()/numEventsList[1]*nReal*0.774/1017*1000
# %%
for id, w in enumerate(W_Z):
    print(w[(dfs[id+2].PNN> 0.4) & (dfs[id+2].jet1_btagDeepFlavB> 0.6)].sum()/w.sum())
#W_Z[(dfZ.PNN> 0.4) & (dfZ.jet1_btagDeepFlavB> 0.6)].sum()*41.6/0.774*1017/nReal
# %%
sorted_pnn = np.sort(dfZ.PNN)
cumulative_pnn = np.cumsum(sorted_pnn) / np.sum(sorted_pnn)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sorted_pnn, cumulative_pnn, color="blue", label="Cumulative Distribution", lw=2)
ax.set_xlabel("PNN", fontsize=14)
ax.set_ylabel("Cumulative Probability", fontsize=14)
ax.set_title("Cumulative Distribution of PNN for ZJets", fontsize=16)
ax.grid(True)


# %%    
# PNN output score for H and Data

fig, ax = plt.subplots(1, 1)
bins= np.linspace(0, 1, 31)
c = np.histogram(dfs[0].PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', color='C0')
ax.hist(bins[:-1], bins=bins, weights=c,color='C0', alpha=0.5)

c = np.histogram(dfs[1].PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c,  histtype=u'step', color='C1')
ax.hist(bins[:-1], bins=bins, weights=c,color='C1', alpha=0.5)
ax.set_xlabel("PNN output score")
ax.set_ylabel("Normalized Events")
roc(thresholds=np.linspace(0, 1, 101), signal_predictions=dfs[1].PNN, realData_predictions=dfs[0].PNN, signalTrain_predictions=None, realDataTrain_predictions=None, outName=None)



# %%
# ggH score scan
fig, ax = plt.subplots(1, 1,  constrained_layout=True)
bins = np.linspace(0, 300, 100)
t = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
for i in range(len(t)):
        combinedMask = (dfs[0].PNN>t[i]) 
        ax.hist(dfs[0].dijet_mass[combinedMask], bins=bins, weights=dfs[0][combinedMask].sf, label='ggH score >%.1f'%t[i], histtype=u'step', density=True)[0]

ax.legend()
#fig.patch.set_facecolor('none') 
#ax.set_facecolor('none')    
# %%
# ggh score scan bin by bin
fig, ax = plt.subplots(1, 1,  constrained_layout=True)
bins = np.linspace(40, 300, 21)
t = [0, 0.4,  1]
for i in range(len(t)-1):
        combinedMask = (dfs[0].PNN>t[i]) & (dfs[0].PNN<t[i+1]) 
        ax.hist(dfs[0].dijet_mass[combinedMask], bins=bins, weights=dfs[0][combinedMask].sf, label='%.1f<ggH<%.1f'%(t[i],t[i+1]), histtype=u'step', density=False)[0]
ax.set_xlim(40, 300)
ax.set_yscale('log')
ax.legend()

# %%
fig, ax = plt.subplots(1, 1,  constrained_layout=True)
bins = np.linspace(40, 300, 21)
t = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
for i in range(len(t)-1):
        combinedMask = (dfZ.PNN>t[i]) & (dfZ.PNN<t[i+1]) 
        ax.hist(dfZ.dijet_mass[combinedMask], bins=bins, weights=dfZ[combinedMask].sf, label='%.1f<ggH<%.1f'%(t[i],t[i+1]), histtype=u'step', density=True)[0]
ax.set_xlim(40, 300)
ax.set_yscale('log')
ax.legend()


# %%
def sig(dfs, W_H, t):
    massWindowH_data = (dfs[0].dijet_mass>100) & (dfs[0].dijet_mass<150)
    massWindowH_H = (dfs[1].dijet_mass>100) & (dfs[1].dijet_mass<150)
    
    mask_data = dfs[0].PNN>t
    mask_H = dfs[1].PNN>t
    

    sig = np.sum(W_H[(mask_H) & (massWindowH_H)])/np.sqrt(dfs[0][(mask_data) & (massWindowH_data)].sf.sum() + 1e-7) *np.sqrt(41.6/0.774*1017/nReal)
    return sig

# %%
thresholds = np.linspace(0, 1, 51)
sigList = []
for t in thresholds:
    sigList.append(sig(dfs=dfs, W_H=W_H, t=t))
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, sigList, color='green')
ax.set_ylabel("S/sqrt(B)")
ax.set_xlabel("PNN threshold")
xmax = np.argmax(sigList)
ax.set_ylim(ax.get_ylim())
ax.vlines(x=thresholds[xmax], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dotted')
ax.text(x=0.2, y=1.8, s="Sig(%.2f) = %.3f"%(thresholds[xmax], sigList[xmax]))
#fig.savefig("/t3home/gcelotto/ggHbb/PNN/sig.png")


# %%
fig, ax = plt.subplots(1, 1)
bins= np.linspace(0, 1, 31)
c = np.histogram(dfs[0].PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', color='C0')
ax.hist(bins[:-1], bins=bins, weights=c,color='C0', alpha=0.5)

c = np.histogram(dfZ.PNN, bins=bins)[0]
c = c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c,  histtype=u'step', color='C1')
ax.hist(bins[:-1], bins=bins, weights=c,color='C1', alpha=0.5)
ax.set_xlabel("PNN output score")
ax.set_ylabel("Normalized Events")
roc(thresholds=np.linspace(0, 1, 101), signal_predictions=dfZ.PNN, realData_predictions=dfs[0].PNN, signalTrain_predictions=None, realDataTrain_predictions=None, outName=None)

# %%
def sig_Z(dfs, W_Z, t):
    massWindowZ_data = (dfs[0].dijet_mass>75) & (dfs[0].dijet_mass<105)
    massWindowZ_H = (dfs[1].dijet_mass>75) & (dfs[1].dijet_mass<105)
    
    mask_data = dfs[0].PNN>t
    mask_H = dfs[1].PNN>t
    num = np.sum(W_Z[(mask_H) & (massWindowZ_H)])
    den = np.sqrt(dfs[0][(mask_data) & (massWindowZ_data)].sf.sum() + 1e-7)
    sig = num/den
    return sig


thresholds = np.linspace(0.1, .99, 51)
sigList_Z = []
for t in thresholds:
    sigList_Z.append(sig_Z(dfs=[dfs[0], dfZ], W_Z=W_Z, t=t))
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, sigList_Z, color='green')
ax.set_ylabel("S/sqrt(B)")
ax.set_xlabel("PNN threshold")
xmax = np.argmax(sigList_Z)
ax.set_ylim(ax.get_ylim())
ax.vlines(x=thresholds[xmax], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dotted')
ax.text(x=0.2, y=(ax.get_ylim()[1]+ax.get_ylim()[0])/2, s="Sig(%.3f) = %.3f"%(thresholds[xmax], sigList_Z[xmax]))

# %%
print("Significance full lumi : ", sigList_Z[xmax]*np.sqrt(41.6/0.774*1017/nReal))
print("N(Z) @WP : ",       np.sum(W_Z[dfZ.PNN>thresholds[xmax]]))
print("N(Z) @FullLumi : ", np.sum(W_Z)*(41.6/0.774*1017/nReal))
print("N(Z) @WP @FullLumi : ", np.sum(W_Z[dfZ.PNN>thresholds[xmax]])*(41.6/0.774*1017/nReal))

print("N(Z) @SR @FullLumi : ", np.sum(W_Z[(dfZ.PNN>0.4) & (dfZ.jet1_btagDeepFlavB>0.6)])*(41.6/0.774*1017/nReal))
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(30, 300, 51)
PNN_t = 0.7
m = dfs[0].PNN>0.7
mH = dfs[1].PNN>0.7
mZ = dfZ.PNN>0.7
ax.hist(dfs[0].dijet_mass[m], bins=bins, histtype='step', color='blue', label='Corrected')
ax.hist(dfs[1].dijet_mass[mH], bins=bins, histtype='step', color='red', label='Corrected')
ax.hist(dfZ.dijet_mass[mZ], bins=bins, histtype='step', color='green', label='Corrected')


ax.text(x=0.9,y=.5,s="PNN>%.1f"%PNN_t, transform=ax.transAxes, ha='right')
ax.legend()


# %%
import shap
import sys
from tensorflow.keras.models import load_model
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from doPlots import getShapNew
outFolder="/t3home/gcelotto/ggHbb/PNN/results/v3b_prova"
modelName="myModel.h5"
model = load_model(outFolder +"/model/"+modelName)
dfZ_scaled  = scale(dfs[2], scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
getShapNew(Xtest=dfZ_scaled[featuresForTraining].head(1000), model=model, outName=outFolder+'/performance/shap_Z.png', nFeatures=15, class_names=['NN output'])

# %%
