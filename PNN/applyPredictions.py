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
from hist import Hist
# %%

nReal, nMC = 100, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_dec10"
isMCList = [0, 1, 2, 36, 20, 21, 22, 23]
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/others/*.parquet"%p))
for id in range(len(predictionsFileNames)):
    predictionsFileNames[id] = sorted(predictionsFileNames[id], key=lambda x: int(''.join(filter(str.isdigit, x))))

# %%
featuresForTraining, columnsToRead = getFeatures()
featuresForTraining = list(featuresForTraining)+['dijet_mass']
#featuresForTraining = ['dijet_mass', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'leptonClass', 'dijet_pt']
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

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=np.append(featuresForTraining, ['sf', 'PU_SF', 'jet1_btagDeepFlavB']), returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)



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
#dfs = cut(dfs, 'PNN', 0.4, None)
#dfs = cut(dfs, 'jet1_btagDeepFlavB', 0.2783, None)
#dfs = cut(dfs, 'jet2_btagDeepFlavB', 0.2783, None)
#dfs = cut(dfs, 'muon_pt', 9, None)
# %%
dfs = cut(dfs, 'jet1_btagDeepFlavB', 0.2783, None)
dfs = cut(dfs, 'jet2_btagDeepFlavB', 0.2783, None)
# %%
W_Z = []
for idx, df in enumerate(dfs[2:]):
    w = df.sf*df.PU_SF*getZXsections(EWK=True)[idx]/numEventsList[idx+2]*nReal*0.774/1017*1000
    W_Z.append(w)
dfZ = pd.concat(dfs[2:])
W_Zar=np.concatenate(W_Z)

W_H = dfs[1].sf*dfs[1].PU_SF*getXSectionBR()/numEventsList[1]*nReal*0.774/1017*1000
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 1, 21)
cData = np.histogram(dfs[0].jet2_btagDeepFlavB, bins=bins)[0]
cH = np.histogram(dfs[1].jet2_btagDeepFlavB, bins=bins, weights=W_H)[0]
cZ = np.histogram(dfZ.jet2_btagDeepFlavB, bins=bins, weights=W_Zar)[0]
cData, cH, cZ = cData/np.sum(cData), cH/np.sum(cH), cZ/np.sum(cZ)
ax.hist(bins[:-1], bins=bins, weights=cData, label='Data', histtype='step', linewidth=2)
ax.hist(bins[:-1], bins=bins, weights=cZ, label='Z', histtype='step', linewidth=2)
ax.hist(bins[:-1], bins=bins, weights=cH, label='Higgs', histtype='step', linewidth=2)
ax.set_xlabel("Jet2_btagDeepFlavB")
ax.set_ylabel("Normalized Counts")
ax.legend()
# %%

# %%
# Contamination
print("NHiggs at FullLumi : ", np.sum(W_H*(41.6/0.774*1017/nReal)))
print("Higgs in %d files"%nReal, np.sum(W_H))
contamination = np.sum(W_H)*1e5/len(dfs[0])
print("NHiggs currently present 1e5 events of BParking : ", contamination)
# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 200, 51)
ax.hist(np.clip(dfs[1].dijet_pt, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, color='red', label='GluGluHToBB', weights=W_H, density=True)
ax.hist(np.clip(dfZ.dijet_pt, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, color='orange', label='Zbb', weights=W_Zar, density=True)
ax.hist(np.clip(dfs[0].dijet_pt, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, color='blue', label='Data', density=True)
ax.set_xlabel("Dijet p$_T$ [GeV]")
ax.set_ylabel("Normalized Counts")
hep.cms.label(lumi=np.round(0.774*nReal/1017, 3))
ax.legend()
ax.set_xlim(bins[0], bins[-1])
# %%
from scipy.stats import ks_2samp, chisquare, chi2
from checkOrthogonality import checkOrthogonality

# %%
mask1 = (dfs[0]['jet1_btagDeepFlavB'] >= 0.7100) 
mask2 = (dfs[0]['jet1_btagDeepFlavB'] < 0.7100) 
checkOrthogonality(df=dfs[0], featureToPlot='PNN', mask1=mask1, mask2=mask2, label_mask1='Jet1 btag Tight Pass', label_mask2='Jet1 btag Tight Fail', label_toPlot='Jet2 btagDeepFlavB', bins=np.linspace(0.2783, 1, 51) )
mask1 = (dfs[0]['PNN'] >=0.4) 
mask2 = (dfs[0]['PNN'] < 0.4) 
checkOrthogonality(df=dfs[0], featureToPlot='jet1_btagDeepFlavB', mask1=mask1, mask2=mask2, label_mask1='Jet2 btag Tight Pass', label_mask2='Jet2 btag Tight Fail', label_toPlot='Jet1 btagDeepFlavB', bins=np.linspace(0.2783, 1, 51) )




# %%
from checkOrthogonality import checkOrthogonalityInMassBins
# %%

mass_bins = np.linspace(40, 300, 25)
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=dfs[0],
    featureToPlot='jet2_btagDeepFlavB',
    mask1=(dfs[0]['jet1_btagDeepFlavB'] >= 0.7100),
    mask2=(dfs[0]['jet1_btagDeepFlavB'] < 0.7100),
    label_mask1  = 'Jet1 btag Tight Pass',
    label_mask2  = 'Jet1 btag Tight Fail',
    label_toPlot = 'jet2_btagDeepFlavB',
    bins=np.linspace(0.2783, 1, 51),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30)
)
# %%



def plotLocalPvalues(pvalues, mass_bins, pvalueLabel="KS"):
    labels=[]
    for low, high in zip(mass_bins[:-1], mass_bins[1:]):
        labels.append("%.1f ≤ $m_{jj}$ < %.1f"%(low, high))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(labels, pvalues, color='blue', edgecolor='black')

    # Annotate the bars with the p-value
    for bar, pval in zip(bars, pvalues):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X coordinate
            height,  # Y coordinate
            f"{pval:.3f}",  # Text to display
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Customize the plot
    ax.set_xlabel("Bins", fontsize=16)
    ax.set_ylabel("%s P-value"%(pvalueLabel), fontsize=16)
    #ax.set_title("P-value Distribution Across Bins", fontsize=14)
    ax.set_ylim(0, max(pvalues) * 1.2)  # Add some space above the tallest bar for annotations
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$")
# %%

sum_chi2_PNN = np.sum(chi2_values_PNN)
ddof = (50-1-1)*24 - 1
p_value_PNN = chi2.sf(sum_chi2_PNN, ddof)
print(p_value_PNN)
# %%


# %%
ks_p_value_btag, p_value_btag, chi2_values_btag = checkOrthogonalityInMassBins(
    df=dfs[0],
    featureToPlot='jet1_btagDeepFlavB',
    mask1=(dfs[0]['PNN'] >= 0.4),
    mask2=(dfs[0]['PNN'] < 0.4),
    label_mask1  = 'NN >= 0.4',
    label_mask2  = 'NN < 0.4',
    label_toPlot = 'Jet1 btag',
    bins=np.linspace(0.2783, 1, 51),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30)
)

plotLocalPvalues(pvalues=ks_p_value_btag, mass_bins=mass_bins, pvalueLabel="KS")
plotLocalPvalues(pvalues=p_value_btag, mass_bins=mass_bins, pvalueLabel="$\chi^2$")


# %%
sum_chi2_btag = np.sum(chi2_values_btag)
ddof = (50-1-1)*24 - 1
p_value_btag = chi2.sf(sum_chi2_btag, ddof)
print(p_value_btag)





# %%    
# PNN output score for H and Data

fig, ax = plt.subplots(1, 1)
bins= np.linspace(0, 1, 51)
# First dataset
c = np.histogram(dfs[0].PNN, bins=bins)[0]
c = c / np.sum(c)
ax.hist(
    bins[:-1],
    bins=bins,
    weights=c,
    color='C0',
    histtype='step',
    linewidth=2,
    label='Data'
)

c = np.histogram(dfs[1].PNN, bins=bins, weights=dfs[1].sf * dfs[1].PU_SF)[0]
c = c / np.sum(c)
ax.hist(
    bins[:-1],
    bins=bins,
    weights=c,
    color='C2',
    edgecolor='C2',  # Solid border
    histtype='step',
    linewidth=2,
    label='GluGluHToBB'

)


c = np.histogram(dfZ.PNN, bins=bins, weights=W_Zar)[0]
c = c / np.sum(c)
ax.hist(
    bins[:-1],
    bins=bins,
    weights=c,
    color='C1',
    edgecolor='C1',  # Solid border
    histtype='step',
    linewidth=2,
    label='ZToBB'

)
ax.set_xlabel("NN output score")
ax.legend()
hep.cms.label()
ax.set_ylabel("Normalized Events")
#roc(thresholds=np.linspace(0, 1, 101), signal_predictions=dfs[1].PNN, realData_predictions=dfs[0].PNN, signalTrain_predictions=None, realDataTrain_predictions=None, outName=None)
# %%
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(
    y_true=np.concatenate([np.ones(len(dfs[1])), np.zeros(len(dfs[0]))]),
    y_score=pd.concat([dfs[1].PNN, dfs[0].PNN]))
     
roc_auc = auc(fpr, tpr)

# %%
# Repeat for the Zbb
fpr_Z, tpr_Z, thresholds = roc_curve(
    y_true=np.concatenate([np.ones(len(dfZ)), np.zeros(len(dfs[0]))]),
    y_score=pd.concat([dfZ.PNN, dfs[0].PNN]),
    sample_weight=np.concatenate([W_Zar/np.mean(W_Zar), np.ones(len(dfs[0]))]))
     
roc_auc_Z = auc(fpr_Z, tpr_Z)

# %%
# Plot the ROC curve
fig, ax = plt.subplots(1, 1)
ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC Test Hbb (AUC = {roc_auc:.3f})')
ax.plot(fpr_Z, tpr_Z, color='blue', lw=2, label=f'ROC Test Zbb (AUC = {roc_auc_Z:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
import mplhep as hep
hep.style.use("CMS")
hep.cms.label()
ax.legend(loc='lower right')
ax.grid()


# %%
# ggH score scan
fig, ax = plt.subplots(1, 1,  figsize=(10, 8), constrained_layout=True)
bins = np.linspace(0, 300, 100)
t = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
for i in range(len(t)-1):
        combinedMask = (dfs[0].PNN>t[i]) & (dfs[0].PNN<t[i+1]) 
        ax.hist(dfs[0].dijet_mass[combinedMask], bins=bins, weights=dfs[0][combinedMask].sf, label='%.1f < NN score < %.1f'%(t[i], t[i+1]), histtype=u'step', density=True)[0]

ax.legend()
hep.cms.label()
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Normalized Events")
#fig.patch.set_facecolor('none') 
#ax.set_facecolor('none')    


# %%
fig, ax = plt.subplots(1, 1,  constrained_layout=True)
bins = np.linspace(40, 300, 51)
t = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
for i in range(len(t)-1):
        combinedMask = (dfZ.PNN>t[i]) & (dfZ.PNN<t[i+1]) 
        ax.hist(dfZ.dijet_mass[combinedMask], bins=bins, weights=dfZ[combinedMask].sf, label='%.1f < NN score < %.1f'%(t[i],t[i+1]), histtype=u'step', density=True)[0]
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
    

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, sigList, color='green')
ax.set_ylabel("$S\,/\,\sqrt{B}$")
ax.set_xlabel("NN threshold")
xmax = np.argmax(sigList)
ax.set_ylim(ax.get_ylim())
ax.vlines(x=thresholds[xmax], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dotted')
ax.text(x=0.1, y=0.5, s="Sig(%.2f) = %.3f"%(thresholds[xmax], sigList[xmax]))
#ax.text(x=0.1, y=0.35, s="Muon p$_{T}$ > 9 GeV")
hep.cms.label()
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
ax.set_xlabel("NN output score")
ax.set_ylabel("Normalized Events")
roc(thresholds=np.linspace(0, 1, 101), signal_predictions=dfZ.PNN, realData_predictions=dfs[0].PNN, signalTrain_predictions=None, realDataTrain_predictions=None, outName=None)

# %%
def sig_Z(dfs, W_Z, t):
    massWindowZ_data = (dfs[0].dijet_mass>75) & (dfs[0].dijet_mass<105)
    massWindowZ_Z = (dfs[1].dijet_mass>75) & (dfs[1].dijet_mass<105)
    
    mask_data = dfs[0].PNN>t
    mask_Z = dfs[1].PNN>t
    
    correctionFactor = np.sqrt(41.6/0.774*1017/nReal)
    sig = np.sum(W_Z[(mask_Z) & (massWindowZ_Z)])/np.sqrt(dfs[0][(mask_data) & (massWindowZ_data)].sf.sum() + 1e-7) * correctionFactor
    return sig


thresholds = np.linspace(0.1, .99, 51)
sigList_Z = []
for t in thresholds:
    sigList_Z.append(sig_Z(dfs=[dfs[0], dfZ], W_Z=W_Zar, t=t))
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, sigList_Z, color='green')
ax.set_ylabel("$S\,/\,\sqrt{B}$")
ax.set_xlabel("NN threshold")
xmax = np.argmax(sigList_Z)
ax.set_ylim(ax.get_ylim())
ax.vlines(x=thresholds[xmax], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dotted')
ax.text(x=0.2, y=(ax.get_ylim()[1]+ax.get_ylim()[0])/2, s="Sig(%.3f) = %.3f"%(thresholds[xmax], sigList_Z[xmax]))
hep.cms.label()

# %%
#print("Significance full lumi : ", sigList_Z[xmax]*np.sqrt(41.6/0.774*1017/nReal))
#print("N(Z) @WP : ",       np.sum(W_Z[dfZ.PNN>thresholds[xmax]]))
print("N(Z) @FullLumi : ", np.sum(W_Zar)*(41.6/0.774*1017/nReal))
#print("N(Z) @WP @FullLumi : ", np.sum(W_Z[dfZ.PNN>thresholds[xmax]])*(41.6/0.774*1017/nReal))

print("N(Z) @SR @FullLumi : ", np.sum(W_Zar[(dfZ.PNN>0.4) & (dfZ.jet1_btagDeepFlavB>0.6)])*(41.6/0.774*1017/nReal))

# %%
#fig, ax = plt.subplots(1, 1)
#ax.hist(dfs_precut[0].jet1_btagDeepFlavB, bins=31)

# %%
import shap
import sys
from tensorflow.keras.models import load_model
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from doPlots import getShapNew
# %%
outFolder="/t3home/gcelotto/ggHbb/PNN/results/nov18"
modelName="myModel.h5"
model = load_model(outFolder +"/model/"+modelName)
# %%
#dfZ_scaled  = scale(dfZ, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
#getShapNew(Xtest=dfZ_scaled[featuresForTraining].head(5000), model=model, outName=outFolder+'/performance/shap_Z.png', nFeatures=15, class_names=['NN output'])

dfH_scaled  = scale(dfs[1].copy(), scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
featureImportanceHiggs = getShapNew(Xtest=dfH_scaled[featuresForTraining].head(8000), model=model, outName=outFolder+'/performance/shap_H_SR.png', nFeatures=20, class_names=['Higgs'])
# %%
dfData_scaled  = scale(dfs[0].head(8000), scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
featureImportanceData = getShapNew(Xtest=dfData_scaled[featuresForTraining].head(8000), model=model, outName=outFolder+'/performance/shap_Data_SR.png', nFeatures=20, class_names=['Data'])

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(dfs[1].muon_pt, bins=np.linspace(0, 30, 101), histtype='step', label='muonPt in H events', weights=W_H, density=True)
ax.hist(dfs[0].muon_pt, bins=np.linspace(0, 30, 101), histtype='step', label='muonPt in Data', density=True)
ax.set_yscale('log')
ax.legend()
# %%

# Plot in SR
maskSR_H = (dfs[1].jet1_btagDeepFlavB>0.7100) & (dfs[1].jet2_btagDeepFlavB>0.2783) & (dfs[1].PNN>0.4) 
maskSR_Data = (dfs[0].jet1_btagDeepFlavB>0.7100) & (dfs[0].jet2_btagDeepFlavB>0.2783) & (dfs[0].PNN>0.4) 
maskSR_Z = (dfZ.jet1_btagDeepFlavB>0.7100) & (dfZ.jet2_btagDeepFlavB>0.2783) & (dfZ.PNN>0.4) 
print(np.sum(W_H[maskSR_H])/len(dfs[0][maskSR_Data])*100 , "%")


maskSR_H_leptonClass    = (maskSR_H) & (dfs[1].leptonClass==1)
maskSR_Data_leptonClass = (maskSR_Data) & (dfs[0].leptonClass==1)
print("="*10)
print(len(dfs[0][maskSR_Data_leptonClass])/len(dfs[0][maskSR_Data])*100 , "%")
print(np.sum(W_H[maskSR_H_leptonClass])/np.sum(W_H[maskSR_H])*100 , "%")
print("="*10)
#print(np.sum(W_H[maskSR_H_leptonClass])/len(dfs[0][maskSR_Data_leptonClass])*100 , "%")

fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 2, 3)
cData = ax.hist(dfs[0][maskSR_Data].leptonClass==1, bins=bins ,density=True, histtype='step', label='Data', linewidth=3, color='blue')[0]
cH = ax.hist(dfs[1][maskSR_H].leptonClass==1, bins=bins , weights=W_H[maskSR_H], density=True, histtype='step', label='Hbb', linewidth=3, color='red')[0]
cZ = ax.hist(dfZ[maskSR_Z].leptonClass==1, bins=bins , weights=W_Zar[maskSR_Z], density=True, histtype='step', label='Zbb', linewidth=3, color='green')[0]

cH, cData, cZ = cH/np.sum(cH), cData/np.sum(cData), cZ/np.sum(cZ)
# Plot the percentages on top of the bars
bin_centers = 0.5 * (bins[:-1] + bins[1:])

ax.text(bin_centers[1], cH[1], f"{cH[1]*100:.1f}%", ha='center', va='bottom', fontsize=22, color='red')
ax.text(bin_centers[1], cData[1], f"{cData[1]*100:.1f}%", ha='center', va='bottom', fontsize=22, color='blue')
ax.text(bin_centers[1], cZ[1], f"{cZ[1]*100:.1f}%", ha='center', va='bottom', fontsize=22, color='green')

#for bin_center, height in zip(bin_centers, cData):
#    ax.text(bin_center, height, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=14, color='blue')
#
#for bin_center, height in zip(bin_centers, cZ):
#    ax.text(bin_center, height, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=14, color='green')
hep.cms.label()
ax.set_xlabel("TrigMuon in Jet2")
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['Fail', 'Pass'])
ax.set_yscale('log')
ax.legend()

# %%






# %%

import matplotlib.pyplot as plt
import numpy as np

# Combine datasets
all_keys = set(featureImportanceData.keys()).union(set(featureImportanceHiggs.keys()))
combined_data = {key: (featureImportanceData.get(key, 0), featureImportanceHiggs.get(key, 0)) for key in all_keys}

# Sort by the sum of values and select top 10
sorted_keys = sorted(combined_data.keys(), key=lambda k: sum(combined_data[k]), reverse=True)[:12]

# Data for plotting
labels = sorted_keys
values1 = [combined_data[key][0][0] for key in sorted_keys]
values2 = [combined_data[key][1][0] for key in sorted_keys]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels))
bar_width = 0.4

ax.bar(x, values2, width=bar_width, label='Data', color='blue')
ax.bar(x, values1, width=bar_width, bottom=values2, label='Hbb', color='red')

# Customization
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_xlabel("Features")
ax.set_ylabel("Mean(|SHAP|)")
#ax.set_title("Top 10 Features by Combined Value")
ax.legend()

plt.tight_layout()
plt.show()

# %%
