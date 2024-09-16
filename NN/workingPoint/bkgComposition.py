# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, re, glob
from functions import loadMultiParquet, cut, getXSectionBR
sys.path.append("/t3home/gcelotto/ggHbb/NN")
from helpersForNN import preprocessMultiClass, scale, unscale
import mplhep as hep
hep.style.use("CMS")
from applyMultiClass_Hpeak import getPredictions, splitPtFunc
# %%

# load data

pTClass, nReal, nMC = 0, 300, 100
dfProcess=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
isMCList = [0, # Data
            1, # GluGlu
            3, 4, 5, #diboson
            6, 7, 8, 9, 10,11, #ST
             12, 13, 14, # ttbar
             15, 16, 17, 18, 19, # Wjets
            20, 21, 22, 23, # ZJets
             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, # QCD MU enriched
             36, #ZJets
            ]
# list of all the flatPaths
flatPath = list(np.array(dfProcess.flatPath)[isMCList])


# check for which fileNumbers the predictions is available
pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
fileNumberList = []
processNames = dfProcess.process
for isMC in isMCList:
    fileNumberProcess = []
    fileNamesProcess = glob.glob(pathToPredictions+"/yMC%d_fn*pt%d*.parquet"%(isMC, pTClass))
    for fileName in fileNamesProcess:
        match = re.search(r'_fn(\d+)_pt', fileName)
        if match:
            fn = match.group(1)
            fileNumberProcess.append(int(fn))
            
        else:
            pass
            #print("Number not found")
    fileNumberList.append(fileNumberProcess)
    print(len(fileNumberProcess), " predictions files for process MC : ", isMC)


# load the files where the prediction is available
columns = ['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR', 'dijet_dPhi', 'jet3_pt', 'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt']
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=flatPath, nReal=nReal, nMC=nMC,
                                                      columns=columns, returnNumEventsTotal=True,
                                                      selectFileNumberList=fileNumberList, returnFileNumberList=True)
pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][pTClass]    
lengthPreCut = []
for df_ in dfs:
    lengthPreCut.append(len(df_))

dfs = preprocessMultiClass(dfs, leptonClass=None, pTmin=pTmin, pTmax=pTmax, suffix=suffix)   # get the dfs with the cut in the pt class

# %%
# Load the predictions
minPt, maxPt = None, None #180, -1
if (minPt is not None) | (maxPt is not None):
    dfs, masks = splitPtFunc(dfs, minPt, maxPt)
    splitPt = True
else:
    masks=None
    splitPt=False
YPred = list(getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass))



# %%
bins = np.linspace(40, 200, 41)
MCCounts = {
    'ggH':np.zeros(len(bins)-1),
    'VV':np.zeros(len(bins)-1),
    'Z':np.zeros(len(bins)-1),
    'ST':np.zeros(len(bins)-1),
    'tt':np.zeros(len(bins)-1),
    'W':np.zeros(len(bins)-1),
    'QCD':np.zeros(len(bins)-1),
}
dataCounts = np.zeros(len(bins)-1)
for idx, df in enumerate(dfs):
    print("Process : %s"%processNames[idx])
    
    mask = (YPred[idx][:,0]<0.22) & (YPred[idx][:,1]>0.34)


    if numEventsList[idx]!=0:
        counts_ = np.histogram(df.dijet_mass[mask], bins=bins, weights=(df.sf*df.PU_SF)[mask])[0]
        counts_ = counts_/(numEventsList[idx]+1e-7) * dfProcess.xsection[idx] * 1000 * nReal/1017 * 0.774
        
        if 'ST' in processNames[idx]:
            MCCounts['ST'] = MCCounts['ST'] + counts_
        elif 'TTTo' in processNames[idx]:
            MCCounts['tt'] = MCCounts['tt'] + counts_
        elif ('WW' in processNames[idx]) | ('ZZ' in processNames[idx]) | ('WZ' in processNames[idx]):
            MCCounts['VV'] = MCCounts['VV'] + counts_
        elif ('WJets' in processNames[idx]):
            MCCounts['W'] = MCCounts['W'] + counts_
        elif ('ZJets' in processNames[idx]) | ('EWKZJets' in processNames[idx]):
            MCCounts['Z'] = MCCounts['Z'] + counts_
        elif ('GluGluHToBB' in processNames[idx]):
            MCCounts['ggH'] = MCCounts['ggH'] + counts_
        elif ('QCD' in processNames[idx]):
            MCCounts['QCD'] = MCCounts['QCD'] + counts_
    else:
        assert ('Data' in processNames[idx])
        counts_ = np.histogram(df.dijet_mass[mask], bins=bins, weights=(df.sf)[mask])[0]
        dataCounts = counts_
        

# %%
fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 12))
tempStackedArray = np.zeros(len(bins)-1)
for process, counts in zip(MCCounts.keys(), MCCounts.values()):
    if (counts==0).all():
        continue
    c_ = ax1.hist(bins[:-1], bins=bins, weights=counts, bottom=tempStackedArray, label=process)[0]
    tempStackedArray = tempStackedArray + c_
x = (bins[1:] + bins[:-1])/2
ax1.errorbar(x, dataCounts, yerr=np.sqrt(dataCounts), label='Data', color='black', linestyle='none', marker='o')
ax1.legend()
ax1.set_xlim(bins[0], bins[-1])
ax1.set_yscale('log')
hep.cms.label(ax=ax1)

MCSum = np.zeros(len(bins)-1)
for value in MCCounts.values():
    MCSum = MCSum + value

ax2.errorbar(x, dataCounts/MCSum, yerr=np.sqrt(dataCounts)/MCSum, color='black', linestyle='none', marker='o')


# %%
