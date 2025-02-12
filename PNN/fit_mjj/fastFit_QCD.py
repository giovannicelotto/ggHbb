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

nReal, nMC = 900, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 39, 1, 36, 20, 21, 22, 23]
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
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data2A",
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

dfs = [pd.concat([dfs[0], dfs[1]]), dfs[2], dfs[3], dfs[4], dfs[5], dfs[6], dfs[7]]

# %%
W_Zlist = []
nReal = 900
nReal = nReal * 2 if nReal!=-1 else 0
for idx, df in enumerate(dfs[2:]):
    print(idx, getZXsections()[idx], numEventsList[idx+3])
    w = df.sf*getZXsections()[idx]/numEventsList[idx+2]*nReal*0.774/1017*1000
    W_Zlist.append(w)


# %%
dfZ = pd.concat(dfs[2:])
W_Z=np.concatenate(W_Zlist)

bins = np.linspace(65, 150, 51)
wp_Z  = (dfZ.PNN>0.678)
x = (bins[:-1] + bins[1:])/2
counts = np.histogram( dfZ[wp_Z].dijet_mass, bins=bins, weights=W_Z[wp_Z])[0]
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

def model_qcd_plus_z(x, p0, p1, p2, p3, tau1):
    xmin=bins[0]
    return  (p0+p1*x+p2*x**2 + p3*x**3)*np.exp(-x/tau1) + 592591 * crystalball_ex.pdf(x, 1.15, 2.9, 18.8, 0.93, 1.38, 14, 93.7)
def polOnly_plus_z(x, p0, p1, p2, tau1):
    xmin=bins[0]
    return  (p0+p1*x+p2*x**2)*np.exp(-x/tau1) + 592591 * crystalball_ex.pdf(x, 1.15, 2.9, 18.8, 0.93, 1.38, 14, 93.7)
#def model_qcd(x, p0, p1, p2, p3,tau1):
#    xmin=bins[0]
#    y_z = np.histogram(dfZ[wp_Z].dijet_mass, bins=bins, weights=W_Z[wp_Z])[0]
#    x_ = (bins[:-1] + bins[1:])/2
#    return (p0+p1*x+p2*x**2+ p3*x**3)*np.exp(-x*tau1) + y_z[(x_<80) | (x_>100) & (x_<300)]
wp_qcd  =dfs[0].PNN>0.678
x = (bins[:-1] + bins[1:])/2
counts = np.histogram( dfs[0].dijet_mass[wp_qcd], bins=bins )[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

#least_squares = LeastSquares(x[(x<89) | (x>91) & (x<150)], counts[(x<89) | (x>91) & (x<150)], errors[(x<89) | (x>91) & (x<150)], model_qcd_plus_z)
least_squares = LeastSquares(x[(x<150)], counts[(x<150)], errors[(x<150)], model_qcd_plus_z)

y_z = np.histogram(dfZ[wp_Z].dijet_mass, bins=bins, weights=W_Z[wp_Z])[0]
norm = np.sum(np.diff(bins)[0] * y_z)
print("Norm ", norm)
from iminuit import Minuit
m = Minuit(least_squares, 9.3e5, -1.7e3, -41, 0.142,127)
m.limits['tau1']=(0,None)
#m.limits['norm']=(0,1e4)
m.migrad()  # finds minimum of least_squares function
m.hesse() 

# %%
def model_qcd(x, p0, p1, p2,p3, tau1):
    xmin=bins[0]
    return  (p0+p1*x+p2*x**2 + p3*x**3)*np.exp(-x/tau1) 
fig, ax = plt.subplots(1, 1)
ax.errorbar(x, counts, yerr=errors, linestyle='none', marker='o', color='black')
ax.plot(x, model_qcd(x, m.values["p0"], m.values["p1"], m.values["p2"], m.values["p3"], m.values["tau1"]))

# %%
y_qcd = model_qcd(x, m.values[0], m.values["p1"], m.values["p2"], m.values["p3"],m.values["tau1"])
y_z = np.histogram(dfZ[wp_Z].dijet_mass, bins=bins, weights=W_Z[wp_Z])[0]

fig, ax = plt.subplots(1, 1)
ax.errorbar(x, counts-y_qcd, yerr=errors, linestyle='none', marker='o', color='black')
ax.plot(x, y_z,  color='red')


# %%
bins=np.linspace(0, 500, 101)
x=(bins[1:] + bins[:-1])/2
fig, ax = plt.subplots(3, 3, figsize=(10,10))
ptcut = [100, 150,175,
         200,250, 300,
         400, 450, 500]
for i in [0,1,2]:
    for j in [0,1,2]:
        idx = i*3+j
        pt = ptcut[idx]

        m = (dfs[0].dijet_pt>pt) & (dfs[0].PNN>0.9)
        mZ = (dfZ.dijet_pt>pt) & (dfZ.PNN>0.9)
        c=ax[i,j].hist(dfs[0].dijet_mass[m], bins=bins, histtype='step', color='black', linestyle='none')[0]
        ax[i,j].errorbar(x, c, yerr=np.sqrt(c), color='black', linestyle='none')
        y_z = np.histogram(dfZ[mZ].dijet_mass, bins=bins, weights=W_Z[mZ])[0]
        ax[i,j].plot(x, y_z,  color='red')

# %%
