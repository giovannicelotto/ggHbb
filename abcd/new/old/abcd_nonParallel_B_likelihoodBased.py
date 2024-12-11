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
# %%

nReal, nMC = 100, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_v3b"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            39    # Data2A
]

dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    print(p)
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


# %%
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
paths = list(dfProcesses.flatPath[isMCList])
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                      columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt',
                                                               'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                               'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR',
                                                                'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt',
                                                               'jet2_btagDeepFlavB', 'dijet_cs',
                                                               'jet1_btagDeepFlavB'],
                                                               returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                               returnFileNumberList=True)
if isMCList[-1]==39:
    nReal = nReal *2
    print("Duplicating nReal")
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

# %%
for idx, df in enumerate(dfs):
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
 # make uinque data columns
if isMCList[-1]==39:
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
# remove the last element (data2a)
    dfs = dfs[:-1]
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))

# %%
# Add weight columns
for idx, df in enumerate(dfs):
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)
dfs_precut = dfs.copy()

# %%
# Define variables regions and cuts
x1 = 'dijet_cs_abs'
x2 = 'PNN'
t11=0.3
t12=0.3
t21 =0.5
t22 = 0.5
xx = 'dijet_mass'
# further preprocess
from functions import cut
#dfs = cut (data=dfs, feature='PNN', min=0.3, max=None
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)


# %%
fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/dataMC_stacked.png")


# %%
# Define Z dataframe and quantify in the regions
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

mA      = (dfZ[x1]<t11 ) & (dfZ[x2]>t22 ) 
mB      = (dfZ[x1]>t12 ) & (dfZ[x2]>t22 ) 
mC      = (dfZ[x1]<t11 ) & (dfZ[x2]<t21 ) 
mD      = (dfZ[x1]>t12 ) & (dfZ[x2]<t21 ) 



print("Region A : ", np.sum(dfZ.weight[mA])/dfZ.weight.sum())
print("Region B : ", np.sum(dfZ.weight[mB])/dfZ.weight.sum())
print("Region C : ", np.sum(dfZ.weight[mC])/dfZ.weight.sum())
print("Region D : ", np.sum(dfZ.weight[mD])/dfZ.weight.sum())


# %%
# Fill regions with data
bins = np.linspace(40, 300, 2)
x=(bins[1:]+bins[:-1])/2
regions = {
    'A':np.zeros(len(bins)-1),
    'B':np.zeros(len(bins)-1),
    'D':np.zeros(len(bins)-1),
    'C':np.zeros(len(bins)-1),
}


mA      = (dfs[0][x1]<t11 ) & (dfs[0][x2]>t22 ) 
mB      = (dfs[0][x1]>t12 ) & (dfs[0][x2]>t22 ) 
mC      = (dfs[0][x1]<t11 ) & (dfs[0][x2]<t21 ) 
mD      = (dfs[0][x1]>t12 ) & (dfs[0][x2]<t21 ) 
regions['A'] = regions['A'] + np.histogram(dfs[0][mA][xx], bins=bins)[0]
regions['B'] = regions['B'] + np.histogram(dfs[0][mB][xx], bins=bins)[0]
regions['C'] = regions['C'] + np.histogram(dfs[0][mC][xx], bins=bins)[0]
regions['D'] = regions['D'] + np.histogram(dfs[0][mD][xx], bins=bins)[0]
print("Data counts in ABCD regions")
print("Region A : ", regions["A"].sum())
print("Region B : ", regions["B"].sum())
print("Region C : ", regions["C"].sum())
print("Region D : ", regions["D"].sum())
regions['Bdepleted'] = regions["B"]
# remove MC simulations from a, c, d
for idx, df in enumerate(dfs[1:]):
    print(idx, df.dijet_mass.mean())
    mA      = (df[x1]<t11 ) & (df[x2]>t22 ) 
    mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 
    mC      = (df[x1]<t11 ) & (df[x2]<t21 ) 
    mD      = (df[x1]>t12 ) & (df[x2]<t21 ) 
    regions['A'] = regions['A'] - np.histogram(df[mA][xx], bins=bins, weights=df[mA].weight)[0]
    regions['Bdepleted'] = regions['Bdepleted'] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['C'] = regions['C'] - np.histogram(df[mC][xx], bins=bins, weights=df[mC].weight)[0]
    regions['D'] = regions['D'] - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight)[0]


# %%
from scipy.optimize import minimize
from scipy.stats import poisson
import numpy as np
from scipy.stats import norm


def negative_log_likelihood(params):
    N_D_tilde, N_C_tilde, m = params
    # Poisson likelihoods for each region
    log_likelihood_B = poisson.logpmf(N_B, m * N_D_tilde)
    log_likelihood_D = poisson.logpmf(N_D, N_D_tilde)
    log_likelihood_A = poisson.logpmf(N_A, m * N_C_tilde)
    log_likelihood_C = poisson.logpmf(N_C, N_C_tilde)
    
    # Total negative log-likelihood (negated for minimization)
    return -(log_likelihood_A + log_likelihood_B + log_likelihood_C + log_likelihood_D)

def negative_log_likelihood_gaussian(params):
    N_D_tilde, N_C_tilde, m = params
    # Gaussian approximation for large counts
    log_likelihood_B = norm.logpdf(N_B, loc=m * N_D_tilde, scale=np.sqrt(m * N_D_tilde))
    log_likelihood_D = norm.logpdf(N_D, loc=N_D_tilde, scale=np.sqrt(N_D_tilde))
    log_likelihood_A = norm.logpdf(N_A, loc=m * N_C_tilde, scale=np.sqrt(m * N_C_tilde))
    log_likelihood_C = norm.logpdf(N_C, loc=N_C_tilde, scale=np.sqrt(N_C_tilde))
    
    # Total negative log-likelihood (negated for minimization)
    return -(log_likelihood_A + log_likelihood_B+ log_likelihood_C + log_likelihood_D)

# Perform the fit using Gaussian likelihood




# %%
mL, mUL = [], []
for i in range(len(regions["A"])):
    N_A, N_B, N_C, N_D = regions["A"][i], regions["Bdepleted"][i], regions["C"][i], regions["D"][i]
    print(N_A)
    initial_guess = [N_D, N_C, 1.0] 
    bounds = [(0, None), (0, None), (0.01, 10)] 
    result = minimize(negative_log_likelihood_gaussian, initial_guess, bounds=bounds, method='L-BFGS-B')
    mL.append(result.x[2])
    print(f"Bin {i}: Best fit parameters: N_D = {result.x[0]}, N_C = {result.x[1]}, m = {result.x[2]}")

    cov_matrix = result.hess_inv.todense()  # Inverse Hessian gives the covariance matrix
    param_errors = np.sqrt(np.diag(cov_matrix))
    mUL.append(param_errors[2])

mUL = np.array(mUL)
mL = np.array(mL)
fig, ax = plt.subplots(1, 1)
ax.errorbar(list(range(len(regions["A"]))), mL, yerr=mUL)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/likelihood_m.png")

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
ax[0].hist(bins[:-1], bins=bins, weights=regions["Bdepleted"])
ax[0].errorbar(x, mL*regions["D"], yerr=mUL, linestyle='none', marker='o')
yerr = regions["Bdepleted"]/(mL*regions["D"]) * np.sqrt(mUL**2/mL**2 + 1/regions["Bdepleted"] + 1/regions["D"])
ax[1].errorbar(x, regions["Bdepleted"]/(mL*regions["D"]), yerr=yerr, color='black', linestyle='none', marker='o')
ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
ax[1].set_xlim(bins[0], bins[-1])
ax[1].set_ylim(0.95, 1.05)
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/depleted_minus_qcd_abcd.png")

qcd_abcd = mL * regions["D"]

# %%

fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
x=(bins[1:]+bins[:-1])/2
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["Bdepleted"], histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')

ax[0,1].hist(bins[:-1], bins=bins, weights=qcd_abcd, histtype=u'step', label=r'$A\times D / C$ ')
#ax[0,1].errorbar(x, regions["Bdepleted"], yerr=np.sqrt(regions["Bdepleted"]), linestyle='none', color='black', marker='o')


ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t11, x2, t22), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t12, x2, t22), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t11, x2, t21), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t12, x2, t21), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/controlregions_likelihood.png")

# %%
x = (bins[1:] + bins[:-1])/2
data_minus_qcd = regions['B']-qcd_abcd


fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
b_err = np.sqrt(regions['B'])
adc_err = regions['A']*regions['D']/regions['C']*np.sqrt(1/regions['A'] + 1/regions['D'] + 1/regions['C'])
ax[0].errorbar(x,data_minus_qcd, yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
cTot = np.zeros(len(bins)-1)
countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
    }

for idx, df in enumerate(dfs[1:]):
    isMC = isMCList[idx+1]
    process = dfProcesses.process[isMC]
    mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 
    c = np.histogram(df.dijet_mass[mB], bins=bins,weights=df.weight[mB])[0]
    if 'Data' in process:
        countsDict['Data'] = countsDict['Data'] + c
        print("adding data with", process)
    elif 'GluGluHToBB' in process:
        countsDict['H'] = countsDict['H'] + c
    elif 'ST' in process:
        countsDict['ST'] = countsDict['ST'] + c
    elif 'TTTo' in process:
        countsDict['ttbar'] = countsDict['ttbar'] + c
    elif 'QCD' in process:
        countsDict['QCD'] = countsDict['QCD'] + c
    elif 'ZJets' in process:
        countsDict['Z+Jets'] = countsDict['Z+Jets'] + c
    elif 'WJets' in process:
        countsDict['W+Jets'] = countsDict['W+Jets'] + c
    elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
        countsDict['VV'] = countsDict['VV'] + c

    #c = ax[0].hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, label=dfProcesses.process[isMC])[0]
    
for key in countsDict.keys():
    if np.sum(countsDict[key])==0:
        continue
    print(key, np.sum(countsDict[key]))
    ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=cTot, label=key)
    cTot = cTot + countsDict[key]
ax[0].legend()

ax[1].set_xlim(ax[1].get_xlim())    
ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[1].set_ylim(0.5, 1.5)
ax[1].errorbar(x, data_minus_qcd/mc, yerr=np.sqrt(b_err**2 + adc_err**2)/mc , marker='o', color='black', linestyle='none')

fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/likelihood_mc_comparison.png")
#ax.set_yscale('log')
# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))
# %%




qcd_mc = regions['B'] - mc
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


ax[0].hist(bins[:-1], bins=bins, weights=(qcd_abcd), label='QCD = ABCD estimation', histtype='step')
ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc, label='QCD = B - MC estimation[B]', histtype='step')
ax[1].errorbar(x, qcd_abcd/qcd_mc, yerr=adc_err/qcd_mc,linestyle='none', marker='o', color='black')
ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
ax[1].set_xlim(bins[0], bins[-1])
ax[1].set_ylim(0.9, 1.1)
ax[0].legend()
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/qcd_control_likelihood.png")



# %%
