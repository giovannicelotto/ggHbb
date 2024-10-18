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
from loadDataAndPredictions import loadDataAndPredictions
from plotDfs import plotDfs
# %%

nReal, nMC = 1000, -1


predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_v3b"
isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            #39    # Data2A
]

dfs, numEventsList, preds,dfProcesses = loadDataAndPredictions(isMCList=isMCList, predictionsPath=predictionsPath,
                                    nReal=nReal, nMC=nMC)

# given the fn load the data


# preprocess 
dfs = preprocessMultiClass(dfs=dfs)
# %%
#Add PNNm dijet_cs_abs and weight
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)

    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
dfs_precut = dfs.copy()
# %%
# Weights of data = 1

if (isMCList[-1]==39) & (len(dfs)==len(isMCList)):
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
    dfs = dfs[:-1]
dfs[0]['weight'] = np.ones(len(dfs[0]))



# %%
# Define variables and regions
x1 = 'dijet_cs_abs'
x2 = 'PNN'
t11=0.5
t12=0.5
t21 =0.5
t22 = 0.5
xx = 'dijet_mass'

from functions import cut
dfs = cut (data=dfs, feature='jet1_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='jet2_btagDeepFlavB', min=0.3, max=None)
dfs = cut (data=dfs, feature='leptonClass', min=None, max=1.1)
#dfs = cut (data=dfs, feature='dijet_pt', min=50, max=None)


# %%
#fig = plotDfs(dfs=dfs, isMCList=isMCList, dfProcesses=dfProcesses)
#fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/dataMC_stacked.png")


# %%
dfZ = []
for idx,df in enumerate(dfs):
    if (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
        dfZ.append(df)
dfZ=pd.concat(dfZ)

# %%
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

bins = np.linspace(40, 300, 15)
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

regions["Bdepleted"] = regions["B"]
## remove MC simulations from a, b, c
for idx, df in enumerate(dfs[1:]):
    print(idx, df.dijet_mass.mean())
    mA      = (df[x1]<t11 ) & (df[x2]>t22 ) 
    mB      = (df[x1]>t12 ) & (df[x2]>t22 ) 
    mC      = (df[x1]<t11 ) & (df[x2]<t21 ) 
    mD      = (df[x1]>t12 ) & (df[x2]<t21 ) 
    regions['A'] = regions['A'] - np.histogram(df[mA][xx], bins=bins, weights=df[mA].weight)[0]
    regions["Bdepleted"] = regions["Bdepleted"] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]
    regions['C'] = regions['C'] - np.histogram(df[mC][xx], bins=bins, weights=df[mC].weight)[0]
    regions['D'] = regions['D'] - np.histogram(df[mD][xx], bins=bins, weights=df[mD].weight)[0]
# %%
# Plot transfer factor
fig, ax = plt.subplots(1, 1)
ax.set_title("Transfer factor dependence")
ax.errorbar(x, regions["A"]/regions["C"], yerr = regions["A"]/regions["C"]*np.sqrt(np.sqrt(1/regions["A"]  + 1/regions["C"])), marker='o', label='A/C')
ax.errorbar(x+2, regions["Bdepleted"]/regions["D"], yerr=regions["Bdepleted"]/regions["D"]*np.sqrt(1/regions["Bdepleted"] + 1/regions["D"]), marker='o', label='B/D')
ax.legend()
# %%
# write here the logL
#import numpy as np
#from scipy.stats import poisson
#from iminuit import Minuit
#from scipy.special import gammaln
#
## Define the log of the Poisson PMF
#def log_poisson_pmf(N, Ntilde):
#    return N * np.log(Ntilde) - Ntilde - gammaln(N + 1)
#
## Extract the relevant data from the regions
#Na = regions["A"]
#Nc = regions["C"]
#Nd = regions["D"]
#Ntot = Na + Nc + Nd
#
## Define the log-likelihood function
#def logL(params):
#    Nd_tilde, Nc_tilde, m_tilde = params  # Unpack the parameters
#    Ntot_tilde = Nd_tilde + Nc_tilde + m_tilde * Nc_tilde
#
#    # Compute the log of the Poisson PMF
#    pois_log = log_poisson_pmf(Ntot, Ntot_tilde)
#
#    # Compute the logs of the other terms
#    log_term1 = np.log(Nc_tilde) - np.log(Ntot_tilde)
#    log_term2 = np.log(m_tilde * Nc_tilde) - np.log(Ntot_tilde)
#    log_term3 = np.log(Nd_tilde) - np.log(Ntot_tilde)
#
#    # The total log-likelihood is the sum of these terms
#    log_likelihood = pois_log + log_term1 + log_term2 + log_term3
#    return -log_likelihood  # Return negative log-likelihood for minimization
#
#from scipy.optimize import minimize
#m_tilde_opt, Nd_tilde_opt, Nc_tilde_opt = [], [], []
#for i in range(len(bins)-1):
#    Na=regions["A"][i]
#    Nc=regions["C"][i]
#    Nd=regions["D"][i]
#    Ntot = Na + Nc +Nd
#    initial_guess = np.array([regions["D"][i], regions["C"][i], regions["A"][i]/regions["C"][i]])
#    result = minimize(logL, x0=initial_guess)
#    Nd_tilde_opt.append(result.x[0])
#    Nc_tilde_opt.append(result.x[1])
#    m_tilde_opt.append(result.x[2])
#
## Output the result
#    print("Minimized function value:", result.fun)
#    print("Optimal values for x, y, z:", result.x)
#
#
## %%
## Fit with i iminuit
#initial_guess = (regions["D"], regions["C"], regions["A"]/regions["C"])
#
#
#def logL(Nd_tilde,Nc_tilde,m_tilde):
#    Ntot_tilde = Nd_tilde + Nc_tilde + m_tilde * Nc_tilde
#
#    # Compute the log of the Poisson PMF
#    pois_log = log_poisson_pmf(Ntot, Ntot_tilde)
#
#    # Compute the logs of the other terms
#    log_term1 = np.log(Nc_tilde) - np.log(Ntot_tilde)
#    log_term2 = np.log(m_tilde * Nc_tilde) - np.log(Ntot_tilde)
#    log_term3 = np.log(Nd_tilde) - np.log(Ntot_tilde)
#
#    # The total log-likelihood is the sum of these terms
#    log_likelihood = pois_log + log_term1 + log_term2 + log_term3
#    return -log_likelihood  # Return negative log-likelihood for minimization
#
#m_tilde_opt, Nd_tilde_opt, Nc_tilde_opt = [], [], []
#for i in range(len(bins)-1):
#    Na=regions["A"][i]
#    Nc=regions["C"][i]
#    Nd=regions["D"][i]
#    Ntot = Na + Nc +Nd
#    initial_guess = np.array([regions["D"][i], regions["C"][i], regions["A"][i]/regions["C"][i]])
#    m = Minuit(logL, Nd_tilde=initial_guess[0], Nc_tilde=initial_guess[1], m_tilde=initial_guess[2])
#
#
##
### Perform the minimization
#    m.migrad()  # Minimize the negative log-likelihood using the MIGRAD algorithm
#    Nd_tilde_opt.append(m.values[0])
#    Nc_tilde_opt.append(m.values[1])
#    m_tilde_opt.append(m.values[2])
##
### Print the results
#    print(m.values)    # Fitted parameter values
#    print(m.errors)    # Uncertainties of the fitted parameters


# %%

# %%
#fig, ax = plt.subplots(1, 1)
#x = np.linspace(0.15, 1, 100)
#ax.plot(x, logL(Nd, Nc, x))
#
#x = np.linspace(5160, 5300, 10)
#ax.plot(x, logL(Nd, x, .35))
#ax.set_xlim(5.15e3, 5.3e3)
#ax.set_ylim(8,14)















# %%
#m_tilde_opt=np.array(m_tilde_opt)
#Nc_tilde_opt=np.array(Nc_tilde_opt)
#Nd_tilde_opt=np.array(Nd_tilde_opt)
fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"], histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')

#Division
ax[0,1].hist(bins[:-1], bins=bins, weights=regions['A']*regions['D']/(regions['C']+1e-6), histtype=u'step', label=r'$A\times D / C$ ')
#ax[0,1].errorbar(x, regions["B"], yerr=np.sqrt(regions["B"]), linestyle='none', color='black', marker='o')

# MLE
#ax[1,0].hist(bins[:-1], bins=bins, weights=Nc_tilde_opt, histtype=u'step', color='red')
#ax[1,1].hist(bins[:-1], bins=bins, weights=Nd_tilde_opt, histtype=u'step', color='red')
#ax[0,1].hist(bins[:-1], bins=bins, weights=m_tilde_opt*regions['D'], histtype=u'step')
#ax[0,0].hist(bins[:-1], bins=bins, weights=m_tilde_opt*Nc_tilde_opt, histtype=u'step', color='red')
#ax[0,1].legend()
#ax[0,1].set_ylim(np.max(m_tilde_opt*regions['D'])*0.95, np.max(m_tilde_opt*regions['D'])*1.05)

ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, t11, x2, t22), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, t12, x2, t22), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, t11, x2, t21), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, t12, x2, t21), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/controlregions.png")

# %%
x = (bins[1:] + bins[:-1])/2

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
b_err = np.sqrt(regions['B'])
adc_err = regions['A']*regions['D']/regions['C']*np.sqrt(1/regions['A'] + 1/regions['D'] + 1/regions['C'])
ax[0].errorbar(x, regions['B']-regions['A']*regions['D']/regions['C'], yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
#ax[0].errorbar(x, regions['B']-m_tilde_opt*regions['D'], yerr=np.sqrt(b_err**2 + adc_err**2) , marker='x', color='gray', linestyle='none')
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
data = regions['B']-regions['A']*regions['D']/regions['C']
mc = countsDict['Z+Jets'] + countsDict['W+Jets'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['H'] + countsDict['VV']
ax[1].set_ylim(0.5, 1.5)
ax[1].errorbar(x, data/mc, yerr=np.sqrt(b_err**2 + adc_err**2)/mc , marker='o', color='black', linestyle='none')
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/dimuon_control.png")

# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))

qcd_mc = regions['B'] - countsDict['H'] - countsDict['ttbar'] - countsDict['ST'] - countsDict['VV'] - countsDict['VV'] - countsDict['W+Jets'] - countsDict['Z+Jets']
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


ax[0].hist(bins[:-1], bins=bins, weights=(regions['A']*regions['D']/regions['C']), label='QCD = ABCD estimation', histtype='step')
ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc, label='QCD = B - MC estimation[B]', histtype='step')
ax[1].errorbar(x, regions['A']*regions['D']/regions['C']/qcd_mc, yerr=adc_err/qcd_mc,linestyle='none', marker='o', color='black')
ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
ax[1].set_xlim(bins[0], bins[-1])
ax[1].set_ylim(0.9, 1.1)
ax[0].legend()
fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/qcd_control.png")



# %%
import dcor
df = dfs[0]
#distance_correlation
#u_distance_correlation_sqr
cuts = [0,0.1,0.2,0.3]
xLog = [1e3, 2e3, 5e3, 8e3,1e4, 1e5]
fig, ax = plt.subplots(1,1 )
for cut in cuts:
    corrList =[]
    for i in xLog:

        i=int(i)
        mask = (df.jet1_btagDeepFlavB>cut) & (df.jet2_btagDeepFlavB>cut)
        print("Correlation bewteen %s and %s in %d events"%(x1, x2, i))
        m = dcor.u_distance_correlation_sqr(df[x1][mask].iloc[:i], df[x2][mask].iloc[:i])
        corrList.append(m)
        print(" correlation %.5f"%m)
    ax.plot(xLog, corrList, label='cut %.1f'%cut)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
# %%
