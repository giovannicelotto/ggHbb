# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2, getCommonFilters
import mplhep as hep
hep.style.use("CMS")
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import shapiro, kstest, norm, chi2
import json
from scipy.special import erf
columns=['dijet_mass', 'dijet_pt']

def pol2N(x,  b, c):
    # norm inside the continuum bkg function
    return 1 + b*x + c*x**2

def expo(x,  B):
    return  np.exp(-B * x)
def continuum_background(x, norm, B,  b, c):
    maxX, minX = x2, x1
    def indef_integral(x, B, b, c):
        term1 = c* (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
        term2 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term3 = -  np.exp(-B*x) / B
        integral =  (term1 + term2 + term3)
        return integral
    integral = indef_integral(maxX, B, b, c) - indef_integral(minX, B, b, c)
    return norm * pol2N(x,  b, c)*expo(x,  B)/integral

def pol1N(x, p1):
    result = 1 + p1 * x
    integral = (x2-x1) + p1*(x2**2-x1**2)/2
    return  result/integral


def dscb(x, mean, sigma, alphaL, nL, alphaR, nR):
    t = (x - mean) / sigma
    A = (nL / abs(alphaL)) ** nL * np.exp(-0.5 * alphaL ** 2)
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)

    left = A * (nL / abs(alphaL) - abs(alphaL) - t) ** (-nL)  # Left tail
    right = B * (nR / abs(alphaR) - abs(alphaR) + t) ** (-nR)  # Right tail

    central = np.exp(-0.5 * t ** 2)


    integralLeft =  -A*sigma/(-nL+1)*(
        (nL / abs(alphaL) - abs(alphaL) - (-alphaL)) ** (-nL+1) - 
        (nL / abs(alphaL) - abs(alphaL) - (x1-mean)/sigma) ** (-nL+1))
    integralRight =  B*sigma/(-nR+1)*(
        (nR / abs(alphaR) - abs(alphaR) + (x2-mean)/sigma) ** (-nR+1) - 
        (nR / abs(alphaR) - abs(alphaR) + (alphaR)) ** (-nR+1))
    integralCentral = 1/2*(erf(alphaR/np.sqrt(2))  -    erf(-alphaL/np.sqrt(2)))*np.sqrt(np.pi)*(np.sqrt(2)*sigma)
    totalIntegral = integralLeft + integralRight + integralCentral
    return np.where(t < -alphaL, left, np.where(t > alphaR, right, central))/totalIntegral

def gaussianN(x, mean, sigmaG):
    integralGauss = 1/2*(erf((x2-mean)/(sigmaG*np.sqrt(2)))  -    erf((x1-mean)/(sigmaG*np.sqrt(2))))
    return  1/(np.sqrt(2*np.pi)*sigmaG)* np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)/integralGauss

def zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, fraction_gaussian, sigmaG, p1):
    return normSig*(fraction_dscb*dscb(x, mean, sigma, alphaL, nL, alphaR, nR) + fraction_gaussian*gaussianN(x, mean, sigmaG)+ (1-fraction_gaussian-fraction_dscb)*pol1N(x, p1))

def continuum_plus_Z(x, normBkg, B, b, c, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, fraction_gaussian, sigmaG, p1):
    return continuum_background(x, normBkg, B,  b, c) + zPeak(x, normSig, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, fraction_gaussian, sigmaG, p1)



x1, x2 = 65, 150



# %%

dfsData_, lumi = loadMultiParquet_Data_new(dataTaking=[0, 1, 2, 3], nReals=[-1, -1, -1, -1], columns=columns+['jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'nJets_20'], filters=getCommonFilters(btagTight=True))
# %%
MCList = [1, 3, 4, 19, 20,21,22]
dfsMC_, gensumw = loadMultiParquet_v2(paths=MCList, nMCs=-1, columns=columns+['genWeight', 'sf', 'PU_SF', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'nJets_20'], returnNumEventsTotal=True, filters=getCommonFilters(btagTight=True))
MCsubtraction = False
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
# %%
MCList_Higgs = [0,36]
dfsMC_Higgs_, gensumw_Higgs = loadMultiParquet_v2(paths=MCList_Higgs, nMCs=-1, columns=columns+['genWeight', 'sf', 'PU_SF', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'nJets_20'], returnNumEventsTotal=True, filters=getCommonFilters(btagTight=True))
dfProcesses_Higgs = getDfProcesses_v2()[0].iloc[MCList_Higgs]
# %%
dfsData = cut(dfsData_, 'dijet_pt', 100, None)
dfsMC = cut(dfsMC_, 'dijet_pt', 100, None)
dfsMC_Higgs = cut(dfsMC_Higgs_, 'dijet_pt', 100, None)

dfsData = cut(dfsData, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC_Higgs = cut(dfsMC_Higgs, 'jet1_btagDeepFlavB', 0.71, None)

dfsData = cut(dfsData, 'jet2_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet2_btagDeepFlavB', 0.71, None)
dfsMC_Higgs = cut(dfsMC_Higgs, 'jet2_btagDeepFlavB', 0.71, None)

#dfsData = cut(dfsData, 'nJets_20', None, 5.1)
#dfsMC = cut(dfsMC, 'nJets_20', None, 5.1)
#dfsMC_Higgs = cut(dfsMC_Higgs, 'nJets_20', None, 5.1)

# %%
df = pd.concat(dfsData)


bins = np.linspace(x1, x2, 51)
x=(bins[1:]+bins[:-1])/2
c=np.histogram(df.dijet_mass, bins=bins)[0]

# %%
# Blind a region for the fit
t1, t2 = 115, 145

mask = (x < t1) | (x > t2)
x_fit = x[mask]
y_tofit = c[mask]
yerr = np.sqrt(y_tofit)

# Plot Data
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x, c, yerr=np.sqrt(c), fmt='o', color='black', markersize=3, label="Data")
# Plot Z
bot = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = lumi*1000*dfMC.genWeight * dfMC.PU_SF * dfMC.sf * dfProcesses.xsection.iloc[idx] /gensumw[idx] , label=dfProcesses.process.iloc[idx], bottom=bot)[0]
    bot += c_
cumulativeMC = bot.copy()
cHiggs = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_Higgs):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = lumi*1000*dfMC.genWeight * dfMC.PU_SF * dfMC.sf * dfProcesses_Higgs.xsection.iloc[idx] /gensumw_Higgs[idx] , label=dfProcesses_Higgs.process.iloc[idx], bottom=cumulativeMC)[0]
    cHiggs += c_
    cumulativeMC +=c_

# Open parameters of bkg
with open("/t3home/gcelotto/newFit/fit_parameters_bkg100_bT.json", "r") as f:
    fit_parameters_bkg = json.load(f)

with open("/t3home/gcelotto/newFit/fit_parameters_Z100_bT.json", "r") as f:
    fit_parameters_zPeak = json.load(f)


# Minuit fits

least_squares = LeastSquares(x_fit, y_tofit - MCsubtraction*bot[mask], yerr, continuum_plus_Z)
m_tot = Minuit(least_squares,
               normBkg=c.sum()*(bins[1]-bins[0])-bot.sum()*(bins[1]-bins[0]),
               B=fit_parameters_bkg['B']['value'],
               b=fit_parameters_bkg['b']['value'],
               c=fit_parameters_bkg['c']['value'],
               normSig=bot.sum()*(bins[1]-bins[0]),
               fraction_dscb=fit_parameters_zPeak["fraction_dscb"]['value'],
               mean=fit_parameters_zPeak['mean']['value'],
               sigma=fit_parameters_zPeak['sigma']['value'],
               alphaL=fit_parameters_zPeak['alphaL']['value'],
               nL=fit_parameters_zPeak['nL']['value'],
               alphaR=fit_parameters_zPeak['alphaR']['value'],
               nR=fit_parameters_zPeak['nR']['value'],
               fraction_gaussian=fit_parameters_zPeak['fraction_gaussian']['value'],
               sigmaG=fit_parameters_zPeak['sigmaG']['value'],
               p1=fit_parameters_zPeak['p1']['value']
               )

m_tot.fixed['sigma'] = True
m_tot.fixed['mean'] = True
m_tot.fixed['sigmaG'] = True
#m_tot.fixed['normSig'] = True
#m_tot.fixed['B'] = True
#m_tot.fixed['b'] = True
#m_tot.fixed['c'] = True
m_tot.fixed['alphaR'] = True
m_tot.fixed['nR'] = True
m_tot.fixed['alphaL'] = True
m_tot.fixed["fraction_gaussian"]=True
m_tot.fixed["fraction_dscb"]=True
m_tot.fixed['nL'] = True
m_tot.fixed['p1'] = True



m_tot.migrad()
m_tot.hesse()

p_tot = m_tot.values

# Generate fit curves
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = continuum_plus_Z(x_plot, *p_tot)



ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=(x < t1) | (x > t2), color='green', alpha=0.2)
ax[0].set_xlim(bins[0], bins[-1])
ax[0].set_ylim(1, max(c)*1.2)

ax[1].errorbar(x, c-continuum_background(x, *p_tot[["normBkg", "B", "b", "c"]]), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
ax[1].plot(x, zPeak(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'fraction_gaussian', 'sigmaG', 'p1']]), color='red', linewidth=2)
ax[1].set_ylabel("Data - Background")
ax[1].set_ylim(ax[1].get_ylim())
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=(x < t1) | (x > t2), color='green', alpha=0.2)

ax[1].hist(bins[:-1], bins=bins, weights = bot)[0]
ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=bot)[0]

hep.cms.label(lumi="%.2f" %lumi, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].legend(bbox_to_anchor=(1, 1))

chi2_stat = np.sum(((c[mask] - continuum_plus_Z(x, *p_tot)[mask])**2) / np.sqrt(c)[mask]**2)
ndof = len(y_tofit)-len(m_tot.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
ax[0].set_yscale('log')

fig.savefig("/t3home/gcelotto/newFit/bkgPlusZPeak.png", bbox_inches='tight')

# %%
















# Plot 2
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(14, 14), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = continuum_plus_Z(x_plot, *p_tot)
ax[0].errorbar(x, c, yerr=np.sqrt(c), fmt='o', color='black', markersize=3, label="Data")
ax[0].plot(x_plot, y_tot, label="Fit Sidebands", color='red')
ax[0].set_ylim(ax[0].get_ylim())
ax[0].fill_between(x, 0, max(c)*1.2, where=(x < t1) | (x > t2), color='green', alpha=0.2, label='Fit Region')
ax[0].set_xlim(bins[0], bins[-1])


ax[1].errorbar(x, c-continuum_background(x, *p_tot[["normBkg", "B", "b", "c"]]), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
ax[1].plot(x, zPeak(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'fraction_gaussian', 'sigmaG', 'p1']]), color='red', linewidth=2)
ax[1].hist(bins[:-1], bins=bins, weights=bot, label='Z')
ax[1].set_ylabel("Data - Background")
#ax[1].set_ylim(ax[1].get_ylim())
cHiggs = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_Higgs):
    c_=ax[1].hist(dfMC.dijet_mass, bins=bins, weights = lumi*1000*dfMC.genWeight * dfMC.PU_SF * dfMC.sf * dfProcesses_Higgs.xsection.iloc[idx] /gensumw_Higgs[idx] , label=dfProcesses_Higgs.process.iloc[idx], bottom=cHiggs)[0]
    cHiggs += c_
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*1.5)
ax[1].legend(ncols=3)
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=(x < t1) | (x > t2), color='green', alpha=0.2)

hep.cms.label(lumi="%.2f" %lumi, ax=ax[0])
ax[1].set_xlabel("Dijet Mass")
ax[0].set_ylabel("Counts")
ax[0].legend(fontsize=24)
chi2_stat = np.sum(((c[mask] - continuum_plus_Z(x, *p_tot)[mask])**2) / np.sqrt(c)[mask]**2)
ndof = len(y_tofit)-len(m_tot.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.95, y=0.95, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
ax[0].set_yscale('log')
ax[0].tick_params(labelsize=24)
ax[1].tick_params(labelsize=24)
# End Plot 2
# %%
