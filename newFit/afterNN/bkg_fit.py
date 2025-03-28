# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import shapiro, kstest, norm, chi2
import json
columns=['dijet_mass', 'dijet_pt']
x1, x2 = 80, 150

# Function defined absorbing one global normalization factor
def pol2(x,  b, c):
    return 1 + b*x + c*x**2

def expo(x,  B):
    return  np.exp(-B * x)

def total_model(x, norm, B,  b, c):
    maxX, minX = x2, x1
    def indef_integral(x, B, b, c):
        term1 = c* (-x**2 * np.exp(-B*x) / B - 2*x * np.exp(-B*x) / B**2 - 2 * np.exp(-B*x) / B**3)
        term2 = b * (-x * np.exp(-B*x) / B - np.exp(-B*x) / B**2)
        term3 = -  np.exp(-B*x) / B
        integral =  (term1 + term2 + term3)
        return integral
    integral = indef_integral(maxX, B, b, c) - indef_integral(minX, B, b, c)
    return norm * pol2(x,  b, c)*expo(x,  B)/integral


# %%

isDataList = [2]
MCList = [1, 3, 4, 19, 20, 21, 22]

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/Feb21_0p0"
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
dfProcessesMC = dfProcessesMC.iloc[MCList]
dfProcessesData = dfProcessesData.iloc[isDataList]

# %%
# Data
dfsData = []
lumi_tot = 0.
for processName in dfProcessesData.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/dataframes_%s_Feb21_0p0.parquet"%processName)
    dfsData.append(df)
    lumi_tot = lumi_tot + np.load(path+"/lumi_%s_Feb21_0p0.npy"%processName)




# %%
dfsMC = []
for processName in dfProcessesMC.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_Feb21_0p0.parquet"%processName)
    dfsMC.append(df)

# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight=dfsMC[idx].weight*lumi_tot
# %%
dfsData = cut(dfsData, 'PNN', 0.6, None)
dfsMC = cut(dfsMC, 'PNN', 0.6, None)
dfsData = cut(dfsData, 'dijet_pt', 100, None)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet1_btagDeepFlavB', 0.71, None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet2_btagDeepFlavB', 0.71, None)

df = pd.concat(dfsData)
dfMC = pd.concat(dfsMC)

bins = np.linspace(x1, x2, 101)
x=(bins[1:]+bins[:-1])/2
c=np.histogram(df.dijet_mass, bins=bins)[0]

# %%

t1, t2 = 75, 115

mask = (x < t1) | (x > t2)
x_fit = x[mask]
y_tofit = c[mask]
yerr = np.sqrt(y_tofit)

# Plot
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x, c, yerr=np.sqrt(c), fmt='o', color='black', markersize=3, label="Data")
bot = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight , label=dfProcessesMC.process.iloc[idx], bottom=bot)[0]
    bot += c_


# Minuit fits
least_squares = LeastSquares(x_fit, y_tofit , yerr, total_model)
m_tot = Minuit(least_squares,  norm=c.sum()*(bins[1]-bins[0])	, B=1e-2, b=1e-1, c=1e-5) # 52 497 
#m_tot.fixed['b'] = True
#m_tot.fixed['c'] = True
m_tot.migrad()
m_tot.hesse()

p_tot = m_tot.values

# Generate fit curves
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = total_model(x_plot, *p_tot)



ax[0].plot(x_plot, y_tot, label="Fit Sidebands", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=(x < t1) | (x > t2), color='green', alpha=0.2)
ax[0].set_xlim(bins[0], bins[-1])
ax[0].set_ylim(1, max(c)*1.2)

ax[1].errorbar(x, c-total_model(x, *p_tot), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
ax[1].set_ylabel("Data - Fit")
ax[1].set_ylim(ax[1].get_ylim())
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=(x < t1) | (x > t2), color='green', alpha=0.2)

ax[1].hist(bins[:-1], bins=bins, weights = bot)[0]
#bot = np.zeros(len(bins)-1)
#for idx, dfMC in enumerate(dfsMC):
#    c_=ax[1].hist(dfMC.dijet_mass, bins=bins, weights = lumi*1000*dfMC.genWeight * dfMC.PU_SF * dfMC.sf * dfProcesses.xsection.iloc[idx] /gensumw[idx] , label=dfProcesses.process.iloc[idx], bottom=bot)[0]
#    bot += c_

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].legend()
chi2_stat = np.sum(((c[mask] - total_model(x, *p_tot)[mask])**2) / np.sqrt(c)[mask]**2)
ndof = len(y_tofit)-len(m_tot.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
ax[0].set_yscale('log')
fig.savefig("/t3home/gcelotto/newFit/afterNN/plots/bkgFitSideBands_Z.png", bbox_inches='tight')



print("Polynomial Fit Parameters:", p_tot)
print("Chi2/ndof:", m_tot.fval / ndof)
print("Cross check\nMy Chi2: %.1f\nMinuit Chi2: %.1f"%(chi2_stat, m_tot.fval))
fit_params = {name: {"value": m_tot.values[name], "error": m_tot.errors[name]} for name in m_tot.parameters}
# %%
with open("/t3home/gcelotto/newFit/afterNN/parameters/fit_parameters_bkg100_bT.json", "w") as f:
    json.dump(fit_params, f, indent=4)


#
#
#
#
# %%
import scipy.stats as stats
def poisson_likelihood(data, model):
    return np.sum(stats.poisson.logpmf(data, model)), stats.poisson.logpmf(data, model)
predictionPerBin = total_model(x, *p_tot)
L_H0_sum, L_H0_ar = poisson_likelihood(c, predictionPerBin)     # Background-only hypothesis
L_H1_sum, L_H1_ar = poisson_likelihood(c, predictionPerBin+bot)  # Signal + background hypothesis

# Compute the Likelihood Ratio Test (LRT) statistic
Lambda_obs_ar = -2 * (L_H0_ar - L_H1_ar)
Lambda_obs_sum = -2 * (L_H0_sum - L_H1_sum)
fig, ax = plt.subplots(1, 1)
ax.plot(x, Lambda_obs_ar)
ax.set_ylabel("$\Lambda$ = -2 * (L_H0 - L_H1)")
ax.set_xlabel("Dijet Mass [GeV]")



chi2_pvalue = 1 - chi2.cdf(Lambda_obs_sum, 1)
print(f"Observed LRT statistic: {Lambda_obs_sum:.3f}")
print("Pvalue under H0 %.3f"%chi2_pvalue)
# %%
import numpy as np
from scipy.stats import chi2

alpha = 0.05  # Significance level

# Generate toy datasets under H0 (background only)
num_toys = 10000
Lambda_sim_h0 = []
Lambda_sim_h1 = []
Lambda_temp = []
for _ in range(num_toys):
    toy_data = np.random.poisson(predictionPerBin)  # Generate data under H0
    L_H0_sum, _ = poisson_likelihood(toy_data, predictionPerBin)
    L_H1_sum, _ = poisson_likelihood(toy_data, predictionPerBin + bot)
    Lambda_sim_h0.append(-2 * (L_H0_sum - L_H1_sum))
    
    toy_data_h1 = np.random.poisson(predictionPerBin+bot)  # Generate data under H0
    L_H0_sum, _ = poisson_likelihood(toy_data_h1, predictionPerBin)
    L_H1_sum, _ = poisson_likelihood(toy_data_h1, predictionPerBin + bot)
    Lambda_sim_h1.append(-2 * (L_H0_sum - L_H1_sum))

#c_critical = np.percentile(Lambda_sim, 100 * (1 - alpha))
#if Lambda_obs_sum > c_critical:
#    print("Reject H0: Significant evidence for signal.")
#else:
#    print("Do not reject H0: No significant evidence for signal.")

fig, ax = plt.subplots(1, 1)
ax.hist(Lambda_sim_h0, bins=100, label='Toy H0')
ax.hist(Lambda_sim_h1, bins=100, label='Toy H1')
ax.vlines(x=Lambda_obs_sum, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label='Obs.', color='black')
ax.set_xlabel("$\Lambda$")
ax.legend()
# %%
chi2_stat = np.sum(((c-(predictionPerBin+bot))/np.sqrt(c))**2)
ndof = len(c)
chi2_pvalue = 1 - chi2.cdf(chi2_stat, ndof)
print("PValue assuming Z Boson", chi2_pvalue)
#ax[1].text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[1].transAxes, ha='right')
# %%













# Generate fit curves
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = total_model(x_plot, *p_tot)


# Plot
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x, c, yerr=np.sqrt(c), fmt='o', color='black', markersize=3, label="Data")
ax[0].plot(x_plot, y_tot, label="Fit Sidebands", color='red')
ax[0].set_ylim(ax[0].get_ylim())
ax[0].fill_between(x, 0, max(c)*1.2, where=(x < t1) | (x > t2), color='green', alpha=0.2, label='Fit Region')
ax[0].set_xlim(bins[0], bins[-1])
#ax[0].set_ylim(50, max(c)*1.2)
ax[0].hist(bins[:-1], bins=bins, weights=bot, bottom=total_model(x, *p_tot))

ax[1].errorbar(x, c-total_model(x, *p_tot), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
ax[1].hist(bins[:-1], bins=bins, weights=bot)
ax[1].set_ylabel("Data - Fit")
ax[1].set_ylim(ax[1].get_ylim())
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=(x < t1) | (x > t2), color='green', alpha=0.2)

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass")
ax[0].set_ylabel("Counts")
ax[0].legend()
chi2_stat = np.sum(((c[mask] - total_model(x, *p_tot)[mask])**2) / np.sqrt(c)[mask]**2)
ndof = len(y_tofit)-len(m_tot.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.95, y=0.95, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top')
ax[0].set_yscale('log')
fig.savefig("/t3home/gcelotto/newFit/afterNN/plots/bkgFit_components.png", bbox_inches='tight')

# %%

# *******************************************************
assert False




# Do several cuts now






# *******************************************************
pvalues_bkg = []
pvalues_z = []

cutpts = [120, 122.5, 125, 127.5, 130, 132.5, 135, 137.5, 140, 142.5, 145, 147.5, 150, 152.5, 155, 160]
for cutpt in cutpts:
    dfsData = cut(dfsData_, 'dijet_pt', cutpt, None)
    dfsMC = cut(dfsMC_, 'dijet_pt', cutpt, None)
    df = pd.concat(dfsData)
    dfMC = pd.concat(dfsMC)
    if cutpt < 121:
        bins = np.linspace(65, 150, 51)
    elif cutpt>141:
        bins = np.linspace(60, 150, 51)
    else:
        bins = np.linspace(55, 150, 51)
    x=(bins[1:]+bins[:-1])/2
    c=np.histogram(df.dijet_mass, bins=bins)[0]

    mcCounts = np.zeros(len(bins)-1)
    for idx, dfMC in enumerate(dfsMC):
        c_=np.histogram(dfMC.dijet_mass, bins=bins, weights = lumi*1000*dfMC.genWeight * dfMC.PU_SF * dfMC.sf * dfProcesses.xsection.iloc[idx] /gensumw[idx])[0]
        mcCounts += c_

    t1, t2 = 70, 115

    mask = (x < t1) | (x > t2)
    x_fit = x[mask]
    y_tofit = c[mask]
    yerr = np.sqrt(y_tofit)

    # Minuit fits
    least_squares = LeastSquares(x_fit, y_tofit - MCsubtraction*mcCounts[mask], yerr, total_model)
    m_tot = Minuit(least_squares,  A=52, B=11e-3, a=498, b=0.64, c=5.3e-3)
    m_tot.migrad()

    p_tot = m_tot.values


    x_plot = np.linspace(bins[0], bins[-1], 500)
    y_tot = total_model(x_plot, *p_tot)




    chi2_stat = np.sum(((c[mask] - total_model(x, *p_tot)[mask])**2) / np.sqrt(c)[mask]**2)
    ndof = len(y_tofit)-len(m_tot.parameters)
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    print("Cut : %d"%cutpt)
    print("Chi2/ndof Sidebands : %.1f/%d" %(chi2_stat, ndof))
    print("PValue Sidebands : %.3f" %(chi2_pvalue))




    predictionPerBin = total_model(x, *p_tot)
    L_H0_sum, L_H0_ar = poisson_likelihood(c, predictionPerBin)     # Background-only hypothesis
    L_H1_sum, L_H1_ar = poisson_likelihood(c, predictionPerBin+bot)  # Signal + background hypothesis

    Lambda_obs_sum = -2 * (L_H0_sum - L_H1_sum)
    chi2_pvalue = 1 - chi2.cdf(Lambda_obs_sum, 1)
    print("Likelihoods", L_H0_sum, L_H1_sum)
    print(f"Observed LRT statistic: {Lambda_obs_sum:.3f}")
    
    chi2_stat_bkg = np.sum(((c-predictionPerBin)/np.sqrt(c))**2)
    ndof = len(c)
    chi2_pvalue_bkg = 1 - chi2.cdf(chi2_stat_bkg, ndof)
    print("Pvalue under Bkg only %.3f"%chi2_pvalue_bkg)
    pvalues_bkg.append(chi2_pvalue)

    chi2_stat_z = np.sum(((c-predictionPerBin-bot)/np.sqrt(c))**2)
    ndof = len(c)
    chi2_pvalue_z = 1 - chi2.cdf(chi2_stat_z, ndof)
    print("PValue assuming Z Boson", chi2_pvalue_z)
    pvalues_z.append(chi2_pvalue_z)

    print("\n\n\n")

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(cutpts, pvalues_z, label='H0(Fit+Z)')
ax.plot(cutpts, pvalues_bkg, label='H0(Fit)')
ax.legend()
ax.set_xlabel("Cut in dijet pT")
ax.set_ylabel("Pvalue")
# %%
