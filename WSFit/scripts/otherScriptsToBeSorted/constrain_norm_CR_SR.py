# Need to run non Interactively!
# %%
import ROOT
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import sys
import yaml
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from array import array
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-c', '--config', type=str, default="2", help='Config File')
sys.path.append("/t3home/gcelotto/ggHbb/WSFit/framework_scripts")
from getDfsFromConfig import getDfsFromConfig

# %%
crs_ratios, cr2s = [], []
ratios = []
cr1s, cr2s, srs = [] ,[] , []
lower_NNs, upper_NNs = [], []
err_ratios = []
dfs = []
categories  =[6,7,8,9]
for i in categories:
    dfMC_Z, dfMC_H, df, nbins, x1, x2, lower_NN, upper_NN = getDfsFromConfig(idx=i, return_nn=True)
    lower_NNs.append(lower_NN)
    upper_NNs.append(upper_NN)
    dfs.append(df)
    print("[INFO] Limits on RooRealVar")
    print(x1, x2, nbins)

    data_CR1 = (df.dijet_mass >=70) & (df.dijet_mass <100)
    data_CR2 = (df.dijet_mass >=150) & (df.dijet_mass <300)
    data_SR = (df.dijet_mass >=100) & (df.dijet_mass <150)
    Z_CR1 = (dfMC_Z.dijet_mass >=70) & (dfMC_Z.dijet_mass <100)
    Z_CR2 = (dfMC_Z.dijet_mass >=150) & (dfMC_Z.dijet_mass <300)
    Z_SR = (dfMC_Z.dijet_mass >=100) & (dfMC_Z.dijet_mass <150)



    # Data yields
    Ndata_CR1 = len(df[data_CR1])
    Ndata_CR2 = len(df[data_CR2])
    Ndata_SR  = len(df[data_SR])

    # MC yields (weighted)
    Nmc_CR1 = dfMC_Z[Z_CR1].weight.sum()
    Nmc_CR2 = dfMC_Z[Z_CR2].weight.sum()
    Nmc_SR  = dfMC_Z[Z_SR].weight.sum()

    # MC uncertainties (sum of squared weights)
    sigma_mc_CR1 = np.sqrt((dfMC_Z[Z_CR1].weight**2).sum())
    sigma_mc_CR2 = np.sqrt((dfMC_Z[Z_CR2].weight**2).sum())
    sigma_mc_SR  = np.sqrt((dfMC_Z[Z_SR].weight**2).sum())

    # Data uncertainties (Poisson)
    sigma_data_CR1 = np.sqrt(Ndata_CR1)
    sigma_data_CR2 = np.sqrt(Ndata_CR2)
    sigma_data_SR  = np.sqrt(Ndata_SR)

    # Subtracted yields
    num   = Ndata_SR  - Nmc_SR
    denom = (Ndata_CR1 - Nmc_CR1) + (Ndata_CR2 - Nmc_CR2)
    ratio = num / denom
    srs.append(Ndata_SR  - Nmc_SR)
    cr1s.append(Ndata_CR1 - Nmc_CR1)
    cr2s.append(Ndata_CR2 - Nmc_CR2)
    # Combined uncertainties for numerator and denominator
    sigma_num = np.sqrt(sigma_data_SR**2 + sigma_mc_SR**2)
    sigma_denom = np.sqrt(sigma_data_CR1**2 + sigma_mc_CR1**2 +
                          sigma_data_CR2**2 + sigma_mc_CR2**2)

    # Propagate uncertainty on the ratio
    sigma_ratio = ratio * np.sqrt((sigma_num/num)**2 + (sigma_denom/denom)**2)
    ratios.append(ratio)
    crs_ratios.append((Ndata_CR1 - Nmc_CR1) / (Ndata_CR2 - Nmc_CR2))
    err_ratios.append(sigma_ratio)

    print(f"Ratio = {ratio:.4f} ± {sigma_ratio:.4f}")

# %%
# Control plot of pNN score
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
PNN_edges = np.sort(np.concatenate((np.linspace(0, 1, 51), np.array(lower_NNs+[1]))))
for i in range(0,len(categories)):
    ax.hist(dfs[i].PNN, density=False, label=str(i), histtype='stepfilled', bins=PNN_edges, alpha=0.4)
    ax.hist(dfs[i].PNN, density=False, histtype='step', bins=PNN_edges, edgecolor='black')
ax.set_xlabel("NN score")
ax.legend()
# %%
fig, ax = plt.subplots(1,1)
for i in range(0, len(categories)):
    if i ==0:
        continue
    ax.errorbar(i, ratios[i], err_ratios[i], color='blue')
ax.set_ylabel("SR/CR")
ax.set_xlabel("Category")

fig, ax = plt.subplots(1,1)
for i in range(0, len(categories)):
    if i ==0:
        continue
    ax.errorbar(cr1s[i], ratios[i], err_ratios[i], color='blue')
ax.set_ylabel("SR/CR")
ax.set_xlabel("CR1")


fig, ax = plt.subplots(1,1)
for i in range(0, len(categories)):
    if i ==0:
        continue
    ax.errorbar(cr2s[i], ratios[i], color='blue')
ax.set_ylabel("SR/CR")
ax.set_xlabel("CR2")



fig, (ax_main, ax_res1, ax_res2) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios':[3,1,1]}, sharex=True)

# --- Scatter and error bars ---
for i in range(len(cr2s)):
    if i == 0:
        continue
    ax_main.errorbar(cr2s[i], cr1s[i], 
                     xerr=np.sqrt(cr2s[i])*2, 
                     yerr=np.sqrt(cr1s[i])*2, 
                     color='blue', marker='o', linestyle='none')
# Add labels
for (i, x, y) in zip(categories, cr2s, cr1s):
    ax_main.text(x, y, str(i+1), fontsize=12, ha='right', va='bottom')

ax_main.set_ylabel("CR1")
ax_main.set_xlabel("CR2")

# --- Linear fits ---
# With all points
fit_all = np.polyfit(cr2s, cr1s, 1, w=1/(2*np.sqrt(cr1s)))
fit_all_fn = np.poly1d(fit_all)

# Without i=6
cr2s=np.array(cr2s)
cr1s=np.array(cr1s)
mask = np.arange(len(cr2s)) != 5
fit_excl = np.polyfit(cr2s[mask], cr1s[mask], 1, w=1/(2*np.sqrt(cr1s[mask])))
fit_excl_fn = np.poly1d(fit_excl)

# Plot fits
xfit = np.linspace(min(cr2s), max(cr2s), 100)
ax_main.plot(xfit, fit_all_fn(xfit), 'r-', label='Fit with i=6')
ax_main.plot(xfit, fit_excl_fn(xfit), 'g--', label='Fit without i=6')

ax_main.legend()

# --- Residuals ---
resid_all = cr1s - fit_all_fn(cr2s)
resid_excl = cr1s[mask] - fit_excl_fn(cr2s[mask])

# Residual plot 1 (with i=6)
ax_res1.axhline(0, color='black', lw=1)
ax_res1.plot(cr2s, resid_all, 'ro', label='Residuals (with i=6)')
ax_res1.set_ylabel('Residuals 1')

# Residual plot 2 (without i=6)
ax_res2.axhline(0, color='black', lw=1)
ax_res2.plot(cr2s[mask], resid_excl, 'go', label='Residuals (without i=6)')
ax_res2.set_ylabel('Residuals 2')
ax_res2.set_xlabel('CR2')

plt.tight_layout()
plt.show()


    #ax.errorbar(i, crs_ratios[i-1], color='red')
# %%








import matplotlib.pyplot as plt
import numpy as np

# Define bins
bins = np.linspace(70, 300, 500)

# Create figure with two subplots (shared x-axis)
fig, (ax_main) = plt.subplots(
    1, 1, figsize=(10, 10)
)

# === Main Plot ===

for i in range(len(categories)):
    ax_main.hist(
        dfs[i].dijet_mass,
        bins=bins,
        density=True,
        histtype='step',
        label=str(lower_NNs[i])+ " < PNN <"+ str(upper_NNs[i])
    )

ax_main.set_ylabel("Normalized entries")
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# === Ratio Plot (only between first two) ===
# Compute histogram values
h1, _ = np.histogram(dfs[0].dijet_mass, bins=bins, density=False)
h2, _ = np.histogram(dfs[1].dijet_mass, bins=bins, density=False)

# Compute ratio safely (avoid divide by zero)
ratio = np.divide(h1.astype(float), h2.astype(float), out=np.zeros_like(h1, dtype=float), where=(h2 != 0))
bin_centers = 0.5 * (bins[1:] + bins[:-1])


blind1, blind2 = 110, 145
ax_main.set_ylabel("Ratio")
ax_main.set_xlabel("Dijet Mass [GeV]")
plt.tight_layout()
plt.show()















#%%
import matplotlib.pyplot as plt
import numpy as np

# Example data setup
# dfs = [df1, df2, ...]  # your DataFrames
# categories = ['cat1', 'cat2', ...]

# Define bins
bins = np.linspace(70, 300, 500)

# Create figure with two subplots (shared x-axis)
fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10)
)

# === Main Plot ===

for i in range(len(categories)):
    ax_main.hist(
        dfs[i].dijet_mass,
        bins=bins,
        density=True,
        histtype='step',
        label=str(lower_NNs[i])+ " < PNN <"+ str(upper_NNs[i])
    )

ax_main.set_ylabel("Normalized entries")
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# === Ratio Plot (only between first two) ===
# Compute histogram values
h1, _ = np.histogram(dfs[0].dijet_mass, bins=bins, density=False)
h2, _ = np.histogram(dfs[1].dijet_mass, bins=bins, density=False)

# Compute ratio safely (avoid divide by zero)
ratio = np.divide(h1.astype(float), h2.astype(float), out=np.zeros_like(h1, dtype=float), where=(h2 != 0))
bin_centers = 0.5 * (bins[1:] + bins[:-1])


blind1, blind2 = 110, 145
ax_ratio.errorbar(bin_centers[(bin_centers<blind1) | (bin_centers>blind2)], ratio[(bin_centers<blind1) | (bin_centers>blind2)], (np.sqrt(h1.astype(float))/h2.astype(float))[(bin_centers<blind1) | (bin_centers>blind2)] , linestyle='none')
ax_ratio.set_ylabel("Ratio")
ax_ratio.set_xlabel("Dijet Mass [GeV]")
ax_ratio.grid(True, alpha=0.3)
#ax_ratio.set_ylim(0, 2)  # adjust as needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Polynomial function (4th degree) ---
def pol4(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4
ratio_err = np.where(h2 != 0, np.sqrt(h1) / h2, 0)
mask = (bin_centers < blind1) | (bin_centers > blind2)
popt, pcov = curve_fit(
    pol4,
    bin_centers[mask],
    ratio[mask],
    sigma=ratio_err[mask],
    absolute_sigma=True
)
x_fit = np.linspace(bins[0], bins[-1], 500)
y_fit = pol4(x_fit, *popt)
ax_ratio.plot(x_fit, y_fit, 'r-', label='Pol4 fit')
ax_ratio.errorbar(bin_centers, ratio, (np.sqrt(h1.astype(float))/h2.astype(float)) , linestyle='none', alpha=0.1)
plt.tight_layout()
plt.show()





# %%
fig, ax = plt.subplots(1, 1)
for i in range(len(categories)):
    ax.errorbar((lower_NNs[i]+upper_NNs[i])/2, ratios[i], err_ratios[i], xerr = (upper_NNs[i]-lower_NNs[i])/2, color='blue')
ax.legend()
ax.set_label("NN score")
ax.set_ylabel("SR/(CR1+CR2)")
ax.set_ylim(np.min(ratios)*0.95, np.max(ratios)*1.05)
# %%
fig, ax = plt.subplots(1, 1)
cr1s, cr2s, srs = np.array(cr1s), np.array(cr2s), np.array(srs)
ax.plot(cr1s, srs+cr1s+cr2s, marker='o', linestyle='none', label='Total vs CR1')
for i, (x, y) in enumerate(zip(cr1s, srs + cr1s + cr2s)):
    ax.text(x, y, str(i+1), fontsize=24, ha='right', va='bottom')

ax.errorbar(cr2s, srs+cr1s+cr2s, xerr=np.sqrt(cr2s), yerr=np.sqrt(srs+cr1s+cr2s), marker='o', linestyle='none', label='Total vs CR2')
ax.errorbar(cr1s+cr2s, srs+cr1s+cr2s, xerr=np.sqrt(cr1s+cr2s), yerr=np.sqrt(srs+cr1s+cr2s), marker='o', linestyle='none', label='Total vs CR1+CR2')
ax.set_xlabel("CR yields (Data - Z)")
ax.set_ylabel("Total Yields (Data - Z)")
ax.legend()

# %%






from iminuit import Minuit
y1 = np.array(srs + cr1s + cr2s )
y2 = np.array(srs + cr1s + cr2s )
y3 = np.array(srs + cr1s + cr2s )


def chi2(a, b, c, d, x, y, yerr):
    return np.sum(((y - pol3(x, a, b, c, d)) / yerr) ** 2)
def pol1(x, m, q):
    return m * x + q
def pol2(x, a, b, c):
    return a * x**2 + b * x + c
def chi2_pol1(m, q, x, y, yerr):
    return np.sum(((y - pol1(x, m, q)) / yerr) ** 2)
def chi2_pol2(a, b, c, x, y, yerr):
    return np.sum(((y - pol2(x, a, b, c)) / yerr) ** 2)


def pol3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c*x +d

# Chi2 function for iminuit


# Data sets
datasets = [
    #(cr1s, y1, np.sqrt(y1)*10, "CR1", "tab:blue"),
    (cr2s,y2, np.sqrt(cr2s), np.sqrt(y2), "CR2", "tab:orange"),
#    ((cr1s + cr2s)/(cr1s + cr2s), y3/(cr1s + cr2s), np.sqrt(y3)*4/(cr1s + cr2s), "CR1+CR2", "tab:green")
]
fig, (ax_fit, ax_res_cr1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
#fig, (ax_fit, ax_res_cr1, ax_res_cr2, ax_res_crs) = plt.subplots(4, 1, figsize=(8, 8), sharex=True,
#                                     gridspec_kw={'height_ratios': [3, 1, 1, 1]})

for i, (x, y, xerr, yerr, label, color) in enumerate(datasets):
    # --- Perform fit with iminuit ---
    def chi2_local(a,b,c):
        return chi2_pol2(a, b, c, x, y, yerr)

    # --- Perform fit ---
    m = Minuit(chi2_local, a=0.1, b=0.1, c=0.1)
    #m = Minuit(chi2, a=0.1, b=1, c=1, x=x, y=y, yerr=yerr)
    #m.errordef = 1
    m.migrad()

    a, b, c = m.values["a"], m.values["b"], m.values["c"]
    #m,q = m.values["m"], m.values["q"]
    chi2_val = chi2_local(a, b, c)
    ndof = len(x) - 2
    chi2_ndof = chi2_val / ndof

    # --- Plot data + fit ---
    xfit = np.linspace(min(x), max(x), 200)
    ax_fit.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=color, label=f"{label} ($χ^2$/ndof={chi2_ndof:.2f})")
    ax_fit.plot(xfit, pol2(xfit, a, b, c), color=color, lw=2)

    # --- Residuals ---
    residuals = (y - pol2(x,a, b, c)) / yerr
    if i==0:
        ax_res_cr1.axhline(0, color='black', lw=1)
        ax_res_cr1.errorbar(x, residuals,xerr=xerr,  yerr=np.ones_like(residuals), fmt='o', color=color)
        #ax_res_cr1.errorbar(x[:4], residuals[:4], yerr=np.ones_like(residuals)[:4], marker='*', markersize=16, color=color, linestyle='none')
    elif i==1:
        pass
        #ax_res_cr2.axhline(0, color='black', lw=1)
        #ax_res_cr2.errorbar(x, residuals, yerr=np.ones_like(residuals), fmt='o', color=color)
        #ax_res_cr2.errorbar(x[:4], residuals[:4], yerr=np.ones_like(residuals)[:4], marker='*', markersize=16, color=color, linestyle='none')
    elif i==2:
        pass
        #ax_res_crs.axhline(0, color='black', lw=1)
        #ax_res_crs.errorbar(x, residuals, yerr=np.ones_like(residuals), fmt='o', color=color)
        #ax_res_crs.errorbar(x[:4], residuals[:4], yerr=np.ones_like(residuals)[:4], marker='*', markersize=16, color=color, linestyle='none')

    


# --- Labels and layout ---
ax_fit.set_ylabel("(CRs+SR)/CR2 (Data - Z 70-200 GeV) Err x 6", fontsize=10)
ax_res_cr1.set_xlabel("CR1/CR2 yields (Data - Z)")
ax_fit.legend()
plt.tight_layout()
plt.show()
# %%
