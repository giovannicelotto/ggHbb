# %%
from functions import *
import pandas as pd
import numpy as np

# %%
dfs, lumi = loadMultiParquet_Data_new(dataTaking=[17], nReals=-1, columns=None, training=True, filters=getCommonFilters(btagTight=True))
# %%
dfs[0]=dfs[0][dfs[0].dijet_pt>100]
# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
m = (dfs[0].dijet_pt>100)  & ((dfs[0].dimuonZZ_mass>1) & (dfs[0].dimuonZZ_mass<100) | (dfs[0].dieleZZ_mass>1) & (dfs[0].dieleZZ_mass<100))
fig ,ax = plt.subplots(1, 1)
bins_jj = np.linspace(50, 200, 51)
ax.hist(dfs[0].dijet_mass[m], bins=bins_jj)

# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit
from scipy.stats import chi2

hep.style.use("CMS")

# Selection mask
m = (
    (dfs[0].dijet_pt > 100) &
    (((dfs[0].dimuonZZ_mass > 0) & (dfs[0].dimuonZZ_mass < 100)) |
     ((dfs[0].dieleZZ_mass > 0) & (dfs[0].dieleZZ_mass < 100)))
)

# Histogram settings
bin_centers = 0.5 * (bins_jj[1:] + bins_jj[:-1])
hist_vals, _ = np.histogram(dfs[0].dijet_mass[m], bins=bins_jj)

# Fit region
t0, t1 = 50, 75
t2, t3 = 95, 200
fit_mask = ((bin_centers >= t0) & (bin_centers <= t1)) | ((bin_centers >= t2) & (bin_centers <= t3))

# Model: exponential * polynomial of degree 2
def exp_poly2(x, a, b, c, d):
    return np.exp(a * x) * (b * x**2 + c * x + d)

# Prepare data for fit
x_fit = bin_centers[fit_mask]
y_fit = hist_vals[fit_mask]
sigma = np.sqrt(y_fit)
sigma[sigma == 0] = 1  # Prevent division by zero

# Initial guess
p0 = [-0.01, 1e-3, 1, 1]

# Perform the fit
popt, pcov = curve_fit(exp_poly2, x_fit, y_fit, sigma=sigma, p0=p0, absolute_sigma=True)
fit_vals = exp_poly2(bin_centers, *popt)

# Chi2 and p-value
f_fit = exp_poly2(x_fit, *popt)
chi2_val = np.sum(((y_fit - f_fit) / sigma) ** 2)
ndof = len(y_fit) - len(popt)
p_val = chi2.sf(chi2_val, ndof)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

# Highlight fit regions
ax1.axvspan(t0, t1, color='orange', alpha=0.3, label="Fit Region")
ax1.axvspan(t2, t3, color='orange', alpha=0.3)

# Histogram
c = ax1.hist(dfs[0].dijet_mass[m], bins=bins_jj, histtype="step", label="Data")[0]
ax1.plot(bin_centers, fit_vals, label="Exp × Poly2 Fit", color="red")

# Annotate chi2
ax1.text(0.95, 0.45, f"$\\chi^2$ = {chi2_val:.1f}\nndof = {ndof}\np = {p_val:.3f}",
         transform=ax1.transAxes, fontsize=10, verticalalignment='top', ha='right')

ax1.set_ylabel("Events")
ax1.legend()

# Residuals
residuals = hist_vals - fit_vals
ax2.axhline(0, color='gray', linestyle='--')
ax2.errorbar(bin_centers, residuals, np.sqrt(c), marker='o', linestyle='none', color='black', markersize=3)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.set_ylabel("Residuals")
ax1.set_ylabel("Counts per %.1f"%(bins_jj[1]-bins_jj[0]))
hep.cms.label(ax=ax1, lumi=np.round(lumi, 2))
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(1, 1)
#ax.hist(dfs[0].dimuonZZ_mass, bins=np.linspace(40, 110, 101))
ax.hist(dfs[0].dieleZZ_mass, bins=np.linspace(40, 110, 101))
# %%
