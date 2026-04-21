# %%
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
# %%
y_tt = np.random.normal(loc=0.45, scale=0.15, size=8001)
y_ggH = np.random.normal(loc=0.55, scale=0.15, size=8001)
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-0.15, 1.25, 50)
ax.hist(y_tt, bins=bins, density=True, histtype='step', label='ttbar', color='blue')
ax.hist(y_ggH, bins=bins, density=True, histtype='step', label='ggH', color='red')
ax.legend()
ax.set_xlabel("Feature of interest")
ax.set_ylabel("Density")
ax.set_xlim(bins[0], bins[-1])
# %%
fig, ax = plt.subplots(1, 1)
x = (bins[:-1] + bins[1:]) / 2
ax.hist(y_tt, bins=bins, density=True, histtype='step', label='ttbar', color='blue')
ax.hist(y_ggH, bins=bins, density=True, histtype='step', label='ggH', color='red')
# Copmute cumulative density functions
cdf_tt = np.cumsum(np.histogram(y_tt, bins=bins, density=True)[0])
cdf_ggH = np.cumsum(np.histogram(y_ggH, bins=bins, density=True)[0])
# Right axis for CDF
ax2 = ax.twinx()

ax2.plot(x, cdf_tt/cdf_tt[-1], linestyle='--', label='ttbar CDF', color='blue')
ax2.plot(x, cdf_ggH/cdf_ggH[-1], linestyle='--', label='ggH CDF', color='red')
ax2.set_ylim(0, 1)
ax2.set_ylabel("Cumulative Density Function")

#ax.plot(np.linspace(0, 1, 49), 3*cdf_tt/cdf_tt[-1], label='ttbar CDF', color='blue')
#ax.plot(np.linspace(0, 1, 49), 3*cdf_ggH/cdf_ggH[-1], label='ggH CDF', color='red')
# add on the right y-axis the cumulative density function values

ax.legend()
ax.set_xlabel("Feature of interest")
ax.set_ylabel("Density")
# %%
fig, ax = plt.subplots(1, 1)
x = (bins[:-1] + bins[1:]) / 2
# Right axis for CDF
ax2 = ax.twinx()
ax2.plot(x, cdf_tt/cdf_tt[-1], linestyle='--', label='ttbar CDF', color='blue')
ax2.plot(x, cdf_ggH/cdf_ggH[-1], linestyle='--', label='ggH CDF', color='red')
ax2.set_ylim(0, 1)
ax2.set_ylabel("Cumulative Density Function")

# add on the right y-axis the cumulative density function values



# Quantile matching: ggH -> ttbar
sorted_tt = np.sort(y_tt)
sorted_ggH = np.sort(y_ggH)

# Compute ranks of ggH values
ranks = np.searchsorted(sorted_tt, y_tt, side='right') / len(sorted_tt)

# Map to ttbar quantiles
y_tt_matched = np.quantile(sorted_ggH, ranks)
# Quantile matching: ggH -> ttbar
sorted_ggH = np.sort(y_ggH)
sorted_tt = np.sort(y_tt)

# Compute ranks of ggH values
ranks = np.searchsorted(sorted_tt, y_tt, side='right') / len(sorted_tt)

# Map to ttbar quantiles
y_ggH_matched = np.quantile(sorted_ggH, ranks)
# BEFORE (already in your code)
ax.hist(y_ggH, bins=bins, density=True, histtype='step', label='ggH', color='red')
ax.hist(y_tt, bins=bins, density=True, histtype='step', label='ttbar (before QM)', color='blue')

# AFTER (quantile matched)
ax.hist(y_tt_matched, bins=bins, density=True,
        histtype='step', linestyle='--',
        label='ttbar (after QM)', color='green', linewidth=2)

ax.set_xlabel("Feature of interest")
ax.set_ylabel("Density")
ax.legend()
# %%
