# %%
from functions import loadMultiParquet_v2, getCommonFilters
import pandas as pd
import numpy as np
import numpy as np
from scipy.optimize import curve_fit
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
# %%
#df = loadMultiParquet_v2(paths=[37], nMCs=-1, columns=None, returnFileNumberList=False, selectFileNumberList=None, returnNumEventsTotal=False, filters=getCommonFilters(btagWP="L", cutDijet=True, ttbarCR=False))[0]
df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
#df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/df_ZJetsToQQ_600to800_Jan21_3_50p0.parquet")


# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
fig, ax_main = plt.subplots(1, 1, sharex=True, sharey=True)
left, bottom = 0.2, 0.2
width, height = 1., 1.
spacing = 0.04

bins_dR = np.linspace(0, 0.1, 60)
bins_dpT = np.linspace(-1, 1, 60)

ax_main.set_position([left, bottom, width, height])
ax_x = fig.add_axes([left, bottom + height + spacing, width, 0.2], sharex=ax_main)
ax_y = fig.add_axes([left + width + spacing, bottom, 0.2, height], sharey=ax_main)

# data
x = df.dpT_jet1_genQuark
y = df.dR_jet1_genQuark

# 2D histogram
h = ax_main.hist2d(
    x, y,
    bins=[bins_dpT, bins_dR]
)

# marginals
ax_x.hist(x, bins=bins_dpT)
ax_y.hist(y, bins=bins_dR, orientation="horizontal")

# labels
ax_main.set_xlabel(r"$\Delta p_T$ jet1-Quark1")
ax_main.set_ylabel(r"$\Delta R$ jet1-Quark1")

# cosmetics
ax_x.tick_params(labelbottom=False)
ax_y.tick_params(labelleft=False)
ax_x.set_title("Jet 1 matching to Gen Quark 1")

fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/jet1_matching_dpT_dR.png", bbox_inches="tight")
# %%



fig, ax_main = plt.subplots(1, 1, sharex=True, sharey=True)
left, bottom = 0.2, 0.2
width, height = 1., 1.
spacing = 0.04

bins_dR = np.linspace(0, 0.1, 60)
bins_dpT = np.linspace(-1, 1, 60)

ax_main.set_position([left, bottom, width, height])
ax_x = fig.add_axes([left, bottom + height + spacing, width, 0.2], sharex=ax_main)
ax_y = fig.add_axes([left + width + spacing, bottom, 0.2, height], sharey=ax_main)

# data
x = df.dpT_jet2_genQuark
y = df.dR_jet2_genQuark

# 2D histogram
h = ax_main.hist2d(
    x, y,
    bins=[bins_dpT, bins_dR]
)

# marginals
ax_x.hist(x, bins=bins_dpT)
ax_y.hist(y, bins=bins_dR, orientation="horizontal")

# labels
ax_main.set_xlabel(r"$\Delta p_T$ jet2-Quark2")
ax_main.set_ylabel(r"$\Delta R$ jet2-Quark2")

# cosmetics
ax_x.tick_params(labelbottom=False)
ax_y.tick_params(labelleft=False)

ax_x.set_title("Jet 2 matching to Gen Quark 2")

fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/jet2_matching_dpT_dR.png", bbox_inches="tight")
# %%

import numpy as np
import matplotlib.pyplot as plt

pt_thr = np.linspace(0.1, 0.5, 31)
dr_thr = np.linspace(0.1, 0.4, 31)

efficiency = np.zeros((len(pt_thr), len(dr_thr)))

for i, pt_t in enumerate(pt_thr):
    for j, dr_t in enumerate(dr_thr):

        mask = (
            (np.abs(df.dpT_jet1_genQuark) < pt_t) &
            (np.abs(df.dpT_jet2_genQuark) < pt_t) &
            (df.dR_jet1_genQuark < dr_t) &
            (df.dR_jet2_genQuark < dr_t)
        )

        efficiency[i, j] = mask.sum() / len(df)

# %%
fig, ax = plt.subplots(1, 1)

im = ax.imshow(
    efficiency,
    origin="lower",
    aspect="auto",
    extent=[dr_thr.min(), dr_thr.max(), pt_thr.min(), pt_thr.max()]
)

ax.set_xlabel(r"$\Delta R$ threshold")
ax.set_ylabel(r"$|\Delta p_T|$ threshold")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Efficiency")

plt.show()

# %%

dr_values = [ 0.2, 0.3, 0.4, 0.5, 0.8]
fig, ax = plt.subplots(1, 1)

for dr_t in dr_values:

    eff_pt = []

    for pt_t in pt_thr:
        mask = (
            (np.abs(df.dpT_jet1_genQuark) < pt_t) &
            (np.abs(df.dpT_jet2_genQuark) < pt_t) &
            (df.dR_jet1_genQuark < dr_t) &
            (df.dR_jet2_genQuark < dr_t)
        )

        eff_pt.append(mask.sum() / len(df))

    ax.plot(pt_thr, eff_pt, label=r"$\Delta R^{thr} < {%.1f}$"%(dr_t)) 
ax.set_xlabel(r"$|\Delta p_T|/p_T$ jet-Quark")
ax.set_ylabel("Efficiency of Jet pair selection")
ax.legend()
ax.grid(True)
#ax.set_xlim(0, 1)
ax.set_ylim(0.4, 1)
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/1D_scan_pair_dpt_dr.png.png", bbox_inches="tight")


# %%
fig, ax = plt.subplots(1, 1)

for dr_t in dr_values:

    eff_pt = []

    for pt_t in pt_thr:
        mask = (
            (np.abs(df.dpT_jet1_genQuark) < pt_t) &
            #(np.abs(df.dpT_jet2_genQuark) < pt_t) &
            (df.dR_jet1_genQuark < dr_t) 
            #(df.dR_jet2_genQuark < dr_t)
        )

        eff_pt.append(mask.sum() / len(df))

    ax.plot(pt_thr, eff_pt, label=r"$\Delta R^{thr} < {%.1f}$"%(dr_t)) 
ax.set_xlabel(r"$|\Delta p_T|/p_T$ jet1-Quark1")
ax.set_ylabel("Efficiency of Jet 1 selection")
ax.legend()

ax.set_ylim(0.4, 1)
ax.grid(True)

fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/1D_scan_jet1_dpt_dr.png.png", bbox_inches="tight")

# %%
plt.close('all')
#mask_mumu = (df.PNN > 0.88) & (df.PNN < 0.9275)& (df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71)
print(mask_mumu.sum()/len(df))
fig, ax = plt.subplots(1, 1)

for dr_t in dr_values:
    eff_pt = []

    for pt_t in pt_thr:
        mask = (
            (np.abs(df.dpT_jet2_genQuark) < pt_t) &
            (df.dR_jet2_genQuark < dr_t) 
        )

        eff_pt.append(mask.sum() / len(df))

    ax.plot(pt_thr, eff_pt, label=r"$\Delta R^{thr} < {%.1f}$"%(dr_t)) 
ax.set_xlabel(r"$|\Delta p_T|/p_T$ jet2-Quark2 with Muon")
ax.set_ylabel("Efficiency of Jet 2")
#ax.text(0.6, 0.4, f"PNN > 0.9275\nEfficiency: {mask_mumu.sum()/len(df):.3f}", transform=ax.transAxes)
ax.legend()

ax.set_ylim(0.35, 1)
ax.grid(True)

fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/1D_scan_jet2mu_dpt_dr.png.png", bbox_inches="tight")
# %%
fig, ax  = plt.subplots(1, 1)
bins= np.linspace(50, 200, 51)
c=ax.hist(df.dijet_mass, bins=bins, histtype='step', label='No cut', density=True)[0]
cmumu=ax.hist(df.dijet_mass[mask_mumu], bins=bins, histtype='step', label='mumu', density=True)[0]
p0 = [c.max(), 120, 20]
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

popt, _ = curve_fit(gauss, (bins[1:]+bins[:-1])/2, c,     p0=p0 )
x_fit = np.linspace(50, 200, 51)
ax.plot(
    x_fit,
    gauss(x_fit, *popt),
    linestyle='--',
    linewidth=1,
    color="C0",
    label=f'Fit: $\sigma={popt[2]:.2f}$'
)
popt, _ = curve_fit(gauss, (bins[1:]+bins[:-1])/2, cmumu,     p0=p0 )
x_fit = np.linspace(50, 200, 101)
ax.plot(
    x_fit,
    gauss(x_fit, *popt),
    linestyle='--',
    linewidth=1,
    color="C1",
    label=f'Fit: $\sigma={popt[2]:.2f}$'
)
#ax.text(0.6, 0.4, f"PNN > 0.9275\nEfficiency: {mask_mumu.sum()/len(df):.3f}", transform=ax.transAxes)
ax.legend()
# %%
# 1D scan


scan_bin = np.linspace(0.5, 0.91, 101)

sigma_arr = np.full(len(scan_bin), np.nan)


mjj_bins = np.linspace(50, 200, 51)
bin_centers = 0.5 * (mjj_bins[1:] + mjj_bins[:-1])

# reference (no cut)
c_ref = np.histogram(df.dijet_mass, bins=mjj_bins, weights=df.weight)[0]
p0_ref = [c_ref.max(), bin_centers[np.argmax(c_ref)], 20]

for i, t_ in enumerate(scan_bin):

    mask = (df.PNN > t_) & (df.PNN < 0.9275) & (df.jet1_btagDeepFlavB>0.71)& (df.jet2_btagDeepFlavB>0.71)
    cmasked = np.histogram(df.dijet_mass[mask], bins=bins)[0]

    if cmasked.sum() < 20:
        continue  # not enough stats

    nonzero = cmasked > 0

    try:
        popt, _ = curve_fit(
            gauss,
            bin_centers[nonzero],
            cmasked[nonzero],
            p0=p0_ref
        )
        sigma_arr[i] = popt[2]

    except RuntimeError:
        pass
# %%
fig, ax = plt.subplots(1, 1)

im = ax.plot(scan_bin,abs(sigma_arr))

ax.set_xlabel('NN score cut')
ax.set_ylabel('Sigma Signal')






# %%








pt1_bins = np.linspace(20, 20.001, 21)
pt2_bins = np.linspace(0.5, 0.95, 101)

sigma_mat = np.full((len(pt1_bins), len(pt2_bins)), np.nan)

bins = np.linspace(50, 200, 51)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# reference (no cut)
c_ref = np.histogram(df.dijet_mass, bins=bins)[0]
p0_ref = [c_ref.max(), bin_centers[np.argmax(c_ref)], 20]

for i, pt1_ in enumerate(pt1_bins):
    for j, pt2_ in enumerate(pt2_bins):

        mask = (df.jet1_pt > pt1_) & (df.PNN > pt2_)
        cmumu = np.histogram(df.dijet_mass[mask], bins=bins)[0]

        if cmumu.sum() < 20:
            continue  # not enough stats

        nonzero = cmumu > 0

        try:
            popt, _ = curve_fit(
                gauss,
                bin_centers[nonzero],
                cmumu[nonzero],
                p0=p0_ref
            )
            sigma_mat[i, j] = popt[2]

        except RuntimeError:
            pass
fig, ax = plt.subplots(1, 1)

im = ax.imshow(
    sigma_mat,
    origin='lower',
    extent=[pt2_bins[0], pt2_bins[-1], pt1_bins[0], pt1_bins[-1]],
    aspect='auto'
)

ax.set_xlabel('jet2 muon pT cut')
ax.set_ylabel('jet1 muon pT cut')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'Gaussian $\sigma$')
# %%
# Make sure to ignore NaNs
min_idx = np.unravel_index(np.nanargmin(sigma_mat), sigma_mat.shape)

# Get the corresponding pt1 and pt2 values
pt1_opt = pt1_bins[min_idx[0]]
pt2_opt = pt2_bins[min_idx[1]]
sigma_min = sigma_mat[min_idx]

print(f"Minimum sigma: {sigma_min}")
print(f"Optimal pt1 cut: {pt1_opt}")
print(f"Optimal pt2 cut: {pt2_opt}")

# %%
fig, ax = plt.subplots(1, 1)

for dr_t in dr_values:

    eff_pt = []

    for pt_t in pt_thr:
        mask = (
            #(np.abs(df.dpT_jet1_genQuark) < pt_t) &
            (np.abs(df.dpT_jet2_genQuark) < pt_t) &
            #(df.dR_jet1_genQuark < dr_t) 
            (df.dR_jet2_genQuark < dr_t)
        )

        eff_pt.append(mask.sum() / len(df))

    ax.plot(pt_thr, eff_pt, label=fr"$\Delta R < {dr_t}$")
ax.set_xlabel(r"$|\Delta p_T|/p_T$ jet2-Quark2")
ax.set_ylabel("Efficiency of Jet 2 selection")
ax.legend()
#ax.set_ylim(ax.get_xlim()[0], 1)
ax.set_ylim(ax.get_ylim()[0], 1)
ax.grid(True)

fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/jet_selection_efficiency/1D_scan_jet2_dpt_dr.png.png", bbox_inches="tight")
# %%




df = loadMultiParquet_v2(paths=[37], nMCs=-1, columns=None, returnFileNumberList=False, selectFileNumberList=None, returnNumEventsTotal=False, filters=getCommonFilters(btagWP="L", cutDijet=False, ttbarCR=False))[0]
# %%
print("Efficiency for dijet pt > 100 is ", (df[df.dijet_pt>100].flat_weight).sum()/(df.flat_weight.sum())*100)
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_pt, bins=np.linspace(0, 200, 301))
ax.set_xlabel("Dijet pT [GeV]")
ax.set_ylabel("Unweighted Counts")

# %%


import numpy as np
from scipy.optimize import curve_fit



fig, ax = plt.subplots(1, 1)

bins = np.linspace(0, 200, 51)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# dpt/pt bins
ranges = [
    (0.0, 0.2),
    #(0.1, 0.2),
    (0.2, 0.4),
    #(0.3, 0.4),
    (0.4, np.inf),
]

labels = [
    "0.0 < dpt/pt < 0.2",
    #"0.1 < dpt/pt < 0.2",
    "0.2 < dpt/pt < 0.4",
    #"0.3 < dpt/pt < 0.4",
    "0.4 < dpt/pt < Inf",
]
colors = [
    "C0",
    #"0.1 < dpt/pt < 0.2",
    "C1",
    #"0.3 < dpt/pt < 0.4",
    "C2",
]

bottom = np.zeros(len(bins) - 1)

xfit = np.linspace(50, 200, 1000)

# --- loop over slices ---
for (lo, hi), label, color in zip(ranges, labels, colors):

    mask = (lo < df.dpT_jet2_genQuark) & (df.dpT_jet2_genQuark < hi)
    values = df.dijet_mass[mask]

    counts, _ = np.histogram(values, bins=bins)
    print(f"{label}: {counts.max()}")

    ax.hist(
        values,
        bins=bins,
        histtype='step',
        bottom=None,
        label=label,
        color=color,
        density=False
    )

    # Gaussian fit (only where there are entries)
    nonzero = counts > 0
    p0 = [counts.max(), 120, 20]

    popt, _ = curve_fit(
        gauss,
        bin_centers[nonzero],
        counts[nonzero],
        p0=p0
    )

    ax.plot(
        xfit,
        gauss(xfit, *popt),
        linestyle='--',
        linewidth=1,
        color=color,
        label=fr'{label} fit: $\sigma={popt[2]:.2f}$'
    )

    bottom += counts


# --- All events ---
counts_all, _ = np.histogram(df.dijet_mass, bins=bins)
p0_all = [3000, 125, 19]
popt_all, _ = curve_fit(gauss, bin_centers, counts_all, p0=p0_all)

ax.hist(
    df.dijet_mass,
    bins=bins,
    histtype='step',
    color='black',
    linewidth=1.5,
    label=fr'All: $\sigma={popt_all[2]:.2f}$'
)

ax.plot(
    xfit,
    gauss(xfit, *popt_all),
    color='black',
    linestyle='--'
)

ax.set_xlabel('dijet mass')
ax.set_ylabel('Events')
ax.legend()
