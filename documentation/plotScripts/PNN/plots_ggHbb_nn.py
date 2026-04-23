# %%
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
# %%
hep.style.use("CMS")
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"
modelName = "Jan21_3_50p0"
df = pd.read_parquet(folder + modelName + "/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
df_VBF = pd.read_parquet(folder + modelName + "/df_VBFHToBB_Jan21_3_50p0.parquet")
Z_processes = ["ZJetsToQQ_100to200", 
               "ZJetsToQQ_200to400",
                "ZJetsToQQ_400to600",
                "ZJetsToQQ_600to800",
                "ZJetsToQQ_800toInf"]
df_Z=pd.DataFrame()
for Z_proc in Z_processes:
    df_Z__ = pd.read_parquet(folder + modelName + f"/df_{Z_proc}_Jan21_3_50p0.parquet")
    df_Z = pd.concat([df_Z, df_Z__], ignore_index=True)
# %%
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(1, 1)
bins_nn, bins_dijet_pt = np.linspace(0, 1, 101), np.linspace(80, 500, 101)
ax.hist2d(df.PNN, df.dijet_pt, bins=(bins_nn, bins_dijet_pt), cmap="viridis", norm=LogNorm())
ax.set_xlabel("NN")
ax.set_ylabel("dijet pt")

ax_top = inset_axes(
    ax,
    width="100%",
    height="40%",
    loc="upper center",
    bbox_to_anchor=(0, 0.85, 1, 0.25),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# right marginal
ax_right = inset_axes(
    ax,
    width="40%",
    height="100%",
    loc="center right",
    bbox_to_anchor=(0.85, 0, 0.25, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# MC marginals
ax_top.hist(df.PNN, bins=bins_nn, density=True, color='red', alpha=0.4)
ax_top.hist(df.PNN, bins=bins_nn, density=True, color='red', linewidth=3, histtype='step')



ax_right.hist(df.dijet_pt, bins=bins_dijet_pt, density=True, color='red', alpha=0.4, orientation="horizontal")
ax_right.hist(df.dijet_pt, bins=bins_dijet_pt, density=True, color='red', linewidth=3, histtype='step', orientation="horizontal")

# Data marginals
#ax_top.set_yscale('log')
#ax_right.set_xscale('log')

ax_right.set_ylim(ax.get_ylim())
ax_top.set_xlim(ax.get_xlim())
#ax_right.set_xlim(ax.get_xlim())
#ax_top.axis("off")
ax_right.set_xticks([])
ax_right.set_yticks([])
ax_top.set_xticks([])
ax_top.set_yticks([])
#ax_right.axis("off")

# %%
dfs = []
categories = {
    0:"No Cut",
    1:"NN Loose",
    7:"NN Medium",
    8:"NN Tight"}
# %%
import yaml
for cat in categories.keys():
    yaml_file=f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{cat}.yml"
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

        df_=df.copy().query(cfg["cuts_string"])
        print("Appended")
        dfs.append(df_)


dfs_VBF = []
import yaml
for cat in categories.keys():
    yaml_file=f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{cat}.yml"
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

        df_=df_VBF.copy().query(cfg["cuts_string"])
        print("Appended")
        dfs_VBF.append(df_)
bins_dijet_pt = np.linspace(80, 900, 31)


dfs_Z = []
import yaml
for cat in categories.keys():
    yaml_file=f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{cat}.yml"
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

        df_=df_Z.copy().query(cfg["cuts_string"])
        print("Appended")
        dfs_Z.append(df_)


# %%
        
VARIABLE = "jet3_pt"
BINS = np.linspace(-0.01, 500, 51)
fig, ax  = plt.subplots(1, 4, figsize=(25, 5))
fig.subplots_adjust( hspace=0.35,  wspace=0.25,  top=0.95,bottom=0.08,left=0.07,right=0.98)
for i, cat in enumerate(categories.keys()):
    print("Plotting category %d: %s"%(cat, categories[cat]))

    a = ax[i%(len(categories))]

    # --- ggF ---
    x = np.clip(dfs[i][VARIABLE], BINS[0], BINS[-1])
    w = dfs[i].weight * 41.6

    counts, _ = np.histogram(x, bins=BINS, weights=w)
    sumw2, _ = np.histogram(x, bins=BINS, weights=w**2)
    err = np.sqrt(sumw2)

    centers = 0.5 * (BINS[1:] + BINS[:-1])
    widths = np.diff(BINS)

    a.bar(centers, counts, width=widths, align="center", alpha=0.5, label="ggF")
    a.errorbar(centers, counts, yerr=err, fmt="none", capsize=2)
    print("Events with no third jet in ggF %.2f%%"%(counts[0]/sum(counts)*100))

    # --- VBF ---
    x = np.clip(dfs_VBF[i][VARIABLE], BINS[0], BINS[-1])
    w = dfs_VBF[i].weight * 41.6

    counts, _ = np.histogram(x, bins=BINS, weights=w)
    sumw2, _ = np.histogram(x, bins=BINS, weights=w**2)
    err = np.sqrt(sumw2)

    a.step(BINS[:-1], counts, where="post", linewidth=2, label="VBF", color='red')
    a.errorbar(centers, counts, yerr=err, fmt="none", capsize=2, color='red')

    print("Events with no third jet in VBF %.2f%%"%(counts[0]/sum(counts)*100))

    # --- ZJets ---
    # Scaled by factor
    r_factor = 30
    x = np.clip(dfs_Z[i][VARIABLE], BINS[0], BINS[-1])
    w = dfs_Z[i].weight * 41.6

    counts, _ = np.histogram(x, bins=BINS, weights=w/r_factor)
    sumw2, _ = np.histogram(x, bins=BINS, weights=w**2)
    err = np.sqrt(sumw2)/r_factor

    a.step(BINS[:-1], counts, where="post", linewidth=2, label=f"ZJets / {r_factor}", color='green')
    a.errorbar(centers, counts, yerr=err, fmt="none", capsize=2, color='green')
    print("Events with no third jet in ZJets %.2f%%"%(counts[0]/sum(counts)*100))

    # --- styling ---
    a.set_xlabel(VARIABLE)
    a.set_ylabel("Events")
    
    a.set_title(categories[cat], fontsize=14)
    a.set_xlim(BINS[0], BINS[-1])
    a.legend()
# %%
category_plot = 0
dfs[category_plot][['jet1_eta', 'jet2_eta', 'jet1_phi', 'jet2_phi']]
dfs[category_plot]['dPhi'] = np.abs(dfs[category_plot]['jet1_phi'] - dfs[category_plot]['jet2_phi'])
dfs[category_plot]['dEta'] = np.abs(dfs[category_plot]['jet1_eta'] - dfs[category_plot]['jet2_eta'])
dfs[category_plot]['dR'] = np.sqrt(dfs[category_plot]['dPhi']**2 + dfs[category_plot]['dEta']**2)
dfs[category_plot][['jet1_eta', 'jet2_eta', 'jet1_phi', 'jet2_phi', 'dR']]

fig, ax = plt.subplots(1, 1)
ax.hist(dfs[category_plot].dR, bins=np.linspace(0, 3, 101), weights=dfs[category_plot].weight)
ax.set_xlabel("dR")
# %%


from functions import loadMultiParquet_v2, getCommonFilters
dfs = loadMultiParquet_v2(paths=[37],  nMCs=-1,columns=None, filters=getCommonFilters(btagWP="M", cutDijet=False, ttbarCR=0))
# %%
df = pd.concat(dfs, ignore_index=True)
df =df[(df.NN > 0.825) & (df.NN < 0.85)]
# %%
from matplotlib.patches import Circle

fig, ax = plt.subplots(1, 1)

ev = 3
pt1 = df['jet1_pt'].iloc[ev]
eta1 = df['jet1_eta'].iloc[ev]
phi1 = df['jet1_phi'].iloc[ev]

pt2 = df['jet2_pt'].iloc[ev]
eta2 = df['jet2_eta'].iloc[ev]
phi2 = df['jet2_phi'].iloc[ev]


higgs_gen_pt = df['higgs_gen_pt'].iloc[ev]
higgs_gen_eta = df['higgs_gen_eta'].iloc[ev]
higgs_gen_phi = df['higgs_gen_phi'].iloc[ev]
# compute dijet phi and eta


eta3 = dfs[0]['jet3_eta'].iloc[ev]
phi3 = dfs[0]['jet3_phi'].iloc[ev]
pt3 = dfs[0]['jet3_pt'].iloc[ev]


pt_dijet = df['dijet_pt'].iloc[ev]
eta_dijet = df['dijet_eta'].iloc[ev]
phi_dijet = df['dijet_phi'].iloc[ev]



ax.scatter(eta1, phi1, s=50, label='Jet 1')
ax.scatter(eta2, phi2, s=50, label='Jet 2')
ax.scatter(eta3, phi3, s=50, label='Jet 3')
ax.scatter(eta_dijet, phi_dijet, s=50, label='DiJet')
ax.scatter(higgs_gen_eta, higgs_gen_phi, s=50, label='Higgs_gen', marker='*')

#ax.scatter(eta4, phi4, s=50, label='Jet 4')

R = 0.4

circle1 = Circle((eta1, phi1), R, fill=False)
circle2 = Circle((eta2, phi2), R, fill=False)
circle3 = Circle((eta3, phi3), R, fill=False)
circle4 = Circle((eta_dijet, phi_dijet), R, fill=False)
#circle5 = Circle((eta4, phi4), R, fill=False)

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
#ax.add_patch(circle5)
ax.set_xlim(-5, 5)
ax.set_ylim(-3.14, 3.14)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\phi$')
ax.legend()

plt.show()

# %%
df = pd.concat(dfs, ignore_index=True)
df8 =df[(df.NN > 0.925) ]
df7 =df[(df.NN > 0.85) & (df.NN < 0.925) ]
df1 =df[(df.NN > 0.825) & (df.NN < 0.85) ]
fig, ax = plt.subplots(1, 1)
bins=np.linspace(-1, 1, 81)
ax.hist((df1.higgs_gen_pt - df1.dijet_pt)/df1.higgs_gen_pt, bins=bins, label="pT Resolution Hbb1",density=True, histtype='step')
ax.hist((df7.higgs_gen_pt - df7.dijet_pt)/df7.higgs_gen_pt, bins=bins, label="pT Resolution Hbb7",density=True, histtype='step')
ax.hist((df8.higgs_gen_pt - df8.dijet_pt)/df8.higgs_gen_pt, bins=bins, label="pT Resolution Hbb8",density=True, histtype='step')
ax.legend()

fig, ax = plt.subplots(1, 1)
bins=np.linspace(-1, 1, 81)
ax.hist((df1.higgs_gen_mass - df1.dijet_mass)/df1.higgs_gen_mass, bins=bins, label="mass Resolution Hbb1",density=True, histtype='step')
ax.hist((df7.higgs_gen_mass - df7.dijet_mass)/df7.higgs_gen_mass, bins=bins, label="mass Resolution Hbb7",density=True, histtype='step')
ax.hist((df8.higgs_gen_mass - df8.dijet_mass)/df8.higgs_gen_mass, bins=bins, label="mass Resolution Hbb8",density=True, histtype='step')
ax.legend()

fig, ax = plt.subplots(1, 1)
bins=np.linspace(-1, 1, 81)
ax.hist((df1.higgs_gen_phi - df1.dijet_phi)/df1.higgs_gen_phi, bins=bins, label="phi Resolution Hbb1",density=True, histtype='step')
ax.hist((df7.higgs_gen_phi - df7.dijet_phi)/df7.higgs_gen_phi, bins=bins, label="phi Resolution Hbb7",density=True, histtype='step')
ax.hist((df8.higgs_gen_phi - df8.dijet_phi)/df8.higgs_gen_phi, bins=bins, label="phi Resolution Hbb8",density=True, histtype='step')
ax.legend()

fig, ax = plt.subplots(1, 1)
bins=np.linspace(-1, 1, 81)
ax.hist((df1.higgs_gen_eta - df1.dijet_eta)/df1.higgs_gen_eta, bins=bins, label="eta Resolution Hbb1",density=True, histtype='step')
ax.hist((df7.higgs_gen_eta - df7.dijet_eta)/df7.higgs_gen_eta, bins=bins, label="eta Resolution Hbb7",density=True, histtype='step')
ax.hist((df8.higgs_gen_eta - df8.dijet_eta)/df8.higgs_gen_eta, bins=bins, label="eta Resolution Hbb8",density=True, histtype='step')
ax.legend()


# %%
import numpy as np
import matplotlib.pyplot as plt

def compute_response_vs_eta(df, eta_col, pt_col, genpt_col, bins):
    eta = df[eta_col].values
    R = (df[pt_col] / df[genpt_col]).values

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    response = []
    response_err = []

    for i in range(len(bins) - 1):
        mask = (eta >= bins[i]) & (eta < bins[i+1])
        values = R[mask]

        if len(values) > 5:
            med = np.median(values)
            err = np.std(values) / np.sqrt(len(values))
        else:
            med = np.nan
            err = np.nan

        response.append(med)
        response_err.append(err)

    return bin_centers, np.array(response), np.array(response_err)


# define eta bins
bins = np.linspace(30, 300, 21)

# compute for each dataset
x1, y1, e1 = compute_response_vs_eta(df1, 'jet1_pt_uncor', 'jet1_pt', 'jet1_genQuark_pt', bins)
x7, y7, e7 = compute_response_vs_eta(pd.concat([df7, df8]), 'jet1_pt_uncor', 'jet1_pt', 'jet1_genQuark_pt', bins)
#x8, y8, e8 = compute_response_vs_eta(df8, 'jet1_eta', 'jet1_pt', 'jet1_genQuark_pt', bins)


# plot
fig, ax = plt.subplots(1, 1)

ax.errorbar(x1+0.5, y1, yerr=e1, fmt='o', label='df1')
ax.errorbar(x7-0.5, y7, yerr=e7, fmt='s', label='df7')
#ax.errorbar(x8, y8, yerr=e8, fmt='^', label='df8')

ax.set_xlabel('jet1 pt_uncor')
ax.set_ylabel('Response (pT_reco / pT_gen)')
ax.set_ylim(0, 2)
ax.legend()

plt.show()
# %%
import numpy as np
from iminuit import Minuit

# remove NaNs
mask = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(e1)
x = x1[mask]
y = y1[mask]
e = e1[mask]

def model(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def chi2(a, b, c, d):
    return np.sum(((y - model(x, a, b, c, d)) / e)**2)

m = Minuit(chi2, a=1.0, b=0.0, c=0.0, d=0.0)
m.errordef = Minuit.LEAST_SQUARES
m.migrad()
m.hesse()

print(m.values)
# %%
fig, ax = plt.subplots(1, 1)

ax.errorbar(x1, y1, yerr=e1, fmt='o', label='df1')

xx = np.linspace(30, 300, 200)
yy = model(xx, *m.values)

ax.plot(xx, yy, label='pol3 fit')

ax.set_xlabel('jet1 pt uncor')
ax.set_ylabel('Response')
ax.legend()

plt.show()
# %%
def correction(variable):
    correction = model(variable, *m.values)
    correction[variable>=300]=1
    correction[variable<=30]=1

    return correction
df1['jet1_pt_corr'] = df1['jet1_pt'] / correction(df1['jet1_pt_uncor'].values)
df1['jet2_pt_corr'] = df1['jet2_pt'] / correction(df1['jet2_pt_uncor'].values)
def dijet_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    E1 = pt1 * np.cosh(eta1)
    E2 = pt2 * np.cosh(eta2)

    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    E  = E1 + E2

    return np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0))

df1['mjj_raw'] = dijet_mass(
    df1['jet1_pt'], df1['jet1_eta'], df1['jet1_phi'],
    df1['jet2_pt'], df1['jet2_eta'], df1['jet2_phi']
)

df1['mjj_corr'] = dijet_mass(
    df1['jet1_pt_corr'], df1['jet1_eta'], df1['jet1_phi'],
    df1['jet2_pt_corr'], df1['jet2_eta'], df1['jet2_phi']
)

def compute_fwhm(values, bins):
    hist, edges = np.histogram(values, bins=bins, density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    max_val = np.max(hist)
    half_max = max_val / 2.0

    # find where histogram crosses half max
    above = hist >= half_max

    if not np.any(above):
        return np.nan

    indices = np.where(above)[0]

    left_idx = indices[0]
    right_idx = indices[-1]

    fwhm = centers[right_idx] - centers[left_idx]

    return fwhm
fwhm_raw = compute_fwhm(df1['mjj_raw'], bins)
fwhm_corr = compute_fwhm(df1['mjj_corr'], bins)

print("FWHM raw     :", fwhm_raw)
print("FWHM corr    :", fwhm_corr)
print("Improvement  :", (fwhm_raw - fwhm_corr) / fwhm_raw)
bins = np.linspace(50, 200, 41)
fig, ax = plt.subplots(1, 1)

ax.hist(df1['mjj_raw'], bins=bins, histtype='step', density=True, label=f'raw (FWHM={fwhm_raw:.1f})')
ax.hist(df1['mjj_corr'], bins=bins, histtype='step', density=True, label=f'corr (FWHM={fwhm_corr:.1f})')

ax.set_xlabel('mjj')
ax.legend()

plt.show()
# %%
# compute for each dataset
x1, y1, e1 = compute_response_vs_eta(df1, 'jet2_pt_uncor', 'jet2_pt', 'jet2_genQuark_pt', bins)
x7, y7, e7 = compute_response_vs_eta(pd.concat([df7, df8]), 'jet2_pt_uncor', 'jet2_pt', 'jet2_genQuark_pt', bins)
#x8, y8, e8 = compute_response_vs_eta(df8, 'jet2_eta', 'jet2_pt', 'jet2_genQuark_pt', bins)


# plot
fig, ax = plt.subplots(1, 1)

ax.errorbar(x1, y1, yerr=e1, fmt='o', label='df1')
ax.errorbar(x7, y7, yerr=e7, fmt='s', label='df7')
#ax.errorbar(x8, y8, yerr=e8, fmt='^', label='df8')

ax.set_xlabel('jet2 pt_uncor')
ax.set_ylabel('Response (pT_reco / pT_gen)')
ax.set_ylim(0, 2)
ax.legend()

plt.show()

#ax.hist(df.dijet_pt, bins=bins, label="Reco Hbb (dijet)")
# %%
