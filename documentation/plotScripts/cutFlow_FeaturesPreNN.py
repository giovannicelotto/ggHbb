# %%
from functions import cut, cut_advanced, loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
nn= False
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
def format_number(value):
    if value >= 1e6:
        return f'{value / 1e6:.1f}M'
    elif value >= 1e3:
        return f'{value / 1e3:.1f}K'
    else:
        return str(round(value))
    

def format_number_percentage(value):
    if value >99:
        return "%.1f"%value
    elif value >10:
        return "%.2f"%value
    elif 0.1<value <10:
        return "%.2f"%value
    else:
        return "%.3f"%value
isHiggsList = [37, 36]
#dfProcessesMC = getDfProcesses_v2()[0]
#xsections = dfProcessesMC.iloc[isHiggsList].xsection

isZList = [1,19,20,21,22]
#isQCDList = [23,24,25,26,27,29,29,30,31,32,33,34]
#xsectionsZ = dfProcessesMC.iloc[isZList].xsection

featuresForTraining = ['dijet_pt', 'dijet_eta', 'jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'muon_pt','muon_dxySig', 'jet1_pt', 'jet2_pt', 'nJets', 'jet1_pt_uncor', 'jet2_pt_uncor']
# %%
dfsData_, lumi, fileNumberListData = loadMultiParquet_Data_new(dataTaking=[0,6,12,17], nReals=[-1, 100,100,100], columns=featuresForTraining+['jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], filters=[('dijet_pt', '>=', 100)], returnFileNumberList=True)
dfsMC_, sumw, fileNumberListMC = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+['xsection', 'btag_central', 'flat_weight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=[('dijet_pt', '>=', 100)], returnFileNumberList=True)
#dfsMC_Check, sumw_Check, fileNumberListMC_Check= loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+['xsection', 'flat_weight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=getCommonFilters(btagWP="L"), returnFileNumberList=True)
dfsZ, sumwZ, fileNumberListZ = loadMultiParquet_v2(paths=isZList, nMCs=-1, columns=featuresForTraining+['xsection', 'flat_weight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=[('dijet_pt', '>=', 100)], returnFileNumberList=True)
#dfs_qcd, sumw_qcd, fileNumberListZ = loadMultiParquet_v2(paths=isQCDList, nMCs=-1, columns=featuresForTraining+['xsection', 'flat_weight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=[('dijet_pt', '>=', 100)], returnFileNumberList=True)

# %%
#dfsMC_[0].btag_central = np.where(dfsMC_[0].btag_central.values<-9000, 1, dfsMC_[0].btag_central.values)
#dfsMC_[1].btag_central = np.where(dfsMC_[1].btag_central.values<-9000, 1, dfsMC_[1].btag_central.values)

data = pd.concat(dfsData_)
for idx, df in enumerate(dfsZ):
    df['sumw'] = sumwZ[idx]
dfZ = pd.concat(dfsZ)
# %%
cuts = [
    (   
        "Baseline (pt_jj>100)",
        lambda df: (df.jet1_mass > 0) & (df.jet2_mass > 0) & 
                    (df.jet1_pt_uncor>0) & (df.jet2_pt_uncor>0) & 
                    (abs(df.jet1_eta) < 2.5) & (abs(df.jet2_eta) < 2.5) & 
                    (df.dijet_pt > 100),
        "Baseline (pt$_{jj}$>100)",
    ),
    (
        "Muon cuts",
        lambda df: (df.muon_pt >= 9)
                   & (abs(df.muon_dxySig) >= 6)
                   & (abs(df.muon_eta) <= 1.5),
        "Muon cuts",
    ),
    (
        "B-tagged (L)",
        lambda df: (df.jet1_btagDeepFlavB >= 0.049)
                   & (df.jet2_btagDeepFlavB >= 0.049),
        "B-tagged (L)",
    ),
    (
        "40 < m_jj < 300",
        lambda df: (df.dijet_mass > 40) & (df.dijet_mass < 300),
        "40 < $m_{jj}$ < 300",
    ),
]

def compute_cutflow(df, base_mask, weight=None):
    mask = base_mask.copy()
    yields = []
    effs = []

    if weight is None:
        n0 = np.sum(mask)
    else:
        n0 = weight[mask].sum()

    for name, cut,label in cuts:
        mask &= cut(df)

        if weight is None:
            ni = np.sum(mask)
        else:
            ni = weight[mask].sum()

        yields.append(ni)
        effs.append(ni / n0 if n0 > 0 else 0.0)

    return yields, effs, df[mask]

maskData = (data.dijet_mass.values>0) & (data.dijet_mass.values<np.inf)
maskHiggs = (dfsMC_[0].dijet_mass.values>0) & (dfsMC_[0].dijet_mass.values<np.inf)
maskVBF = (dfsMC_[1].dijet_mass.values>0) & (dfsMC_[1].dijet_mass.values<np.inf)
maskZ = (dfZ.dijet_mass.values>0) & (dfZ.dijet_mass.values<np.inf)

# DATA
data_yields, data_eff, data_cut = compute_cutflow(
    data,
    maskData
)

# ggH
w_higgs = (
    dfsMC_[0].flat_weight 
) / sumw[0] * dfsMC_[0].xsection * lumi * 1000

higgs_yields, higgs_eff, higgs_cut = compute_cutflow(
    dfsMC_[0],
    maskHiggs,
    weight=w_higgs
)

# VBF
w_vbf = (
    dfsMC_[1].flat_weight 
) / sumw[1] * dfsMC_[1].xsection * lumi * 1000

vbf_yields, vbf_eff, vbf_cut = compute_cutflow(
    dfsMC_[1],
    maskVBF,
    weight=w_vbf
)

# ZJets
w_Z = (
    dfZ.flat_weight 
) / dfZ.sumw * dfZ.xsection * lumi * 1000

Z_yields, Z_eff, z_cut = compute_cutflow(
    dfZ,
    maskZ,
    weight=w_Z
)
# %%

for i, (name, _,label) in enumerate(cuts):
    print(
        f"{name:25s} | "
        f"{data_yields[i]:7.0f} ({data_eff[i]:.3f}) | "
        f"{higgs_yields[i]:7.1f} ({higgs_eff[i]:.3f}) | "
        f"{vbf_yields[i]:7.1f} ({vbf_eff[i]:.3f}) | "
        f"{Z_yields[i]:7.1f} ({Z_eff[i]:.3f})"
    )


# %%
    
def format_events(x):
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 10:
        return f"{x:.2e}"   # integer-like
    return f"{x:.2f}"       # small signals

x = np.arange(len(cuts) + 1)

data_eff_step  = np.r_[data_eff,  data_eff[-1]]
higgs_eff_step = np.r_[higgs_eff, higgs_eff[-1]]
vbf_eff_step   = np.r_[vbf_eff,   vbf_eff[-1]]
Z_eff_step     = np.r_[Z_eff,     Z_eff[-1]]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

Lfull = 41.6  # fb^-1

Nfull = {
    "Data":  data_yields[-1]  * Lfull / lumi,
    "ggH":   higgs_yields[-1] * Lfull / lumi,
    "VBF":   vbf_yields[-1]   * Lfull / lumi,
    "Zbb":   Z_yields[-1]     * Lfull / lumi,
}



ax.step(x, data_eff_step,  where="post",
        label=f"Data ({format_events(Nfull['Data'])})")

ax.step(x, higgs_eff_step, where="post",
        label=f"ggH ({format_events(Nfull['ggH'])})")
ax.step(x, vbf_eff_step,  where="post",
        label=f"VBF ({format_events(Nfull['VBF'])})")

ax.step(x, Z_eff_step, where="post",
        label=f"Zbb ({format_events(Nfull['Zbb'])})")


ax.set_xlim(0, len(cuts))
ax.set_xticks(np.arange(len(cuts)) + 0.5)
ax.set_xticklabels([c[2] for c in cuts], rotation=30, ha="right")

ax.set_ylabel("Cumulative efficiency")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True)
hep.cms.label()
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/cutflow_efficiency_preNN.png", bbox_inches="tight")
#ax.grid(True)

# %%


import matplotlib.pyplot as plt

def plot_hist(ax, data, mc, var_name, xlabel, bins, range_, weights=None, logy=False, firstColumn=True):
    
    """
    Plot a single histogram comparison on the given axis.
    
    ax       : matplotlib axis
    data     : numpy array for data
    mc       : numpy array for MC
    var_name : str, variable name for title
    xlabel   : str, x-axis label
    bins     : int, number of bins
    range_   : tuple, min/max for histogram
    weights  : numpy array for MC weights (optional)
    logy     : bool, use logarithmic y-axis
    """
    hist_opts = dict(histtype='step', density=True, linewidth=1.5)
    
    ax.hist(data, bins=bins, range=range_, label='Data', color='blue', **hist_opts)
    ax.hist(mc, bins=bins, range=range_, label='ggH', color='red', weights=weights, **hist_opts)
    
    ax.set_xlabel(xlabel, fontsize=18)
    if firstColumn==1:
        ax.set_ylabel('Events [a.u.]', fontsize=18)
    else:
        pass
    #ax.set_title(var_name)
    ax.legend(fontsize=13)
    if logy:
        ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)



# Define variables to plot
plot_vars = [
    ('dijet_mass', 'Dijet Mass [GeV]', 30, (30, 320), False),
    ('dijet_pt',   'Dijet $p_T$ [GeV]', 30, (80, 300), False),
    
    ('dijet_eta',   'Dijet η [GeV]', 30, (-3, 3), False),

    ('jet1_eta',   'Jet1 η', 30, (-2.6, 2.6), False),
    ('jet2_eta',   'Jet2 η', 30, (-2.6, 2.6), False),
    ('jet1_pt_uncor',   'Jet1 $p_T$ (no b-reg)', 30, (18, 300), False),
    ('jet2_pt_uncor',   'Jet2 $p_T$ (no b-reg)', 30, (18, 300), False),

    ('muon_pt',    'Muon $p_T$ [GeV]', 30, (8, 31), False),
    ('muon_dxySig',    'Muon dxy/dxyErr ', 30, (-15, 15), False),
    ('muon_eta',    'Muon η ', 30, (-2.1, 2.1), False),
    ('jet1_btagDeepFlavB', 'Jet1 btag', 30, (0,1), True),
    ('jet2_btagDeepFlavB', 'Jet2 btag', 30, (0,1), True)
]

# Create figure
fig, axes = plt.subplots(4, 3, figsize=(10, 10), tight_layout=True, sharey=False)
#fig.subplots_adjust(wspace=0.05, hspace=0.05)

for ax, (var, xlabel, bins, range_, logy) in zip(axes.flat, plot_vars):
    firstColumn = ax in axes[:, 0]
    plot_hist(ax,
              data=getattr(data_cut, var).values,
              mc=getattr(higgs_cut, var).values,
              var_name=var,
              xlabel=xlabel,
              bins=bins,
              range_=range_,
              weights=getattr(higgs_cut, 'flat_weight').values,
              logy=logy,
              firstColumn=firstColumn)
fig.align_ylabels(axes[:, 0])
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/preNN_variable_distributions.png", bbox_inches="tight")


# %%
