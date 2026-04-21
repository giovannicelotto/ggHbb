# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2, loadMultiParquet_Data_new
import mplhep as hep
hep.style.use("CMS")  # Use the CMS style for plots
# %%
QCD_list = list(range(23,35))
# %%
filters= getCommonFilters(btagWP="M", cutDijet=False)
filters[0] = filters[0] + [('dijet_pt', '>=', 60), ('dijet_pt', '<', 120)]
filters[1] = filters[1] + [('dijet_pt', '>=', 60), ('dijet_pt', '<', 120)]
dfs, genW = loadMultiParquet_v2(paths=QCD_list, nMCs=-1, columns=None, returnNumEventsTotal=True, filters=filters)
# %%
dfsData, lumi = loadMultiParquet_Data_new(dataTaking=[0], columns=None, filters=filters, nReals=6)
dfData = pd.concat(dfsData, ignore_index=True) 
# %%
dfProcesses = getDfProcesses_v2()[0]
# %%
for idx, df in enumerate(dfs):
    dfs[idx]['weight'] = df.flat_weight /genW[idx] * df.xsection *lumi*1000

# %%
bins = np.linspace(0, 300, 60)  # adjust range as needed
masses = []
weights = []
labels = []
topology = []
for idx, df in enumerate(dfs):
    print(idx, dfProcesses['process'][QCD_list[idx]])
    masses.append(df['dijet_mass'].values)
    weights.append(df['weight'].values)
    topology.append(df['QCD_topology'].values)
    labels.append(dfProcesses['process'][QCD_list[idx]])


# plot
fig, ax = plt.subplots(1, 1)

ax.hist(
    masses,
    bins=bins,
    weights=weights,
    stacked=True,
    histtype='stepfilled',
    label=labels,
    alpha=0.7
)
c=np.histogram(dfData['dijet_mass'], bins=bins)[0]
ax.errorbar((bins[1:]+bins[:-1])/2,y=c, yerr=np.sqrt(c),marker='o', color='black', label='Data', linestyle='none')
#ax.hist(masses, bins=bins, histtype='step', color='black', label='Data')
ax.set_xlabel("Dijet mass [GeV]")
ax.set_ylabel("Events")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/topologies/qcd_mc_data/inclusive.png", bbox_inches='tight')
# %%








bins = np.linspace(0, 300, 60)  # adjust range as needed
masses_asym = []
weights_asym = []
labels_asym = []
topology_asym = []
inv_mass_gen = []


for idx, df in enumerate(dfs):
    print(idx, dfProcesses['process'][list(range(23,35))[idx]])
    masses_asym.append(df['dijet_mass'].values[abs(df.dijet_pT_asymmetry)>0.45])
    weights_asym.append(df['weight'].values[abs(df.dijet_pT_asymmetry)>0.45])
    labels_asym.append(dfProcesses['process'][list(range(23,35))[idx]])
    topology_asym.append(df['QCD_topology'].values[abs(df.dijet_pT_asymmetry)>0.45])
    inv_mass_gen.append(df['inv_mass_MEfinalstate'].values[abs(df.dijet_pT_asymmetry)>0.45])


dfMC = pd.concat(dfs, ignore_index=True)
# plot
fig, ax = plt.subplots(1, 1)

ax.hist(
    masses_asym,
    bins=bins,
    weights=weights_asym,
    stacked=True,
    histtype='stepfilled',
    label=labels_asym,
    alpha=0.7
)
c=np.histogram(dfData.dijet_mass[abs(dfData.dijet_pT_asymmetry)>0.45], bins=bins)[0]
ax.errorbar((bins[1:]+bins[:-1])/2,y=c, yerr=np.sqrt(c),marker='o', color='black', label='Data', linestyle='none')
#ax.hist(masses, bins=bins, histtype='step', color='black', label='Data')
ax.set_xlabel("Dijet mass [GeV]")
ax.set_ylabel("Events")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/topologies/qcd_mc_data/asym.png", bbox_inches='tight')




# %%

bins = np.linspace(0, 300, 60)  # adjust range as needed
masses_asym = []
weights_asym = []
labels_asym = []
topology_asym = []
inv_mass_gen = []


for idx, df in enumerate(dfs):
    print(idx, dfProcesses['process'][list(range(23,35))[idx]])
    masses_asym.append(df['dijet_mass'].values[abs(df.dijet_pT_asymmetry)<0.45])
    weights_asym.append(df['weight'].values[abs(df.dijet_pT_asymmetry)<0.45])
    labels_asym.append(dfProcesses['process'][list(range(23,35))[idx]])
    topology_asym.append(df['QCD_topology'].values[abs(df.dijet_pT_asymmetry)<0.45])
    inv_mass_gen.append(df['inv_mass_MEfinalstate'].values[abs(df.dijet_pT_asymmetry)<0.45])


dfMC = pd.concat(dfs, ignore_index=True)
# plot
fig, ax = plt.subplots(1, 1)

ax.hist(
    masses_asym,
    bins=bins,
    weights=weights_asym,
    stacked=True,
    histtype='stepfilled',
    label=labels_asym,
    alpha=0.7
)
c=np.histogram(dfData.dijet_mass[abs(dfData.dijet_pT_asymmetry)<0.45], bins=bins)[0]
ax.errorbar((bins[1:]+bins[:-1])/2,y=c, yerr=np.sqrt(c),marker='o', color='black', label='Data', linestyle='none')
#ax.hist(masses, bins=bins, histtype='step', color='black', label='Data')
ax.set_xlabel("Dijet mass [GeV]")
ax.set_ylabel("Events")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/topologies/qcd_mc_data/sym.png", bbox_inches='tight')
# %%
# plot
fig, ax = plt.subplots(1, 1)

ax.hist(
    inv_mass_gen,
    bins=bins,
    weights=weights_asym,
    stacked=True,
    histtype='stepfilled',
    label=labels_asym,
    alpha=0.7
)
#c=np.histogram(dfData.dijet_mass[abs(dfData.dijet_pT_asymmetry)>0.45], bins=bins)[0]
#ax.errorbar((bins[1:]+bins[:-1])/2,y=c, yerr=np.sqrt(c),marker='o', color='black', label='Data', linestyle='none')
#ax.hist(masses, bins=bins, histtype='step', color='black', label='Data')
ax.set_xlabel("Inv mass ME final state [GeV]")
ax.set_ylabel("Events")
ax.set_title("")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%

# %%
EVENT = 8
bins = np.linspace(0, 300, 60)
maskData_asym = abs(dfData.dijet_pT_asymmetry)>0.45
fig, ax  = plt.subplots(1, 1)
ax.hist(dfData.jet1_pt[maskData_asym], bins=bins, histtype='step', color='black', label='Data asym>45%')
ax.hist(dfData.jet1_pt[~maskData_asym], bins=bins, histtype='step', color='red', label='Data asym<45%')
#ax.scatter(dfData.jet1_pt[maskData_asym].iloc[EVENT], y=500, color='black', label='Data asym>45%', alpha=0.5)
#ax.scatter(dfData.jet1_pt[~maskData_asym].iloc[0], y=500, color='red', label='Data asym>45%', alpha=0.5)
ax.set_xlabel("jet1 pT [GeV]")
ax.legend()


fig, ax  = plt.subplots(1, 1)
ax.hist(dfData.jet2_pt[maskData_asym], bins=bins, histtype='step', color='black', label='Data asym>45%')
ax.hist(dfData.jet2_pt[~maskData_asym], bins=bins, histtype='step', color='red', label='Data asym<45%')
#ax.scatter(dfData.jet2_pt[maskData_asym].iloc[EVENT], y=500, color='black', label='Data asym>45%', alpha=0.5)
#ax.scatter(dfData.jet2_pt[~maskData_asym].iloc[0], y=500, color='red', label='Data asym>45%', alpha=0.5)
ax.set_xlabel("jet2 pT [GeV]")
ax.legend()

fig, ax  = plt.subplots(1, 1)
bins=np.linspace(0, 3.14, 31)
ax.hist(dfData.dijet_dPhi[maskData_asym], bins=bins, histtype='step', color='black', label='Data bump')
ax.hist(dfData.dijet_dPhi[~maskData_asym], bins=bins, histtype='step', color='red', label='Data no bump')
#ax.scatter(dfData.dijet_dPhi[maskData_asym].iloc[EVENT], y=500, color='black', label='Data asym>45%', alpha=0.5)
#ax.scatter(dfData.dijet_dPhi[~maskData_asym].iloc[0], y=500, color='red', label='Data asym>45%', alpha=0.5)
ax.set_xlabel("dijet dphi")
ax.legend()


fig, ax  = plt.subplots(1, 1)
bins=np.linspace(0, 3, 31)
ax.hist(dfData.dijet_dEta[maskData_asym], bins=bins, histtype='step', color='black', label='Data bump')
ax.hist(dfData.dijet_dEta[~maskData_asym], bins=bins, histtype='step', color='red', label='Data no bump')
#ax.scatter(dfData.dijet_dEta[maskData_asym].iloc[EVENT], y=500, color='black', label='Data asym>45%', alpha=0.5)
#ax.scatter(dfData.dijet_dEta[~maskData_asym].iloc[0], y=500, color='red', label='Data asym>45%', alpha=0.5)
ax.set_xlabel("dijet dEta ")
ax.legend()
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
bins = np.linspace(0, 300, 60)
maskData_asym = abs(dfData.dijet_pT_asymmetry) > 0.45

# 2D histogram for asymmetry > 45%
fig, ax = plt.subplots(1, 1)
h = ax.hist2d(dfData.jet1_pt[maskData_asym], dfData.jet2_pt[maskData_asym], 
               bins=(bins, bins), cmap='Blues', norm=LogNorm())
ax.set_xlabel("jet1 pT [GeV]")
ax.set_ylabel("jet2 pT [GeV]")
fig.colorbar(h[3], ax=ax, label='Counts')
ax.set_title("jet1 vs jet2 (asym > 45%)")

# 2D histogram for asymmetry < 45%
fig, ax = plt.subplots(1, 1)
h = ax.hist2d(dfData.jet1_pt[~maskData_asym], dfData.jet2_pt[~maskData_asym], 
               bins=(bins, bins), cmap='Reds', norm=LogNorm())
ax.set_xlabel("jet1 pT [GeV]")
ax.set_ylabel("jet2 pT [GeV]")
fig.colorbar(h[3], ax=ax, label='Counts')
ax.set_title("jet1 vs jet2 (asym < 45%)")
#%%

bins = np.linspace(0, 300, 60)
maskData_asym = abs(dfData.dijet_pT_asymmetry)>0.45
fig, ax  = plt.subplots(1, 1)
ax.hist(dfData.dijet_mass[(maskData_asym)].values, bins=bins, histtype='step', color='black', label='Data asym>45%', density=True)
#ax.hist(dfData.dijet_mass[(maskData_asym) & (dfData.jet1_pt<dfData.jet2_pt)].values, bins=bins, histtype='step', color='red', label='Data asym>45%', density=True)
ax.hist(dfData.dijet_mass[(~maskData_asym)], bins=bins, histtype='step', color='red', label='Data asym<45%', density=True)
ax.set_xlabel("dijet mass [GeV]")
ax.legend()


# %%

# %%
fig, ax  = plt.subplots(1, 1)
bins=np.linspace(0, 10, 31)
ax.hist(dfData.nJets[maskData_asym], bins=bins, histtype='step', color='black', label='Data asym', density=True)
ax.hist(dfData.nJets[~maskData_asym], bins=bins, histtype='step', color='red', label='Data sym', density=True)
ax.set_xlabel("Njet ")
ax.legend()
# requiring pt1 = pt2 jet pt~ 50/60 gev
# requiring pt1 != pt2 jet pt~ 90 gev
# %%
fig, ax = plt.subplots(1, 1)
def inv_mass_approx(pt1, pt2, dEta, dPhi):
    return np.sqrt(2*pt1*pt2*(np.cosh(dEta) - np.cos(dPhi)))
ax.plot(inv_mass_approx(50, 40, 0.5, 0.5), marker='o', linestyle='none')
ax.plot(inv_mass_approx(90, 25, 2, 1), marker='o', linestyle='none')
# %%


maskMC_asym = abs(dfMC.dijet_pT_asymmetry)>0.45
fig, ax  = plt.subplots(1, 2, figsize=(15,5))
bins=np.linspace(0, 6, 31)
ax[0].hist(dfMC.jet1_genHadronFlavour[maskMC_asym], bins=bins, histtype='step', color='black', label='Data bump', weights=dfMC.weight[maskMC_asym])
ax[0].hist(dfMC.jet1_genHadronFlavour[~maskMC_asym], bins=bins, histtype='step', color='red', label='Data no bump', weights=dfMC.weight[~maskMC_asym])
ax[1].hist(dfMC.jet2_genHadronFlavour[maskMC_asym], bins=bins, histtype='step', color='black', label='Data bump', weights=dfMC.weight[maskMC_asym])
ax[1].hist(dfMC.jet2_genHadronFlavour[~maskMC_asym], bins=bins, histtype='step', color='red', label='Data no bump', weights=dfMC.weight[~maskMC_asym])
ax[0].set_xlabel("jet1_genHadronFlavour ")
ax[1].set_xlabel("jet2_genHadronFlavour ")
ax[0].legend()
ax[1].legend()
# %%
#plt.hist(df.jet3_genHadronFlavour, histtype='step')
fig, ax  = plt.subplots(1, 2, figsize=(15,5))
bins=np.linspace(0, 22, 45)
ax[0].hist(abs(dfMC.jet1_partonFlavour[maskMC_asym]), bins=bins, histtype='step', color='black', label='Data bump', weights=dfMC.weight[maskMC_asym], density=True)
ax[0].hist(abs(dfMC.jet1_partonFlavour[~maskMC_asym]), bins=bins, histtype='step', color='red', label='Data no bump', weights=dfMC.weight[~maskMC_asym], density=True)
ax[1].hist(abs(dfMC.jet2_partonFlavour[maskMC_asym]), bins=bins, histtype='step', color='black', label='Data bump', weights=dfMC.weight[maskMC_asym], density=True)
ax[1].hist(abs(dfMC.jet2_partonFlavour[~maskMC_asym]), bins=bins, histtype='step', color='red', label='Data no bump', weights=dfMC.weight[~maskMC_asym], density=True)
ax[0].set_xlabel("jet1_partonFlavour ")
ax[1].set_xlabel("jet2_partonFlavour ")
ax[0].legend()
ax[1].legend()
# %%

# %%
import collections

counter_sym = collections.Counter(dfMC[~maskMC_asym].QCD_topology)
counter_asym = collections.Counter(dfMC[maskMC_asym].QCD_topology)
all_labels = sorted(set(counter_sym) | set(counter_asym))

values_sym = [counter_sym.get(l, 0) for l in all_labels]
values_asym = [counter_asym.get(l, 0) for l in all_labels]

values_sym = np.array(values_sym) / np.sum(values_sym)
values_asym = np.array(values_asym) / np.sum(values_asym)
fig, ax = plt.subplots(1, 1)

y = np.arange(len(all_labels))
h = 0.4

ax.barh(y - h/2, values_sym, height=h, label="symmetric")
ax.barh(y + h/2, values_asym, height=h, label="asymmetric")

ax.set_yticks(y)
ax.set_yticklabels(all_labels)

ax.set_xlabel("Fraction Events")
ax.set_ylabel("Topology")
ax.legend()

plt.tight_layout()
# %%
fig, ax  = plt.subplots(1, 1)
bins=np.linspace(0, 900, 45)
ax.hist(dfMC.dijet_mass[maskMC_asym], bins=bins, histtype='step', color='black', label='MC asym', weights=dfMC.weight[maskMC_asym], density=False)
ax.hist(dfMC.dijet_mass[~maskMC_asym], bins=bins, histtype='step', color='red', label='MC sym', weights=dfMC.weight[~maskMC_asym], density=False, linewidth=2)
ax.hist(dfMC.dijet_mass[(maskMC_asym) & (dfMC.QCD_topology!="gq->gq | g + q")], bins=bins, histtype='step', label='MC asym without gq->gq', weights=dfMC.weight[(maskMC_asym) & (dfMC.QCD_topology!="gq->gq | g + q")], density=False)
ax.hist(dfMC.dijet_mass[(~maskMC_asym) & (dfMC.QCD_topology!="gg->qq | b + b")], bins=bins, histtype='step', label='MC sym without gg->bb', weights=dfMC.weight[(~maskMC_asym) & (dfMC.QCD_topology!="gg->qq | b + b")], density=False)
ax.set_xlabel("dijet mass ")
ax.legend()
# %%
bins = np.linspace(0, 300, 31)

dfMC_asym = pd.concat(dfs, ignore_index=True)
dfMC_asym = dfMC_asym[abs(dfMC_asym.dijet_pT_asymmetry) > 0.45]

grouped = dfMC_asym.groupby("QCD_topology")

for name, group in grouped:

    fig, ax = plt.subplots(1, 1)

    # Define flavour masks
    jet1_g = group.jet1_partonFlavour == 21
    jet2_g = group.jet2_partonFlavour == 21

    jet1_b = group.jet1_partonFlavour.abs() == 5
    jet2_b = group.jet2_partonFlavour.abs() == 5

    jet1_q = group.jet1_partonFlavour.abs().isin([1, 2, 3, 4])
    jet2_q = group.jet2_partonFlavour.abs().isin([1, 2, 3, 4])


    masks = {
        "gg": jet1_g & jet2_g,

        "gq": (jet1_g & jet2_q) | (jet1_q & jet2_g),
        "gb": (jet1_g & jet2_b) | (jet1_b & jet2_g),

        "qq": jet1_q & jet2_q,
        "qb": (jet1_q & jet2_b) | (jet1_b & jet2_q),

        "bb": jet1_b & jet2_b,
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue

        ax.hist(
            group.loc[mask, "dijet_mass"].values,
            bins=bins,
            weights=group.loc[mask, "weight"].values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label=label
        )
    ax.hist(
            group.dijet_mass.values,
            bins=bins,
            weights=group.weight.values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label="Inclusive",
            color='black',
            linestyle='--'
        )
    ax.set_xlabel("Dijet mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Dijet Mass - {name}")
    ax.legend()

    fig.savefig(
        f"/t3home/gcelotto/ggHbb/documentation/plots/topologies/dijet_mass_{name}.png",
        bbox_inches='tight'
    )
    plt.close(fig)
# %%


# Now save gen information
for name, group in grouped:

    fig, ax = plt.subplots(1, 1)

    # Define flavour masks
    jet1_g = group.jet1_partonFlavour == 21
    jet2_g = group.jet2_partonFlavour == 21

    jet1_b = group.jet1_partonFlavour.abs() == 5
    jet2_b = group.jet2_partonFlavour.abs() == 5

    jet1_q = group.jet1_partonFlavour.abs().isin([1, 2, 3, 4])
    jet2_q = group.jet2_partonFlavour.abs().isin([1, 2, 3, 4])


    masks = {
        "gg": jet1_g & jet2_g,

        "gq": (jet1_g & jet2_q) | (jet1_q & jet2_g),
        "gb": (jet1_g & jet2_b) | (jet1_b & jet2_g),

        "qq": jet1_q & jet2_q,
        "qb": (jet1_q & jet2_b) | (jet1_b & jet2_q),

        "bb": jet1_b & jet2_b,
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue

        ax.hist(
            group.loc[mask, "inv_mass_MEfinalstate"].values,
            bins=bins,
            weights=group.loc[mask, "weight"].values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label=label
        )
    ax.hist(
            group.inv_mass_MEfinalstate.values,
            bins=bins,
            weights=group.weight.values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label="Inclusive",
            color='black',
            linestyle='--'
        )
    ax.set_xlabel("Dijet mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Dijet Mass - {name}")
    ax.legend()

    fig.savefig(
        f"/t3home/gcelotto/ggHbb/documentation/plots/topologies/inv_mass_MEfinalstate_{name}.png",
        bbox_inches='tight'
    )
# %%



































# %%
bins = np.linspace(0, 300, 31)

dfMC_sym = pd.concat(dfs, ignore_index=True)
dfMC_sym = dfMC_sym[abs(dfMC_sym.dijet_pT_asymmetry) < 0.45]

grouped = dfMC_sym.groupby("QCD_topology")

for name, group in grouped:

    fig, ax = plt.subplots(1, 1)

    # Define flavour masks
    jet1_g = group.jet1_partonFlavour == 21
    jet2_g = group.jet2_partonFlavour == 21

    jet1_b = group.jet1_partonFlavour.abs() == 5
    jet2_b = group.jet2_partonFlavour.abs() == 5

    jet1_q = group.jet1_partonFlavour.abs().isin([1, 2, 3, 4])
    jet2_q = group.jet2_partonFlavour.abs().isin([1, 2, 3, 4])


    masks = {
        "gg": jet1_g & jet2_g,

        "gq": (jet1_g & jet2_q) | (jet1_q & jet2_g),
        "gb": (jet1_g & jet2_b) | (jet1_b & jet2_g),

        "qq": jet1_q & jet2_q,
        "qb": (jet1_q & jet2_b) | (jet1_b & jet2_q),

        "bb": jet1_b & jet2_b,
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue

        ax.hist(
            group.loc[mask, "dijet_mass"].values,
            bins=bins,
            weights=group.loc[mask, "weight"].values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label=label
        )
    ax.hist(
            group.dijet_mass.values,
            bins=bins,
            weights=group.weight.values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label="Inclusive",
            color='black',
            linestyle='--'
        )
    ax.set_xlabel("Dijet mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Dijet Mass - {name}")
    ax.legend()

    fig.savefig(
        f"/t3home/gcelotto/ggHbb/documentation/plots/topologies/dijet_mass_sym{name}.png",
        bbox_inches='tight'
    )
    plt.close(fig)
# %%


# Now save gen information
for name, group in grouped:

    fig, ax = plt.subplots(1, 1)

    # Define flavour masks
    jet1_g = group.jet1_partonFlavour == 21
    jet2_g = group.jet2_partonFlavour == 21

    jet1_b = group.jet1_partonFlavour.abs() == 5
    jet2_b = group.jet2_partonFlavour.abs() == 5

    jet1_q = group.jet1_partonFlavour.abs().isin([1, 2, 3, 4])
    jet2_q = group.jet2_partonFlavour.abs().isin([1, 2, 3, 4])


    masks = {
        "gg": jet1_g & jet2_g,

        "gq": (jet1_g & jet2_q) | (jet1_q & jet2_g),
        "gb": (jet1_g & jet2_b) | (jet1_b & jet2_g),

        "qq": jet1_q & jet2_q,
        "qb": (jet1_q & jet2_b) | (jet1_b & jet2_q),

        "bb": jet1_b & jet2_b,
    }
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue

        ax.hist(
            group.loc[mask, "inv_mass_MEfinalstate"].values,
            bins=bins,
            weights=group.loc[mask, "weight"].values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label=label
        )
    ax.hist(
            group.inv_mass_MEfinalstate.values,
            bins=bins,
            weights=group.weight.values,
            density=False,
            histtype='step',
            linewidth=2,
            alpha=0.9,
            label="Inclusive",
            color='black',
            linestyle='--'
        )
    ax.set_xlabel("Dijet mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Dijet Mass - {name}")
    ax.legend()

    fig.savefig(
        f"/t3home/gcelotto/ggHbb/documentation/plots/topologies/inv_mass_MEfinalstate_sym{name}.png",
        bbox_inches='tight'
    )
# %%
dfMC = pd.concat(dfs, ignore_index=True)
# %
plt.hist(dfMC.jet1_muon_pt, bins=np.linspace(7, 20, 51), histtype='step', weights=dfMC.weight)
plt.hist(dfData.jet1_muon_pt, bins=np.linspace(7, 20, 51), histtype='step')
# %%
