# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import cut
import glob

# %%
pathSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training"
pathData = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training"


df_signal = pd.read_parquet(pathSignal)
df_bkg = pd.read_parquet(glob.glob(pathData+"/*.parquet")[:1])
# %%
dfs = [df_signal, df_bkg]
dfs = cut(dfs, 'dijet_mass', 40, 300)

fig, ax = plt.subplots(1,3, figsize=(15, 5))
for i, feature in enumerate(["dijet_mass", "jet1_btagDeepFlavB", "jet2_btagDeepFlavB"]):
    ax[i].hist(dfs[1][feature], bins=20, alpha=0.6, label="Signal")
    ax[i].hist(dfs[0][feature], bins=20, alpha=0.6, label="Bkg")
    ax[i].set_xlabel(feature)
    ax[i].set_ylabel("Frequency")
    ax[i].legend()

plt.tight_layout()
plt.show()

# %%
bins = np.linspace(40, 300, 2)
for bmin, bmax in zip(bins[:-1], bins[1:]):
    m = (df_bkg.dijet_mass > bmin) & (df_bkg.dijet_mass < bmax)
    print(len(m), " in mass bin")
    best_signal = 0
    best_cut1, best_cut2 = None, None
    for cut1 in np.linspace(0, 1, 20):
        for cut2 in np.linspace(0, 1, 20):
            signal_selected = df_signal[(df_signal['jet1_btagDeepFlavB'] > cut1) & (df_signal['jet2_btagDeepFlavB'] > cut2)]
            background_selected = df_bkg[(df_bkg['jet1_btagDeepFlavB'] > cut1) & (df_bkg['jet2_btagDeepFlavB'] > cut2)]

            background_fraction = len(background_selected) / len(df_bkg)
            if abs(background_fraction)<0.1:
                signal_retained = len(signal_selected)

                if signal_retained > best_signal:
                    best_signal = signal_retained
                    best_cut1, best_cut2 = cut1, cut2
print(best_cut1, best_cut2)


    # Constraint satisfied


    
#
#  %%
retainedDataFraction = 0.2
cutsComboCollector = []
bins = np.linspace(40, 300, 20)
for bmin, bmax in zip(bins[:-1], bins[1:]):
    cutsCombo = {
        "cut1": [],
        "cut2": [],
        "effBkg":[],
        "effSignal":[],
    }
    m_bkg = (df_bkg.dijet_mass > bmin) & (df_bkg.dijet_mass < bmax)
    m_signal = (df_signal.dijet_mass > bmin) & (df_signal.dijet_mass < bmax)


    for cut1 in np.linspace(0, 1, 20):
        print(cut1)
        retainedBkg = len(df_bkg[m_bkg][(df_bkg[m_bkg]['jet1_btagDeepFlavB'] > cut1)])/len(df_bkg[m_bkg])
        if retainedBkg < retainedDataFraction:
            secondCut = 0
        else:
            for cut2 in np.linspace(0, 1, 20):
                retainedBkg = len(df_bkg[m_bkg][(df_bkg[m_bkg]['jet1_btagDeepFlavB'] > cut1) & (df_bkg[m_bkg]['jet2_btagDeepFlavB'] > cut2)])/len(df_bkg[m_bkg])
                if retainedBkg<retainedDataFraction:

                    cutsCombo["cut1"].append(cut1)
                    cutsCombo["cut2"].append(cut2)
                    cutsCombo["effBkg"].append(len(df_bkg[m_bkg][(df_bkg[m_bkg]['jet1_btagDeepFlavB'] > cut1) & (df_bkg[m_bkg]['jet2_btagDeepFlavB'] > cut2)])/len(df_bkg[m_bkg]))
                    cutsCombo["effSignal"].append(len(df_signal[m_signal][(df_signal[m_signal]['jet1_btagDeepFlavB'] > cut1) & (df_signal[m_signal]['jet2_btagDeepFlavB'] > cut2)])/len(df_signal[m_signal]))
                    break
    cutsComboCollector.append(cutsCombo)




# %%

ncol = 4
nrow = len(cutsComboCollector) // ncol + (len(cutsComboCollector) % ncol > 0)  # Ensure enough columns
fig, ax = plt.subplots(nrow, ncol, constrained_layout=True, figsize=(30, 20), sharex=True, sharey=True)
axes = ax.flat if nrow > 1 and ncol > 1 else [ax]
for i, dataset in enumerate(cutsComboCollector):
    ax = axes[i]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    scatter = ax.scatter(
            cutsComboCollector[i]["cut1"], 
            cutsComboCollector[i]["cut2"], 
            c=cutsComboCollector[i]["effSignal"],
            cmap="viridis",  
            s=50)
    maxIdx = np.argmax(cutsComboCollector[i]["effSignal"])
    ax.text(x=0.1, y=0.9, s="Signal Eff : %.1f%%"%(cutsComboCollector[i]["effSignal"][maxIdx]*100), fontsize=24)
    ax.plot(cutsComboCollector[i]["cut1"][maxIdx], cutsComboCollector[i]["cut2"][maxIdx], marker='x', color='red')
    ax.set_title("%.1f < mjj < %.1f"%(bins[i], bins[i+1]), fontsize=28)
    ax.vlines(x=cutsComboCollector[i]["cut1"][maxIdx], ymin=0, ymax=cutsComboCollector[i]["cut2"][maxIdx], color='black', linestyle='dotted')
    ax.hlines(y=cutsComboCollector[i]["cut2"][maxIdx], xmin=0, xmax=cutsComboCollector[i]["cut1"][maxIdx], color='black', linestyle='dotted')
    ax.tick_params(labelsize=24)
for j in range(len(cutsComboCollector), len(axes)):
    axes[j].axis("off")
fig.savefig("/t3home/gcelotto/ggHbb/abcd/dynamicABCD/scan.png")

# %%
import json
with open('/t3home/gcelotto/ggHbb/abcd/dynamicABCD/cutsComboCollector.json', 'w') as f:
    json.dump(cutsComboCollector, f, indent=4)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Use scatter plot to include color mapping
scatter = ax.scatter(
    cutsCombo["cut1"],  # X-axis
    cutsCombo["cut2"],  # Y-axis
    c=cutsCombo["effSignal"],  # Color based on effSignal
    cmap="viridis",  # Colormap (you can use other colormaps like "plasma", "coolwarm", etc.)
    s=50,  # Point size
)

maxIdx = np.argmax(cutsCombo["effSignal"])
ax.plot(cutsCombo["cut1"][maxIdx], cutsCombo["cut2"][maxIdx], marker='x', color='red')
ax.vlines(x=cutsCombo["cut1"][maxIdx], ymin=0, ymax=cutsCombo["cut2"][maxIdx], color='black', linestyle='dotted')
ax.hlines(y=cutsCombo["cut2"][maxIdx], xmin=0, xmax=cutsCombo["cut1"][maxIdx], color='black', linestyle='dotted')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Add a colorbar to indicate effSignal values
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Signal Efficiency", rotation=270, labelpad=15)

# Set axis labels
ax.set_xlabel("Cut on Jet1 btag")
ax.set_ylabel("Cut on Jet2 btag")
#ax.set_title("Signal Efficiency vs. Cuts on Jet btags")

plt.tight_layout()
plt.show()