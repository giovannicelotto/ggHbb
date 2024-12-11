# %%
import numpy as np
import pandas as pd

# %%
import matplotlib.pyplot as plt

# Number of samples for background and signal
num_bkg = 10000
num_signal = 10000

# Generate data for Feature A (Shoule exponential for bkg, Gaussian for signal)
feature_a_bkg = np.random.exponential(scale=1.0, size=num_bkg)  # Exponential background
feature_a_signal = np.random.normal(loc=2.0, scale=0.5, size=num_signal)  # Gaussian signal

# Generate data for Feature B (0-1 range, peaks at 0 and 1 for bkg, peak at 1 for signal)
feature_b_bkg = np.hstack((np.random.beta(0.5, 0.5, num_bkg // 2), np.random.beta(0.5, 0.5, num_bkg // 2)))
feature_b_signal = np.random.beta(3, 1, num_signal)  # Signal peaks sharply near 1

# Generate data for Feature C (similar to B but wider)
feature_c_bkg = np.hstack((np.random.beta(1, 1, num_bkg // 2), np.random.beta(2, 1, num_bkg // 2)))
feature_c_signal = np.random.beta(2, 1, num_signal)  # Signal wider but peaking near 1

weights_bkg = np.ones(num_bkg)
weights_signal = np.ones(num_signal)*0.01

# Combine background and signal
df = pd.DataFrame({
    "dijet_mass": np.hstack((feature_a_bkg, feature_a_signal)),
    "jet1_btagDeepFlavB": np.hstack((feature_b_bkg, feature_b_signal)),
    "jet2_btagDeepFlavB": np.hstack((feature_c_bkg, feature_c_signal)),
    "label": np.hstack((np.zeros(num_bkg), np.ones(num_signal))),
    "Weights": np.hstack((weights_bkg, weights_signal))
})



from functions import cut
df = cut(data=[df], feature="dijet_mass", min=None, max=5)[0]
df = cut(data=[df], feature="jet1_btagDeepFlavB", min=0.1, max=None)[0]
# Plot the df
plt.figure(figsize=(15, 10))

for i, feature in enumerate(["dijet_mass", "jet1_btagDeepFlavB", "jet2_btagDeepFlavB"], 1):
    plt.subplot(2, 2, i)
    plt.hist(df[df["label"] == 0][feature], bins=20, alpha=0.6, label="Background", weights=df[df["label"] == 0].Weights)
    plt.hist(df[df["label"] == 1][feature], bins=20, alpha=0.6, label="Signal x 100", weights=df[df["label"] == 1].Weights*100)
    plt.title(f"{feature} Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()

plt.tight_layout()
plt.show()

# %%
bins = np.linspace(0, 5, 2)
df_bkg = df[df.label==0]
df_signal = df[df.label==1]
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


    
# %%
cutsCombo = {
    "cut1": [],
    "cut2": [],
    "effBkg":[],
    "effSignal":[],
}
for bmin, bmax in zip(bins[:-1], bins[1:]):
    m = (df_bkg.dijet_mass > bmin) & (df_bkg.dijet_mass < bmax)
    print(len(m), " in mass bin")

    for cut1 in np.linspace(0, 1, 400):
        retainedBkg = len(df_bkg[(df_bkg['jet1_btagDeepFlavB'] > cut1)])/len(df_bkg)
        if retainedBkg < 0.1:
            secondCut = 0
        else:
            for cut2 in np.linspace(0, 1, 400):
                retainedBkg = len(df_bkg[(df_bkg['jet1_btagDeepFlavB'] > cut1) & (df_bkg['jet2_btagDeepFlavB'] > cut2)])/len(df_bkg)
                if retainedBkg<0.1:

                    cutsCombo["cut1"].append(cut1)
                    cutsCombo["cut2"].append(cut2)
                    cutsCombo["effBkg"].append(len(df_bkg[(df_bkg['jet1_btagDeepFlavB'] > cut1) & (df_bkg['jet2_btagDeepFlavB'] > cut2)])/len(df_bkg))
                    cutsCombo["effSignal"].append(len(df_signal[(df_signal['jet1_btagDeepFlavB'] > cut1) & (df_signal['jet2_btagDeepFlavB'] > cut2)])/len(df_signal))
                    break


        

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

# Add a colorbar to indicate effSignal values
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Signal Efficiency", rotation=270, labelpad=15)

# Set axis labels
ax.set_xlabel("Cut on Jet1 btag")
ax.set_ylabel("Cut on Jet2 btag")
#ax.set_title("Signal Efficiency vs. Cuts on Jet btags")

plt.tight_layout()
plt.show()
# %%
