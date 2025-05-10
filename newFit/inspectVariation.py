# %%
import matplotlib.pyplot as plt
import numpy as np
import uproot 
syst = "JECAbsoluteScale"
f = uproot.open("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/hists/counts_cat2p1.root")
histograms = {
    "higgs": f['Higgs_nominal'].to_numpy()[0],
    "higgs_%s_Up"%(syst): f['Higgs_%s_Up'%(syst)].to_numpy()[0],
    "higgs_%s_Down"%(syst): f['Higgs_%s_Down'%(syst)].to_numpy()[0],
    "Fit": f['Fit_nominal'].to_numpy()[0],
    "Fit_%s_Up"%(syst): f['Fit_%s_Up'%(syst)].to_numpy()[0],
    "Fit_%s_Down"%(syst): f['Fit_%s_Down'%(syst)].to_numpy()[0],
}
# %%
def plot_systematic_variation(histograms, syst_name, processes=["higgs", "Fit"], bins=None):
    for proc in processes:
        nominal = histograms[proc]
        up = histograms.get(f"{proc}_{syst_name}_Up", None)
        down = histograms.get(f"{proc}_{syst_name}_Down", None)

        if bins is None:
            bins = np.arange(len(nominal))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Top: Nominal and systematic variations
        ax1.step(bins, nominal, where='mid', label=f"{proc} (nominal)", linewidth=2)
        if up is not None:
            ax1.step(bins, up, where='mid', linestyle='--', label=f"{proc} ({syst_name} up)")
        if down is not None:
            ax1.step(bins, down, where='mid', linestyle='--', label=f"{proc} ({syst_name} down)")

        ax1.set_ylabel("Events")
        ax1.set_title(f"Systematic Variation: {syst_name} ({proc})")
        ax1.legend()
        ax1.grid(True)

        # Bottom: Residuals (variation - nominal)
        if up is not None:
            ax2.step(bins, np.array(up) - np.array(nominal), where='mid', linestyle='--', label="Up - Nominal")
        if down is not None:
            ax2.step(bins, np.array(down) - np.array(nominal), where='mid', linestyle='--', label="Down - Nominal")

        ax2.axhline(0, color='black', linestyle=':')
        ax2.set_xlabel("Bin")
        ax2.set_ylabel("Δ Events")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
# %%
edges = f['Higgs_nominal'].to_numpy()[1]  # [1] gives bin edges
bin_centers = 0.5 * (edges[:-1] + edges[1:])  # compute bin centers
plot_systematic_variation(histograms, syst, bins=bin_centers)

# %%
