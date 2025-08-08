import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from hist import Hist
def compute_efficiency(jet_pt, jet_eta, jet_puid, pt_bins, eta_bins, WP):
    abs_eta = abs(jet_eta)

    eff_map = {}

    # Mask per flavour


    pt_f = ak.flatten(jet_pt)
    eta_f = ak.flatten(abs_eta)

    jet_puid_f = ak.flatten(jet_puid)


    # Histogram total jets
    h_total = Hist.new.Var(pt_bins).Var(eta_bins).Double()
    h_total.fill(pt_f, eta_f)

    # Histogram b-tagged jets
    tagged_mask = jet_puid_f >= WP
    h_tagged = Hist.new.Var(pt_bins).Var(eta_bins).Double()
    h_tagged.fill(pt_f[tagged_mask], eta_f[tagged_mask])

    # Efficiency
    #efficiency = h_total.values()
    efficiency =  np.divide(h_tagged.values(), h_total.values(),
                    out=np.zeros_like(h_tagged.values()),
                    where=h_total.values() != 0)

    return efficiency


def plotEfficiencyMaps(pt_bins, eta_bins, eff_map, outFolder=None, process=None, PUID_WP=None):
    if process is None:
        process=""


    # Calculate bin centers for annotation (logical positions, not physical pT)
    pt_bin_centers = np.arange(len(pt_bins) - 1)
    eta_bin_centers = np.arange(len(eta_bins) - 1)

    # For display: make string labels for ticks
    pt_bin_labels = [f"{pt_bins[i]}-{pt_bins[i+1]}" for i in range(len(pt_bins) - 1)]
    eta_bin_labels = [f"{eta_bins[i]}-{eta_bins[i+1]}" for i in range(len(eta_bins) - 1)]


    efficiency = eff_map


    print(efficiency)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        efficiency.T,
        origin='lower',
        aspect='equal',  # make all bins same size visually
        extent=[0, len(pt_bins) - 1, 0, len(eta_bins) - 1],
        interpolation='nearest',
        cmap='viridis',
        vmin=0,  # fix lower limit
        vmax=1   # fix upper limit
    )
    cbar = fig.colorbar(im, ax=ax, label='Efficiency', fraction=0.046, pad=0.04)


    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(pt_bin_labels)) + 0.5)
    ax.set_xticklabels(pt_bin_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(eta_bin_labels)) + 0.5)
    ax.set_yticklabels(eta_bin_labels)
    ax.set_xlabel('Jet $p_T$ bin [GeV]')
    ax.set_ylabel(r'$|\eta|$ bin')
    ax.set_title(f'{process} jet Efficiency')

    # Add text annotations
    for i in range(len(pt_bin_centers)):
        for j in range(len(eta_bin_centers)):
            val = efficiency[i, j]
            if not np.isnan(val) and val > 0:
                ax.text(i + 0.5, j + 0.5, f"{val:.3f}",
                        color="white", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    if outFolder is None:
        plt.show()
    else:
        fig.savefig(outFolder+f"/{process}_{PUID_WP}.png", bbox_inches='tight')
        plt.close('all')