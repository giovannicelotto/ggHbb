import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import uproot

f = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/ggH_noTrig_Run2_mc_124X.root")
tree = f['Events']
branches = tree.arrays()
maxEntries = 1000#tree.num_entries
GenPart_pt          = branches["GenPart_pt"]
GenPart_pdgId       = branches["GenPart_pdgId"]
GenPart_statusFlags = branches["GenPart_statusFlags"]

m = (GenPart_pdgId==25) & (GenPart_statusFlags>=8192)
GenPart_pt = GenPart_pt[m]
print("length = %d"%len(GenPart_pt))
bins = np.linspace(0, 800, 801)
counts = np.histogram(GenPart_pt, bins=bins)[0]
midpoints = (bins[1:] + bins[:-1])/2



# Plot the line with exponential decay background and a small signal on top
with plt.xkcd():
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(midpoints, counts, color='blue', label='ggF Higgs pT')
    #ax.plot(x_signal, background_data*10000, '--', color='red', label='Background Fit')

    # Style the plot
    ax.set_xlim(0, 600)
    #ax.title('Invariant Mass Distribution with Exponential Decay and Signal')
    ax.set_xlabel('Higgs p$_T$')
    ax.set_ylabel('Events')
    #ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # Show the plot
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/Higgs_pT_cartoon.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saving in %s"%outName)
