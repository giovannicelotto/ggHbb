# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import argparse
import numpy as np
import uproot
# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=2, help='Index value (default: 2)')
    parser.add_argument('-b', '--boson', type=str, default="Higgs", help='Higgs or Z')

    args = parser.parse_args()
    idx = args.idx
except:
    idx = 2
# %%
path = [
    "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/hists/counts_cat1.root",
    "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/hists/counts_cat2p0.root",
    "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/hists/counts_cat2p1.root"
][idx]
outFolder=["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/plots/variations",
           "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/plots/variations",
           "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/plots/variations"][idx]
file = uproot.open(path)

variations = [
"btag",
"JER","JECAbsoluteMPFBias","JECAbsoluteScale","JECAbsoluteStat","JECFlavorQCD","JECFragmentation","JECPileUpDataMC","JECPileUpPtBB","JECPileUpPtEC1","JECPileUpPtEC2","JECPileUpPtHF","JECPileUpPtRef","JECRelativeBal","JECRelativeFSR","JECRelativeJEREC1","JECRelativeJEREC2","JECRelativeJERHF","JECRelativePtBB","JECRelativePtEC1","JECRelativePtEC2","JECRelativePtHF","JECRelativeSample","JECRelativeStatEC","JECRelativeStatFSR","JECRelativeStatHF","JECSinglePionECAL","JECSinglePionHCAL","JECTimePtEta",
]
# %%
from matplotlib import gridspec
values_nom = file["%s_nominal"%(args.boson)].to_numpy()[0]
bins = file["%s_nominal"%(args.boson)].to_numpy()[1]
bin_centers_nom = (bins[1:] + bins[:-1])/2
err_nom = np.sqrt(file["Fit_nominal"].variances())
for var in variations:
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    ax_main = fig.add_subplot(gs[:, 0])  # Main plot (S/σ), spans two rows
    ax_val = fig.add_subplot(gs[0, 1])   # Upper right (values)
    ax_err = fig.add_subplot(gs[1, 1])   # Lower right (errors)

    totalNaiveSig = {}
    totalNaiveSig["Nominal"]=np.sqrt(np.sum(values_nom**2 / (err_nom + 1e-12)**2))
    for dir in ["Up", "Down"]:
        values = file["%s_%s_%s"%(args.boson, var, dir)].to_numpy()[0]
        bins = file["%s_%s_%s"%(args.boson, var, dir)].to_numpy()[1]
        bin_centers = (bins[1:] + bins[:-1])/2
        err = np.sqrt(file["Fit_%s_%s"%(var, dir)].variances())

        # Main plot
        ax_main.plot(bin_centers, values / (err + 1e-12), label='%s'%(dir), marker='o', linestyle='none')

        # Value plot
        ax_val.plot(bin_centers, values, label='%s'%(dir), marker='o', linestyle='none')

        # Error plot
        ax_err.plot(bin_centers, err, label='%s'%(dir), marker='o', linestyle='none')

        totalNaiveSig[dir]=np.sqrt(np.sum(values**2 / (err + 1e-12)**2))


#    print(totalNaiveSig)
    ax_main.hist(bins[:-1], bins=bins, weights=values_nom / (err_nom + 1e-12), histtype='step', color='black')
    ax_main.set_ylabel(r"S/$\sigma$")
    ax_main.text(x=0.9, y=0.9, s="%s"%(var), transform=ax_main.transAxes, ha='right')
    ax_main.text(x=0.9, y=0.7, s="Up: %.2f \nNom: %.2f \nDown: %.2f "%(totalNaiveSig["Up"], totalNaiveSig["Nominal"], totalNaiveSig["Down"]), transform=ax_main.transAxes, ha='right', fontsize=18)
    ax_main.legend(loc='center right')

    ax_val.hist(bins[:-1], bins=bins, weights=values_nom, histtype='step', color='black')
    ax_val.set_ylabel("Values")
    ax_err.hist(bins[:-1], bins=bins, weights=err_nom, histtype='step', color='black')
    ax_err.set_ylabel("Errors")
    ax_val.set_xlim(70, 180)
    ax_err.set_xlim(70, 180)

    ax_main.legend(fontsize='small', loc='best')

    fig.savefig(outFolder+"/%s.png"%var)

    plt.tight_layout()
    plt.show()
    plt.close()

# %%
