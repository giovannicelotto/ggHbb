import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import uproot
import mplhep as hep
hep.style.use("CMS")
import glob
signal=True


tag='MC' if signal else 'Bkg'
if signal:
    tag='MC'
    nFiles = 40
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000"
else:
    tag = 'Bkg'
    nFiles = 5
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*/"

print("Signal is set to ", signal)
print("path : ", path)
print("nFiles : ", nFiles)
fileNames = glob.glob(path +"/*.root")
ntrigMu = []
for fileName in fileNames[:nFiles]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    print("Entries : %d"%maxEntries)
    for ev in range(maxEntries):
        Muon_isTriggering   = branches["Muon_isTriggering"][ev]
        ntrigMu.append(np.sum(Muon_isTriggering))

fig, ax = plt.subplots(1, 1)
bins=np.arange(-0.5, 4.5)
midpoints = (bins[1:]+bins[:-1])/2
counts= np.histogram(ntrigMu, bins=bins)[0]
counts=counts*100/np.sum(counts)
ax.hist(bins[:-1], bins=bins, weights=counts, color='blue', histtype=u'step')[0]
ax.set_ylim(0, 104)
ax.set_ylabel("Percentage [%]")
ax.set_xlabel("N triggering muon")
for i, count in enumerate(counts):
    ax.text(midpoints[i], count, "%.1f%%"%(count), ha='center', va='bottom', color='black')
outName = "/t3home/gcelotto/ggHbb/outputs/plots/nTrigMu_%s.png"%tag
print("Saving in ", outName)
fig.savefig(outName)


print("Done")





