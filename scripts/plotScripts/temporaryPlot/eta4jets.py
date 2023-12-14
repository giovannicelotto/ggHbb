import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import uproot
import mplhep as hep
hep.style.use("CMS")
import glob
signal=False



nFilesSignal = 100
pathSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000"

nFilesBkg = 10
pathBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*/"

print("Signal is set to ", signal)

fileNamesSignal = glob.glob(pathSignal +"/*.root")
fileNamesBkg = glob.glob(pathBkg +"/*.root")
Eta4JetsSignal, Eta4JetsBkg = [], []

for fileName in fileNamesSignal[:nFilesSignal]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    print("Entries : %d"%maxEntries)
    for ev in range(maxEntries):
        Jet_eta             = branches["Jet_eta"][ev]
        nJet                = branches["nJet"][ev]
        
        
        jetsToCheck = 4 if 4 < nJet else int(nJet)
        for jetIdx in range(jetsToCheck):
            Eta4JetsSignal.append(Jet_eta[jetIdx])

for fileName in fileNamesBkg[:nFilesBkg]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    print("Entries : %d"%maxEntries)
    for ev in range(maxEntries):
        Jet_eta             = branches["Jet_eta"][ev]
        nJet                = branches["nJet"][ev]
        
        
        jetsToCheck = 4 if 4 < nJet else int(nJet)
        for jetIdx in range(jetsToCheck):
            Eta4JetsBkg.append(Jet_eta[jetIdx])

fig, ax = plt.subplots(1, 1)
bins=np.linspace(-5, 5, 101)
midpoints = (bins[1:]+bins[:-1])/2
counts= np.histogram(Eta4JetsSignal, bins=bins)[0]
counts=counts*100/np.sum(counts)
ax.hist(bins[:-1], bins=bins, weights=counts, color='blue', histtype=u'step', label='Signal')[0]
counts= np.histogram(Eta4JetsBkg, bins=bins)[0]
counts=counts*100/np.sum(counts)
ax.hist(bins[:-1], bins=bins, weights=counts, color='red', histtype=u'step', label='Background')[0]
ax.legend()



ax.set_xlabel("$\eta$ first 4 jets")
ax.set_ylabel("Probability")
#for i, count in enumerate(counts):
#    ax.text(midpoints[i], count, "%.1f%%"%(count), ha='center', va='bottom', color='black')
outName = "/t3home/gcelotto/ggHbb/outputs/plots/nEta4Jets.png"
print("Savin in ", outName)
fig.savefig(outName)



print("Done")





