import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import uproot
import mplhep as hep
hep.style.use("CMS")
import glob
signal=False


tag='MC' if signal else 'Bkg'
if signal:
    tag='MC'
    nFiles = 1
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000"
else:
    tag = 'Bkg'
    nFiles = 1
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data1A2024Mar05/ParkingBPH1/crab_data_Run2018A_part1/240305_082004/0000"

print("Signal is set to ", signal)
print("path : ", path)
print("nFiles : ", nFiles)
fileNames = glob.glob(path +"/*.root")
trigJet = []
nTotalJet = []
for fileName in fileNames[:nFiles]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    print("Entries : %d"%maxEntries)
    for ev in range(maxEntries):
        Jet_muonIdx1        = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2        = branches["Jet_muonIdx2"][ev]
        Jet_eta             = branches["Jet_eta"][ev]
        nJet                = branches["nJet"][ev]
        Muon_isTriggering   = branches["Muon_isTriggering"][ev]
        nTrigJetEvent       = 0
        nTotalJet.append(nJet)

        for jetIdx in range(nJet):
            if Jet_muonIdx1[jetIdx]>-1:
                if (Muon_isTriggering[Jet_muonIdx1[jetIdx]]) & (abs(Jet_eta[jetIdx])<2.5) :#& (np.sum(abs(Jet_eta[:4])<2.5)>1):
                    nTrigJetEvent=nTrigJetEvent+1
                    continue
            if Jet_muonIdx2[jetIdx]>-1:
                if (Muon_isTriggering[Jet_muonIdx2[jetIdx]]) & (abs(Jet_eta[jetIdx])<2.5) :#& (np.sum(abs(Jet_eta[:4])<2.5)>1):
                    nTrigJetEvent=nTrigJetEvent+1
                    continue
        trigJet.append(nTrigJetEvent)

fig, ax = plt.subplots(1, 1)
bins=np.arange(-0.5, 4.5)
midpoints = (bins[1:]+bins[:-1])/2
counts= np.histogram(trigJet, bins=bins)[0]
counts=counts*100/np.sum(counts)
ax.hist(bins[:-1], bins=bins, weights=counts, color='blue', histtype=u'step')[0]
ax.set_ylim(0, 104)
ax.set_ylabel("Percentage [%]")
ax.set_xlabel("N$_\mathrm{jets}$ with Triggering Muon")
for i, count in enumerate(counts):
    ax.text(midpoints[i], count, "%.1f%%"%(count), ha='center', va='bottom', color='black')
outName = "/t3home/gcelotto/ggHbb/outputs/plots/nTrigJet_%s.png"%tag
print("Savin in ", outName)
fig.savefig(outName)

fig, ax = plt.subplots(1, 1)
bins=np.arange(-0.5, 39.5, 1)
midpoints = (bins[1:]+bins[:-1])/2
counts= np.histogram(nTotalJet, bins=bins)[0]
counts=counts*100/np.sum(counts)
ax.hist(bins[:-1], bins=bins, weights=counts, color='blue', histtype=u'step')[0]
#ax.set_ylim(0, 104)
ax.set_ylabel("Percentage [%]")
ax.set_xlabel("N$_\mathrm{jets}$")
ax.text(x=0.95, y=0.9, s="Mean    %.2f"%np.mean(nTotalJet), ha='right', transform=ax.transAxes)
ax.text(x=0.95, y=0.84, s="Std Dev    %.2f"%np.std(nTotalJet), ha='right', transform=ax.transAxes)
#ax.text(s="(13 TeV)", x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
#for i, count in enumerate(counts):
#    ax.text(midpoints[i], count, "%.1f%%"%(count), ha='center', va='bottom', color='black')
outName = "/t3home/gcelotto/ggHbb/outputs/plots/nJet_%s.png"%tag
print("saving in ", outName)
fig.savefig(outName)


print("Done")





