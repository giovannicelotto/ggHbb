import uproot
import numpy as np
import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
signal=False


signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*/"
signalIso, realDataIso=[], []

totalVisited = 0
fileNames = glob.glob(signalPath +"/*.root")
for fileName in fileNames[:5]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    totalVisited+=maxEntries
    print("Entries : %d"%maxEntries)

    for ev in range(maxEntries):
        Muon_pfRelIso03_all    = branches["Muon_pfRelIso03_all"][ev]
        nMuon                   = branches["nMuon"][ev]
        Muon_tightId            = branches["Muon_tightId"][ev]
        Muon_looseId            = branches["Muon_looseId"][ev]
        Muon_pt            = branches["Muon_pt"][ev]
        Muon_eta            = branches["Muon_eta"][ev]
        
        filled=False
        if nMuon>0:
            maskPt = Muon_pt>10
            maskEta = abs(Muon_eta)<2.4
            maskLoose = Muon_looseId==1
            maskIso = Muon_pfRelIso03_all<0.25
            signalIso+=[i for i in Muon_pfRelIso03_all[(maskEta)&(maskLoose)&(maskPt)]]
        


fileNames = glob.glob(realDataPath +"/*.root")
totalVisited = 0
for fileName in fileNames[:1]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    totalVisited+=maxEntries
    print("Entries : %d"%maxEntries)

    for ev in range(6000):
        Muon_pfRelIso03_all    = branches["Muon_pfRelIso03_all"][ev]
        nMuon                   = branches["nMuon"][ev]
        Muon_tightId            = branches["Muon_tightId"][ev]
        Muon_looseId            = branches["Muon_looseId"][ev]
        Muon_pt            = branches["Muon_pt"][ev]
        Muon_eta            = branches["Muon_eta"][ev]
        
        filled=False
        if nMuon>0:
            maskPt = Muon_pt>10
            maskEta = abs(Muon_eta)<2.4
            maskLoose = Muon_looseId==1
            maskIso = Muon_pfRelIso03_all<0.25
            realDataIso+=[i for i in Muon_pfRelIso03_all[(maskEta)&(maskLoose)&(maskPt)]]
        

bins = np.linspace( 0, 10, 101)
fig, ax = plt.subplots(1, 1)
cs = np.histogram(np.clip(signalIso, bins[0], bins[-1]), bins=bins)[0]
cb = np.histogram(np.clip(realDataIso, bins[0], bins[-1]), bins=bins)[0]
cs=cs/np.sum(cs)
cb=cb/np.sum(cb)
ax.hist(bins[:-1], bins=bins, weights=cs, histtype=u'step', label='signal', color='blue')
ax.hist(bins[:-1], bins=bins, weights=cb, histtype=u'step', label='bkg', color='red')


ax.set_xlabel("Muon_pfRelIso03_all")
ax.text(x=0.9, y=0.55, s="Muon_pt>10", ha='right', transform=ax.transAxes)
ax.text(x=0.9, y=0.5, s="|Muon_eta|<2.4", ha='right', transform=ax.transAxes)
ax.text(x=0.9, y=0.45, s="LooseID=1", ha='right', transform=ax.transAxes)
outName = "/t3home/gcelotto/ggHbb/outputs/plots/muon_isolation_masked.png"
print("saving in ", outName)
ax.legend()
fig.savefig(outName, bbox_inches='tight')




