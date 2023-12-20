import uproot
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep
hep.style.use("CMS")
signal=False


signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*/"
signalIso, realDataIso=[], []
signalPt, realDataPt=[], []

totalVisited = 0
fileNames = glob.glob(signalPath +"/*.root")
for fileName in fileNames[:10]:
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
            signalPt+=[i for i in Muon_pt[(maskEta)&(maskLoose)&(maskPt)]]
            
        


fileNames = glob.glob(realDataPath +"/*.root")
totalVisited = 0
for fileName in fileNames[:1]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    totalVisited+=maxEntries
    print("Entries : %d"%maxEntries)

    for ev in range(10):
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
            realDataPt+=[i for i in Muon_pt[(maskEta)&(maskLoose)&(maskPt)]]
        

x_bins, y_bins = np.linspace(10, 60, 30), np.linspace(0, 6, 30)
fig, ax_main = plt.subplots(figsize=(8, 8))
divider = make_axes_locatable(ax_main)
ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

# Plot the 2D histogram in the main axes
hist, x_edges, y_edges = np.histogram2d(x=signalPt, y=signalIso, bins=[x_bins, y_bins])
im=ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='Blues', norm=LogNorm())
cax = divider.append_axes("right", 0.5, pad=0.3)

# Add colorbar
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Events')

ax_main.set_xlabel("Muon Pt [GeV]")
ax_main.set_ylabel("Muon PfRelIso03_all")

# Plot the marginalized histogram on top
ax_top.hist(signalPt, bins=x_bins, color='lightblue', edgecolor='black')
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_yticks([])
ax_top.xaxis.tick_top()

# Plot the marginalized histogram on the right
ax_right.hist(signalIso, bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
ax_right.set_ylim(ax_main.get_ylim())
ax_right.set_xticks([])
ax_right.yaxis.tick_right()

ax_main.text(x=0.9, y=0.85, s="Pt > 10 GeV", ha='right', transform=ax_main.transAxes, fontsize=15)
ax_main.text(x=0.9, y=0.8, s=r"|$\eta$|<2.4", ha='right', transform=ax_main.transAxes, fontsize=15)
ax_main.text(x=0.9, y=0.75, s="LooseID", ha='right', transform=ax_main.transAxes, fontsize=15)

outName = "/t3home/gcelotto/ggHbb/outputs/plots/correlationMuonPtIsoMasked.png"
print("Saving in ", outName)
fig.savefig(outName, bbox_inches='tight')