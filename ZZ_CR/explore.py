# %%
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

import mplhep as hep
hep.style.use("CMS")
import awkward as ak
import uproot
import vector
vector.register_awkward()
# %%
#path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_1.root"
path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/crab_ZJetsToQQ200to400/250409_155410/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_36.root"
f = uproot.open(path)
tree = f["Events"]
branches = tree.arrays()
Muon_pt = branches["Muon_pt"]
Muon_eta = branches["Muon_eta"]
Muon_phi = branches["Muon_phi"]
Muon_mass = branches["Muon_mass"]
nMuon = branches["nMuon"]

m = nMuon>=2


import vector

# Apply mask: events with at least 2 muons
Muon_pt   = Muon_pt[m]
Muon_eta  = Muon_eta[m]
Muon_phi  = Muon_phi[m]
Muon_mass = Muon_mass[m]

# Build Lorentz vectors per muon using awkward + vector
muons = ak.zip({
    "pt": Muon_pt,
    "eta": Muon_eta,
    "phi": Muon_phi,
    "mass": Muon_mass,
}, with_name="Momentum4D")

# Take the first two muons from each event
mu1 = muons[:, 0]
mu2 = muons[:, 1]
dimuon_mass = (mu1 + mu2).mass
dimuon_pt = (mu1 + mu2).pt
# %%
# Plotting
plt.figure(figsize=(8,6))
plt.hist(dimuon_mass, bins=101, range=(0,120), histtype='step', linewidth=1.5)
plt.xlabel("Dimuon invariant mass [GeV]")
plt.ylabel("Events")
hep.cms.label()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.hist(dimuon_pt[(dimuon_mass>80) & (dimuon_mass<100)], bins=101, range=(0,300), histtype='step', linewidth=1.5)
plt.xlabel("Dimuon pt When 80<mmumu<100 ")
plt.ylabel("Events")
hep.cms.label()
plt.grid(True)
plt.show()


# %%
#   
#   
#   Try again with Electrons
#   
#   
#   
#   

Electron_pt = branches["Electron_pt"]
Electron_eta = branches["Electron_eta"]
Electron_phi = branches["Electron_phi"]
Electron_mass = branches["Electron_mass"]
nElectron = branches["nElectron"]

m = nElectron>=3


import vector

# Apply mask: events with at least 2 muons
Electron_pt   = Electron_pt[m]
Electron_eta  = Electron_eta[m]
Electron_phi  = Electron_phi[m]
Electron_mass = Electron_mass[m]

# Build Lorentz vectors per muon using awkward + vector
electrons = ak.zip({
    "pt": Electron_pt,
    "eta": Electron_eta,
    "phi": Electron_phi,
    "mass": Electron_mass,
}, with_name="Momentum4D")

# Take the first two electrons from each event
ele1 = electrons[:, 0]
ele2 = electrons[:, 1]
diele_mass = (ele1 + ele2).mass
# %%
# Plotting
plt.figure(figsize=(8,6))
plt.hist(diele_mass, bins=100, range=(0,200), histtype='step', linewidth=1.5)
plt.xlabel("Dielectron invariant mass [GeV]")
plt.ylabel("Events")
hep.cms.label()
plt.grid(True)
plt.show()

# %%
