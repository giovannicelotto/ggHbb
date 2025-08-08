# %%
import uproot 
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT
hep.style.use("CMS")
# %%

import awkward as ak
import glob
# Step 1: Use glob to get all matching ROOT files
all_files = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/crab_ZJetsToQQ200to400/250409_155410/0000/*.root")

# Step 2: Select only the first 3 files
selected_files = all_files[:10]

# Step 3: Create the file-to-TTree mapping
files = {file: "Events" for file in selected_files}

# Step 4: Load and concatenate events
tree = uproot.iterate(files, library="ak")
events = ak.concatenate([chunk for chunk in tree])

# Done: 'events' now holds the concatenated data from 3 files
print(events)

# %%
lljj_mass = []
lljj_pt = []
for ev in range(len(events)):
    Muon_pt = events["Muon_pt"][ev]
    Muon_eta = events["Muon_eta"][ev]
    Muon_phi = events["Muon_phi"][ev]
    Muon_charge = events["Muon_charge"][ev]

    Jet_pt = events["Jet_pt"][ev]
    Jet_eta = events["Jet_eta"][ev]
    Jet_phi = events["Jet_phi"][ev]
    Jet_mass = events["Jet_mass"][ev]

    if len(Jet_pt)<2:
        continue
    if len(Muon_pt)<2:
        continue
    jet1    = ROOT.TLorentzVector(0.,0.,0.,0.)
    jet2    = ROOT.TLorentzVector(0.,0.,0.,0.)
    muon1   = ROOT.TLorentzVector(0.,0.,0.,0.)
    muon2   = ROOT.TLorentzVector(0.,0.,0.,0.)
    jet1.SetPtEtaPhiM(Jet_pt[0], Jet_eta[0], Jet_phi[0], Jet_mass[0])
    jet2.SetPtEtaPhiM(Jet_pt[1], Jet_eta[1], Jet_phi[1], Jet_mass[1])
    if Muon_charge[0]!=Muon_charge[1]:
        muon1.SetPtEtaPhiM(Muon_pt[0], Muon_eta[0], Muon_phi[0], 0.106)
        muon2.SetPtEtaPhiM(Muon_pt[1], Muon_eta[1], Muon_phi[1], 0.106)
    else:
        continue

    lljj = jet1 + jet2 + muon1 + muon2
    lljj_mass.append(lljj.M())
    lljj_pt.append(lljj.Pt())
# %%
lljj_mass = np.array(lljj_mass)
lljj_pt = np.array(lljj_pt)
import numpy as np
fig, ax = plt.subplots(1, 1)
ax.hist(lljj_mass[lljj_pt>100], bins=np.linspace(0, 500, 51))
# %%
import numpy as np

for ev in range(len(events)):
    # Per-event arrays
    pdgIds = events["GenPart_pdgId"][ev]
    motherIdx = events["GenPart_genPartIdxMother"][ev]

    daughters_of_photon = []
    
    for i, (pdgId, momIdx) in enumerate(zip(pdgIds, motherIdx)):
        if momIdx < 0 or momIdx >= len(pdgIds):
            continue  # skip if mother index is invalid

        mother_pdgId = pdgIds[momIdx]

        if mother_pdgId == 22:  # photon
            # Now get the grandmother index
            grandmaIdx = motherIdx[momIdx]
            if grandmaIdx < 0 or grandmaIdx >= len(pdgIds):
                continue  # invalid grandmother

            grandmother_pdgId = pdgIds[grandmaIdx]

            if abs(grandmother_pdgId) == 5:  # b-quark
                daughters_of_photon.append((i, pdgId))  # store index and pdgId

    if daughters_of_photon:
        print(f"Event {ev}:")
        for idx, pdgId in daughters_of_photon:
            print(f"  Particle index {idx} → PDG ID: {pdgId}")


# %%
