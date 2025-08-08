# From nanoAOD
# Take events with a muon inside a jet

# Take Jets pT, eta, btag, hadronFlavour, puID, JetID, 


# %%
import uproot
import awkward as ak
from computeEfficiency import compute_efficiency, plotEfficiencyMaps
import matplotlib.pyplot as plt
import numpy as np
# %%
file = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/250409_155207/0000/training/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_6.root")
tree = file["Events"]
nEvents = tree.num_entries
branches=tree.arrays()
Muon_pt = branches["Muon_pt"]
Muon_eta = branches["Muon_eta"]
Muon_isTriggering = branches["Muon_isTriggering"]
Jet_muonIdx1 = branches["Jet_muonIdx1"]
Jet_muonIdx2 = branches["Jet_muonIdx2"]
Jet_jetId = branches["Jet_jetId"]
Jet_pt = branches["Jet_pt"]
Jet_puId = branches["Jet_puId"]
Jet_nConstituents = branches["Jet_nConstituents"]
Jet_btagDeepFlavB = branches["Jet_btagDeepFlavB"]
Jet_eta = branches["Jet_eta"]
Jet_hadronFlavour = branches["Jet_hadronFlavour"]
Jet_mass = branches["Jet_mass"]
nMuon = branches["nMuon"]
nJet = branches["nJet"]
nElectron = branches["nElectron"]
nSV = branches["nSV"]

# %%


# Step 1: Good jets
goodJets = (
    (Jet_jetId == 6) &
    ((Jet_pt > 50) | (Jet_puId >= 4))
)

# Step 2: Apply this mask to get the indices of muons in good jets
mu1_idx = Jet_muonIdx1[goodJets]
mu2_idx = Jet_muonIdx2[goodJets]

# Step 3: Find whether each muon (leading and subleading) is triggering
# Mask out invalid muon indices (-1 means no muon in that slot)
mu1_idx_clean = ak.mask(mu1_idx, mu1_idx != -1)
mu2_idx_clean = ak.mask(mu2_idx, mu2_idx != -1)

# Index into Muon_isTriggering where valid, fill False elsewhere
mu1_isTrig = ak.fill_none(Muon_isTriggering[mu1_idx_clean], False)
mu2_isTrig = ak.fill_none(Muon_isTriggering[mu2_idx_clean], False)


# Step 4: Jet has at least one triggering muon
has_trig_muon = mu1_isTrig | mu2_isTrig  # ← mantiene struttura per evento sui good jets

# Step 5: Index of good jets
goodJets_idx = ak.local_index(Jet_pt)[goodJets]

# Step 6: Indices of jets with at least one triggering muon
triggering_jet_indices = ak.mask(goodJets_idx, has_trig_muon != 0)

# Step 7: Events with exactly one such jet
has_exactly_one_triggering_jet = (ak.sum(triggering_jet_indices>=0, axis=1)>=1)


# Count how many good jets per event
nGoodJets = ak.count(goodJets_idx, axis=1)


final_mask = (has_exactly_one_triggering_jet) & (nGoodJets>=2)
n_events = ak.sum(final_mask)

print(f"{n_events} Events with at least one triggering jet and at least one more good jet: ")
print("Total number of events is %d"%ak.sum(final_mask))
# %%
BTAG_WP = 0.71  
pt_bins = np.array([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000])
eta_bins = np.array([0, 1.5, 2.5])  # Or choose [0, 2.5] for one bin


eff_map = compute_efficiency(
    jet_pt = Jet_pt[goodJets][final_mask],
    jet_eta = Jet_eta[goodJets][final_mask],
    jet_btag = Jet_btagDeepFlavB[goodJets][final_mask],
    jet_flav = Jet_hadronFlavour[goodJets][final_mask],
    pt_bins = pt_bins,
    eta_bins = eta_bins,
    WP = BTAG_WP
)
plotEfficiencyMaps(pt_bins=pt_bins, eta_bins=eta_bins, eff_map=eff_map)