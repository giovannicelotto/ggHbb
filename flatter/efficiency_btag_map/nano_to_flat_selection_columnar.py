# From nanoAOD
# Take events with 
# At leaast one jet with a triggering muon
# And either another jet with triggering muon
# Or another jet with Loose WP DeepJet Btag

# All jets need to be "good Jets" PuID and JetId requirements

# twoTrigJets mask events with two jets with triggering muon
# final_mask mask events with only one jet with triggering muon and one other good jet with at least Loose WP

# Very small differences can arise if in run_flatter the jet_pt is required to be > 20 instead of >=20
# %%
import uproot
import awkward as ak
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
# Step 1: Define the mask for good jets
goodJets = (
    (Jet_jetId == 6) &
    ((Jet_pt > 50) | (Jet_puId >= 4))
)

# Step 2: Apply this mask to get the indices of muons in good jets
# mu1_idx and mu2_idx are the leading and subleading muons inside good jets
mu1_idx = Jet_muonIdx1[goodJets]
mu2_idx = Jet_muonIdx2[goodJets]

# Step 3: Find whether each muon (leading and subleading) is triggering
# Mask out invalid muon indices (-1 means no muon in that slot)
mu1_idx_clean = ak.mask(mu1_idx, mu1_idx != -1)
mu2_idx_clean = ak.mask(mu2_idx, mu2_idx != -1)

# Index into Muon_isTriggering where valid, fill False elsewhere
mu1_isTrig = ak.fill_none(Muon_isTriggering[mu1_idx_clean], False)
mu2_isTrig = ak.fill_none(Muon_isTriggering[mu2_idx_clean], False)


# Step 4: Check if at least one of the two muons in the jet is triggering
jet_has_triggering_muon = mu1_isTrig | mu2_isTrig

# Step 5: Count how many jets per event have at least one triggering muon
numJets_with_trigMuon = ak.sum(jet_has_triggering_muon, axis=1)

# Step 6: Apply your condition for two or more such jets
twoTrigJets = numJets_with_trigMuon >= 2
print("Events with 2 TrigJets are %d"%ak.sum(twoTrigJets))
# %%
import awkward as ak

# Step 1: Good jets
goodJets = (
    (Jet_jetId == 6) &
    ((Jet_pt > 50) | (Jet_puId >= 4))
)

# Step 2: Apply this mask to get the indices of muons in good jets.
# mu1_idx and mu2_idx are the leading and subleading muons inside good jets
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
has_exactly_one_triggering_jet = (ak.sum(triggering_jet_indices>=0, axis=1)==1)

# Step 8: Get that one triggering jet index per event
only_trig_jet_idx = ak.firsts(ak.drop_none(triggering_jet_indices))

# Step 9: b-tag mask on good jets
btag_mask = Jet_btagDeepFlavB >= 0.049

# Step 10: Index of b-tagged jets (among goodJets)
btagged_jet_idx = ak.local_index(Jet_pt)[(btag_mask) & (goodJets)]

# Step 11: Check for b-tagged jets ≠ triggering jet
# btagged index refer to good jets
btag_is_other = btagged_jet_idx != only_trig_jet_idx[:, None]

# Step 12: Events with another b-tagged jet
has_other_btagged_jet = ak.any(btag_is_other, axis=1)

# Step 13: Final selection
final_mask = has_exactly_one_triggering_jet & has_other_btagged_jet
n_events = ak.sum(final_mask)

print(f"{n_events} Eventi con esattamente un jet triggering e almeno un altro b-tagged: ")
print("Total number of events is %d"%ak.sum(final_mask | twoTrigJets))
total_mask = final_mask | twoTrigJets

# %%
