# %%
import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/250409_155207/0000/others/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_45.root"
file = uproot.open(path)
tree = file['Events']
branches = tree.arrays()

# %%
GenPart_pt = branches["GenPart_pt"]
GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"]
GenPart_pdgId = branches["GenPart_pdgId"]
GenPart_statusFlags = branches["GenPart_statusFlags"]
GenPart_eta = branches["GenPart_eta"]
GenPart_phi = branches["GenPart_phi"]

GenJetNu_pt = branches["GenJetNu_pt"]
GenJetNu_mass = branches["GenJetNu_mass"]
GenJetNu_eta = branches["GenJetNu_eta"]
GenJetNu_phi = branches["GenJetNu_phi"]
nGenJetNu = branches["nGenJetNu"]

Jet_eta = branches["Jet_eta"]
Jet_phi = branches["Jet_phi"]
nJet = branches["nJet"]
nJets = branches["nJet"]

Muon_isTriggering = branches["Muon_isTriggering"]
nMuon = branches["nMuon"]
Jet_nMuons = branches["Jet_nMuons"]
Jet_muonIdx1 = branches["Jet_muonIdx1"]
Jet_muonIdx2 = branches["Jet_muonIdx2"]
Jet_btagDeepFlavB = branches["Jet_btagDeepFlavB"]
Jet_jetId = branches["Jet_jetId"]
Jet_pt = branches["Jet_pt"]
Jet_puId = branches["Jet_puId"]
Jet_genJetIdx = branches["Jet_genJetIdx"]
GenJet_partonMotherPdgId = branches["GenJet_partonMotherPdgId"]
# %%
m = (   (GenPart_genPartIdxMother>-1) &                         # Particles With Mother
        (GenPart_pdgId[GenPart_genPartIdxMother]==25) &         # Particles daughters of Higgs
        (GenPart_statusFlags[GenPart_genPartIdxMother]>=8192))  # Count only Higgs last copy (to avoid H->H)
m1 = (m) & (abs(GenPart_eta)<2.5)                               # Eta acceptance
m2 = (m1) & (GenPart_pt>0)                                     # pT acceptance
mLast = (m2) & (abs(GenPart_pdgId)==5)                             # Flavor (redundant)

# %%
print("Every Event has 2 Higgs daughters : ", ak.sum(ak.sum(m, axis=1)==2)==tree.num_entries)
print("Eta acceptance : ", ak.sum(ak.sum(m1, axis=1)==2)*100/tree.num_entries)
print("pT acceptance : ", ak.sum(ak.sum(m2, axis=1)==2)*100/tree.num_entries)
print("Flavor==5 : ", ak.sum(ak.sum(mLast, axis=1)==2)*100/tree.num_entries)

# Events with 2 GenParticles
mEvent = ((ak.sum(mLast, axis=1)==2) &
            (nJet>=2) &
            (ak.sum((Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4)), axis=1)>=2) & 
            (nMuon>=1))
muonIdxs = ak.local_index(ak.Array([range(n) for n in nMuon]))
triggeringMuonsIdx = muonIdxs[Muon_isTriggering==1]
nTriggeringMuonPerEvent = ak.Array([len(triggeringMuonsIdx[n]) for n in range(len(triggeringMuonsIdx))])
mEvent = (mEvent) & (nTriggeringMuonPerEvent>=1)
# mEvent cuts 2reco jets one triggering Muon in the event
# %%

#Jet_genJetIdx > -1
m_gen = ak.sum(GenJet_partonMotherPdgId[Jet_genJetIdx[Jet_genJetIdx>-1]]==25,axis=1)==2
GenPart_eta[(~m_gen) & (mLast) & (ak.sum(mLast, axis=1)==2)]
# %%
Jet_hadronFlavour=branches["Jet_hadronFlavour"]
# %%
ak.sum(ak.sum(Jet_hadronFlavour==5, axis=1)==2)
# %%
mBoostedHiggs = GenPart_pt[(GenPart_pdgId==25) & (GenPart_statusFlags>=8192)]>450
ak.sum(ak.sum(Jet_hadronFlavour==5, axis=1)[ak.flatten(mBoostedHiggs)]==2)
# %%
# Mask to pass from GenPart to two b quarks from higgs
m_gen_bb = (abs(GenPart_pdgId)==5) & ((GenPart_genPartIdxMother >= 0)) & (GenPart_pdgId[GenPart_genPartIdxMother]==25)
ak.sum(ak.sum(m_gen_bb, axis=1)==2)==tree.num_entries
# %%
# Mask to pass from GenPart to higgs only
m_gen_higgs = (GenPart_pdgId==25) & (GenPart_statusFlags>=8192)
ak.sum(ak.sum(m_gen_higgs, axis=1)==1)==tree.num_entries
# %%
GenPart_eta[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)]
GenPart_phi[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)]

# %%
import pandas as pd
df = pd.DataFrame({
    'eta1' : GenPart_eta[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)][:,0],
    'eta2' : GenPart_eta[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)][:,1],
    'phi1' : GenPart_eta[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)][:,0],
    'phi2' : GenPart_eta[(m_gen_bb)][ak.flatten(GenPart_pt[m_gen_higgs]>0)][:,1]
})
# %%
df['dR'] = np.sqrt((df.eta1 - df.eta2)**2 + (df.phi1-df.phi2)**2)
fig, ax = plt.subplots(1,1)
ax.hist(df.dR, bins=np.linspace(0, 4, 51))
# %%
m_2jets = (nJet==2) & ak.flatten(GenPart_pt[m_gen_higgs]>10)
plt.hist(GenPart_phi[(m_gen_bb)][m_2jets][:,0] - GenPart_phi[(m_gen_bb)][m_2jets][:,1], bins=100)
#plt.hist(GenPart_phi[(m_gen_bb)][m_2jets][:,0] - GenPart_phi[(m_gen_bb)][m_2jets][:,1], bins=100)
# %%

# %%
