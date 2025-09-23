# %%
import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MCSamples2025Aug15/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/250815_134708/0000/training/MCSamples_Run2_mc_2025Aug15_10.root"
file = uproot.open(path)
tree = file['Events']
branches = tree.arrays()

# %%
#for ev in range(tree.num_entries):
#    GenPart_pt = branches["GenPart_pt"][ev]
#    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
#    GenPart_pdgId = branches["GenPart_pdgId"][ev]
#    GenPart_statusFlags = branches["GenPart_statusFlags"][ev]
#
#
#    m = (GenPart_genPartIdxMother>-1) & (GenPart_pdgId[GenPart_genPartIdxMother]==25) & (GenPart_statusFlags[GenPart_genPartIdxMother]>=8192)
#    print(np.sum(m), flush=True)
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

Muon_isTriggering = branches["Muon_isTriggering"]
nMuon = branches["nMuon"]
Jet_nMuons = branches["Jet_nMuons"]
Jet_muonIdx1 = branches["Jet_muonIdx1"]
Jet_muonIdx2 = branches["Jet_muonIdx2"]
Jet_btagDeepFlavB = branches["Jet_btagDeepFlavB"]
Jet_jetId = branches["Jet_jetId"]
Jet_pt = branches["Jet_pt"]
Jet_puId = branches["Jet_puId"]
# %%
m = (   (GenPart_genPartIdxMother>-1) &                         # Particles With Mother
        (GenPart_pdgId[GenPart_genPartIdxMother]==25) &         # Particles daughters of Higgs
        (GenPart_statusFlags[GenPart_genPartIdxMother]>=8192))  # Count only Higgs last copy (to avoid H->H)
m_eta = (m) #& (abs(GenPart_eta)<2.5)                               # Eta acceptance
m_pt = (m_eta) & (GenPart_pt>0)                                     # pT acceptance
mLast = (m_pt) & (abs(GenPart_pdgId)==5)                             # Flavor (redundant)

# %%
print("Every Event has 2 Higgs daughters : ", ak.sum(ak.sum(m, axis=1)==2)==tree.num_entries)
print("Eta acceptance : ", ak.sum(ak.sum(m_eta, axis=1)==2)*100/tree.num_entries)
print("pT acceptance : ", ak.sum(ak.sum(m_pt, axis=1)==2)*100/tree.num_entries)
print("Flavor==5 : ", ak.sum(ak.sum(mLast, axis=1)==2)*100/tree.num_entries)

# Events with 2 GenParticles
mEvent = ((ak.sum(mLast, axis=1)==2) &
            (nJet>=2) &
            (ak.sum((Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4)), axis=1)>=2) & 
            (nMuon>=1))
print("Mask per Event ", ak.sum(mEvent)*100/tree.num_entries)


#fig, ax = plt.subplots(1, 1)
#ax.hist(GenPart_eta[mLast][~mEvent][:,0],bins=np.linspace(-5, 5, 31), label='Quark 1', histtype='step')
#ax.hist(GenPart_eta[mLast][~mEvent][:,1],bins=np.linspace(-5, 5, 31), label='Quark 2', histtype='step')
#ax.legend()








muonIdxs = ak.local_index(ak.Array([range(n) for n in nMuon]))
triggeringMuonsIdx = muonIdxs[Muon_isTriggering==1]
nTriggeringMuonPerEvent = ak.Array([len(triggeringMuonsIdx[n]) for n in range(len(triggeringMuonsIdx))])
mEvent = (mEvent) & (nTriggeringMuonPerEvent>=1)
print("Mask per Event ", ak.sum(mEvent)*100/tree.num_entries)
# mEvent cuts 2reco jets one triggering Muon in the event
# %%
# Index of GenJetNu that minimize the dR with the first Higgs Daughter
dEta_1 = GenPart_eta[mLast][mEvent][:, 0] - GenJetNu_eta[mEvent]
dPhi_1 = (GenPart_phi[mLast][mEvent][:, 0] - GenJetNu_phi[mEvent] + np.pi) % (2 * np.pi) - np.pi
deltaR_gen1 = np.sqrt(dEta_1**2 + dPhi_1**2)
gen_true1 = ak.argmin(deltaR_gen1, axis=1)
# reshape the genTrue to make them match the shape of GenJetNu
gen_true1 = gen_true1[:, None]
# %%
mGenJetNu_ambiguity = ak.local_index(ak.Array([range(n) for n in nGenJetNu[mEvent]]))
mGenJetNu_ambiguity= mGenJetNu_ambiguity==gen_true1
dEta_2 = (GenPart_eta[mLast][mEvent][:,1] - GenJetNu_eta[mEvent])
dPhi_2 = (GenPart_phi[mLast][mEvent][:,1] - GenJetNu_phi[mEvent])
deltaR_gen2 = np.sqrt(dEta_2**2 + dPhi_2**2)  
gen_true2 = ak.argmin( deltaR_gen2 +   mGenJetNu_ambiguity*1000, axis=1)
gen_true2 = gen_true2[:, None]



# %%
#deltaR_gen1 = ak.where((gen_true1) & (deltaR_gen1<0.4), deltaR_gen1, -1)
#deltaR_gen2[gen_true2] = ak.where(deltaR_gen2[gen_true2]<0.4, deltaR_gen2[gen_true2], -1)

# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 51)
ax.hist(np.clip(ak.flatten(deltaR_gen1[gen_true1]), bins[0], bins[-1]- 1e-6  ), bins=bins, weights = np.ones_like(ak.flatten(deltaR_gen1[gen_true1]))/len(deltaR_gen1[gen_true1]), histtype='step', label='Leading Daughter')
ax.hist(np.clip(ak.flatten(deltaR_gen2[gen_true2]), bins[0], bins[-1]- 1e-6  ), bins=bins, weights = np.ones_like(ak.flatten(deltaR_gen2[gen_true2]))/len(deltaR_gen2[gen_true2]), histtype='step', label='Subleading Daughter')

ax.text(x=0.95, y=0.6, s="%.1f%% leading within dR < 0.4"%(ak.sum(deltaR_gen1[gen_true1]<0.4)/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.55, s="%.1f%% subleading within dR < 0.4"%(ak.sum(deltaR_gen2[gen_true2]<0.4)/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.5, s="%.1f%% both within dR < 0.4"%(ak.sum((deltaR_gen2[gen_true2]<0.4) & (deltaR_gen1[gen_true1]<0.4))/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.7, s="NJets >= 2\nTrigMuon>=1", transform=ax.transAxes, ha='right')
ax.set_xlabel("dR(GenPart, GenJetNu)")
ax.set_ylabel("Probability")
ax.legend()
#ax.set_yscale('log')

# %%
# gen_true1 is to be applied to GenJet[mEvent] collection
# gen_true2 is to be applied to GenJet[mEvent]
print("No ambiguity : ", np.sum(gen_true1==gen_true2)==0)
if np.sum(gen_true1==gen_true2)==0:
    print("genJetNu1 and genJetNu2 are always different per event")

# %%












# ***********************************************************
# *                                                         *
# *             reco                                        *
# ***********************************************************
# %%
dEta_reco_1 = (GenJetNu_eta[mEvent][gen_true1][:,0] - Jet_eta[mEvent])
dPhi_reco_1 = (GenJetNu_phi[mEvent][gen_true1][:,0] - Jet_phi[mEvent] + np.pi) % (2 * np.pi) - np.pi
dR_reco_1 = np.sqrt(dEta_reco_1**2+ dPhi_reco_1**2)
# Note to remove the cases where matching is done to genJetNu that are not matched to genPart<0.4
reco_true1 = ak.argmin(dR_reco_1 , axis=1)
mJet_ambiguity = ak.local_index(ak.Array([range(n) for n in nJet[mEvent]]))
mJet_ambiguity= mJet_ambiguity==reco_true1
# %%
dEta_reco_2 = (GenJetNu_eta[mEvent][gen_true2][:,0] - Jet_eta[mEvent]) 
dPhi_reco_2 = (GenJetNu_phi[mEvent][gen_true2][:,0] - Jet_phi[mEvent]+ np.pi) % (2 * np.pi) - np.pi 
dR_reco_2 = np.sqrt(dEta_reco_2**2 + dPhi_reco_2**2)
reco_true2 = ak.argmin(   dR_reco_2 +    mJet_ambiguity*1000, axis=1)


# %%
# reshape the indices to make them match the shape of recoJets
reco_true1 = reco_true1[:, None]
reco_true2 = reco_true2[:, None]
# %%
fig, ax = plt.subplots(1, 1)
bins_new=np.linspace(0, 1, 11)
mask_genPart_genJet = (deltaR_gen1[gen_true1]<0.4) & (deltaR_gen2[gen_true2]<0.4)
ax.hist(np.clip(ak.flatten(dR_reco_1[reco_true1][mask_genPart_genJet]), bins[0], bins[-1]-1e-6), bins=bins, weights = np.ones_like(ak.flatten(dR_reco_1[reco_true1][mask_genPart_genJet]))/len(ak.flatten(dR_reco_1[reco_true1][mask_genPart_genJet])), histtype='step', label='Leading GenJetNu')
ax.hist(np.clip(ak.flatten(dR_reco_2[reco_true2][mask_genPart_genJet]), bins[0], bins[-1]-1e-6), bins=bins, weights = np.ones_like(ak.flatten(dR_reco_2[reco_true2][mask_genPart_genJet]))/len(ak.flatten(dR_reco_2[reco_true2][mask_genPart_genJet])), histtype='step', label='Subleading GenJetNu')

ax.text(x=0.95, y=0.6, s="%.1f%% leading within dR < 0.4"%(ak.sum(dR_reco_1[reco_true1]<0.4)/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.55, s="%.1f%% subleading within dR < 0.4"%(ak.sum(dR_reco_2[reco_true2]<0.4)/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.5, s="%.1f%% both within dR < 0.4"%(ak.sum((dR_reco_2[reco_true2]<0.4) & (dR_reco_1[reco_true1]<0.4))/tree.num_entries*100), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.7, s="NJets >= 2\nTrigMuon>=1", transform=ax.transAxes, ha='right')
ax.set_xlabel("dR(recoJet, GenJetNu_linked_toGenPart)")
ax.set_ylabel("Probability")
ax.legend()
ax.set_yscale('log')
# %%
print("No ambiguity : ", np.sum(reco_true1==reco_true2)==0)
if np.sum(reco_true1==reco_true2)==0:
    print("reco_Jet_1 (reco_Jet genMatched to genJetNu1) and reco_Jet_2 (reco_Jet genMatched to genJetNu1) are always different per event")
else:
    print(np.sum(reco_true1==reco_true2), " ambiguities")
# %%
dRJets = ak.concatenate([dR_reco_1[reco_true1], dR_reco_2[reco_true2]], axis=1)
trueJets = ak.concatenate([reco_true1, reco_true2], axis=1)
trueJetsOnlyGood = ak.where(dRJets<0.4, trueJets, -1)

# %%
# Index of leading Muon for every jet:
#   Jet_muonIdx1[mEvent]
# List of Triggering Muon

# %%
# Jets with leading Muon as leading Triggering Muon inside
mLL = Jet_muonIdx1[mEvent] == triggeringMuonsIdx[mEvent,0]
# Jets with subleading Muon as leading Triggering Muon inside
mSL = Jet_muonIdx2[mEvent] == triggeringMuonsIdx[mEvent,0]
# Single Jet per Event with leading or subleading Muon inside as the leading triggering Muon
# For each line only one True at most ( same muon cannot be shared among jets)
Jets_withLeadingTriggering = ((mLL) | (mSL))
assert ak.max(ak.sum(Jets_withLeadingTriggering, axis=1))==1, "Leading Muon shared among jets"

# %%
# Jets with leading Muon as Subleading Triggering Muon inside
mLS = Jet_muonIdx1[nTriggeringMuonPerEvent>=2] == triggeringMuonsIdx[nTriggeringMuonPerEvent>=2][:,1]
# Jets with subleading Muon as Subleading Triggering Muon inside
mSS = Jet_muonIdx2[nTriggeringMuonPerEvent>=2] == triggeringMuonsIdx[nTriggeringMuonPerEvent>=2][:,1]
# Single Jet per Event with leading or subleading Muon inside as the SubLeading triggering Muon
Jets_withSubLeadingTriggering = ((mLS) | (mSS))
assert ak.max(ak.sum(Jets_withSubLeadingTriggering, axis=1))==1, "Leading Muon shared among jets"
# %%

# Events where one jet has within the first two leading muons one triggering muon
#Jets_withLeadingTriggering
nJet_whenOneTriggering = ak.local_index(ak.Array([range(n) for n in nJet[mEvent]]))
selected1 = nJet_whenOneTriggering[Jets_withLeadingTriggering]
selected1 = ak.where(ak.sum(Jets_withLeadingTriggering, axis=1)>0, selected1, [[-999]])
# %%
# One Jet always chosen:
# mask_genPart_genJet
effJet1 = ((selected1 == trueJetsOnlyGood[:,0]) | (selected1 == trueJetsOnlyGood[:,1]) ) & (mask_genPart_genJet)
matched1 = ak.sum(effJet1)
print("Jet with Leading triggering ", matched1/tree.num_entries)
print("N(trigJet is genMatched) / N(2 jets from  Higgs) ", matched1/ak.sum(mask_genPart_genJet))

# %%
tightJets = (Jet_btagDeepFlavB[mEvent]>0.71)
mediumJets = (Jet_btagDeepFlavB[mEvent]>0.2783)
looseJets = (Jet_btagDeepFlavB[mEvent]>0.0490)


# If you manage to have:
# In case of second muon take the trig muon otherwise take bscore


scoreBTag = (tightJets*100 + mediumJets*10 + looseJets - 1000*Jets_withLeadingTriggering) 
selected2 = ak.argmax(scoreBTag, axis=1)
selected2 = ak.where(ak.sum(Jets_withLeadingTriggering, axis=1)>0, selected2, [[-999]])
effJet2 = ((selected2 == trueJetsOnlyGood[:,0]) | (selected2 == trueJetsOnlyGood[:,1])) & (mask_genPart_genJet)
matched2 = ak.sum(effJet2)
print("Jet with B score WP ", matched2/tree.num_entries)
print("N(selJet is genMatched) / N(2 jets from  Higgs) ", matched2/ak.sum(mask_genPart_genJet))
eff_tot = ak.sum((effJet1) & (effJet2))
print("Overall matching Efficiency : ", eff_tot/tree.num_entries )
print("Overall matching Efficiency/N(2 jets from higgs) : ", eff_tot/ak.sum(mask_genPart_genJet) )

# %%
# Compare with leading and subleading
effJet1 = ((0 == trueJetsOnlyGood[:,0]) | (0 == trueJetsOnlyGood[:,1])) & (mask_genPart_genJet)
matched1 = ak.sum(effJet1)
print("Leading Jet ", matched1/tree.num_entries)
effJet2 = ((1 == trueJetsOnlyGood[:,0]) | (1 == trueJetsOnlyGood[:,1])) & (mask_genPart_genJet)
matched2 = ak.sum(effJet2)
print("Subleading Jet ", matched2/tree.num_entries)
eff_tot = ak.sum((effJet1) & (effJet2))
print("Overall matching Efficiency : ", eff_tot/tree.num_entries )
# %%



pt1 = GenJetNu_pt[mEvent][gen_true1]
eta1 = GenJetNu_eta[mEvent][gen_true1]
phi1 = GenJetNu_phi[mEvent][gen_true1]
mass1 = GenJetNu_mass[mEvent][gen_true1]

pt2 = GenJetNu_pt[mEvent][gen_true2]
eta2 = GenJetNu_eta[mEvent][gen_true2]
phi2 = GenJetNu_phi[mEvent][gen_true2]
mass2 = GenJetNu_mass[mEvent][gen_true2]

# Compute Cartesian components
px1 = pt1 * np.cos(phi1)
py1 = pt1 * np.sin(phi1)
pz1 = pt1 * np.sinh(eta1)
E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)

px2 = pt2 * np.cos(phi2)
py2 = pt2 * np.sin(phi2)
pz2 = pt2 * np.sinh(eta2)
E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)

# Compute dijet invariant mass
Mjj = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)

# %%
import pandas as pd
df = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training/GluGluHToBB_tr_10.parquet", 
                     columns=None)



fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 150, 51)
ax.hist(np.clip(pt2, bins[0], bins[-1]), bins=bins)
ax.hist(np.clip(df.jet2_pt_uncor, bins[0], bins[-1]), bins=bins, histtype='step')
ax.hist(np.clip(df.jet2_pt, bins[0], bins[-1]), bins=bins, histtype='step')
# %%
