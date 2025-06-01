# %%
import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import ROOT
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/250409_155207/0000/others/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_45.root"
file = uproot.open(path)
tree = file['Events']
branches = tree.arrays()

# %%
for ev in range(tree.num_entries):
    GenPart_pt = branches["GenPart_pt"][ev]
    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
    GenPart_pdgId = branches["GenPart_pdgId"][ev]
    GenPart_statusFlags = branches["GenPart_statusFlags"][ev]
    GenPart_eta = branches["GenPart_eta"][ev]
    GenPart_phi = branches["GenPart_phi"][ev]
    GenJetNu_pt = branches["GenJetNu_pt"][ev]
    GenJetNu_mass = branches["GenJetNu_mass"][ev]
    GenJetNu_eta = branches["GenJetNu_eta"][ev]
    GenJetNu_phi = branches["GenJetNu_phi"][ev]
    nGenJetNu = branches["nGenJetNu"][ev]
    Jet_eta = branches["Jet_eta"][ev]
    Jet_phi = branches["Jet_phi"][ev]
    nJet = branches["nJet"][ev]
    Muon_isTriggering = branches["Muon_isTriggering"][ev]
    nMuon = branches["nMuon"][ev]
    Jet_nMuons = branches["Jet_nMuons"][ev]
    Jet_muonIdx1 = branches["Jet_muonIdx1"][ev]
    Jet_muonIdx2 = branches["Jet_muonIdx2"][ev]
    Jet_btagDeepFlavB = branches["Jet_btagDeepFlavB"][ev]
    Jet_jetId = branches["Jet_jetId"][ev]
    Jet_pt = branches["Jet_pt"][ev]
    Jet_puId = branches["Jet_puId"][ev]
    Jet_genJetIdx = branches["Jet_genJetIdx"][ev]
    GenJet_partonMotherPdgId = branches["GenJet_partonMotherPdgId"][ev]
    GenPart_mass = branches["GenPart_mass"][ev]





    
    # Look at 2 jets events
    if nJet!=2:
        continue
    # Look at Higgs > 100 GeV
    h = ROOT.TLorentzVector(0.,0.,0.,0.)
    b1 = ROOT.TLorentzVector(0.,0.,0.,0.)
    b2 = ROOT.TLorentzVector(0.,0.,0.,0.)
    higgs_pt = GenPart_pt[(GenPart_pdgId==25) & (GenPart_statusFlags>8192)]
    if higgs_pt<100:
        continue
    mb1 = (GenPart_pdgId==5) & ((GenPart_genPartIdxMother >= 0)) & (GenPart_pdgId[GenPart_genPartIdxMother]==25)
    mb2 = (GenPart_pdgId==-5) & ((GenPart_genPartIdxMother >= 0)) & (GenPart_pdgId[GenPart_genPartIdxMother]==25)
    b1.SetPtEtaPhiM(GenPart_pt[mb1][0], GenPart_eta[mb1][0], GenPart_phi[mb1][0], GenPart_mass[mb1][0])
    b2.SetPtEtaPhiM(GenPart_pt[mb2][0], GenPart_eta[mb2][0], GenPart_phi[mb2][0], GenPart_mass[mb2][0])


    #plt.figure(figsize=(5, 5))
    #plt.quiver(0, 0, b1.Px(), b1.Py(), angles='xy', scale_units='xy', scale=1, color='r', label='b1')
    #plt.quiver(0, 0, b2.Px(), b2.Py(), angles='xy', scale_units='xy', scale=1, color='b', label='b2')
    #plt.xlim(-200, 200)
    #plt.ylim(-200, 200)
    #plt.xlabel('x (GeV)')
    #plt.ylabel('y (GeV)')
    #plt.title(f'Event {ev}: Transverse Plane (b-quarks from Higgs)')
    #plt.grid(True)
    #plt.axhline(0, color='gray', linestyle='--')
    #plt.axvline(0, color='gray', linestyle='--')
    #plt.legend()
    #plt.gca().set_aspect('equal')
    #plt.show()


    plt.figure(figsize=(5, 5))
    plt.scatter(b1.Eta(), b1.Phi(),  color='r', label='b1')
    plt.scatter(b2.Eta(), b2.Phi(),  color='b', label='b2')
    plt.xlim(-5, 5)
    plt.ylim(-3.15, 3.15)
    plt.xlabel('Eta')
    plt.ylabel('Phi')
    plt.title(f'Event {ev}: Transverse Plane (b-quarks from Higgs)')
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()
    input("\n\nNext\n\n")
# %%
