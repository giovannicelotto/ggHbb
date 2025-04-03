# %%
#
# Add dR of matching steps and apply a cut at flat level
#
#
import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re
import pandas as pd
import ROOT
import random
sys.path.append("/t3home/gcelotto/ggHbb/flatter/")
from treeFlatter import jetsSelector
# %%
'''
Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                 = sys.argv[2]   (z for Z boson, h for Higgs boson)
'''


# Now open the file and use the previous distribution
def main(nanoFileName, fileNumber, process):


    fileData=[]        # to store mjj for the matched signals
    outFolder = "/scratch"
    
    f = uproot.open(nanoFileName)
    tree = f['Events']
    print("\nEntries : %d"%(tree.num_entries))
    branches = tree.arrays()
    maxEntries = tree.num_entries

    for ev in  range(maxEntries):
        features_ = []
    #Gen Part
        nGenPart = branches["nGenPart"][ev]
        GenPart_pt = branches["GenPart_pt"][ev]
        GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
        GenPart_pdgId = branches["GenPart_pdgId"][ev]
        GenPart_statusFlags = branches["GenPart_statusFlags"][ev]
        GenPart_eta = branches["GenPart_eta"][ev]
        GenPart_phi = branches["GenPart_phi"][ev]
    # Reco Jets
        nJet                        = branches["nJet"][ev]
        Jet_eta                     = branches["Jet_eta"][ev]
        Jet_pt                      = branches["Jet_pt"][ev]
        Jet_phi                     = branches["Jet_phi"][ev]
        Jet_mass                    = branches["Jet_mass"][ev]
        Jet_bReg2018                = branches["Jet_bReg2018"][ev]
        Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
        Jet_rawFactor               = branches["Jet_rawFactor"][ev]
        Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
        Jet_tagUParTAK4B            = branches["Jet_tagUParTAK4B"][ev]
        Jet_puId                    = branches["Jet_puId"][ev]
        Jet_jetId                   = branches["Jet_jetId"][ev]

    #Trig Muon
        Muon_isTriggering = branches["Muon_isTriggering"][ev]
        nMuon = branches["nMuon"][ev]
    # Selection of Jets
        Jet_nMuons = branches["Jet_nMuons"][ev]
        Jet_muonIdx1 = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2 = branches["Jet_muonIdx2"][ev]
        Jet_jetId = branches["Jet_jetId"][ev]
        Jet_pt = branches["Jet_pt"][ev]
        Jet_puId = branches["Jet_puId"][ev]

    #GenJetNu
        nGenJetNu                   = branches["nGenJetNu"][ev]
        GenJetNu_pt = branches["GenJetNu_pt"][ev]
        GenJetNu_mass = branches["GenJetNu_mass"][ev]
        GenJetNu_eta = branches["GenJetNu_eta"][ev]
        GenJetNu_phi = branches["GenJetNu_phi"][ev]
        nGenJetNu = branches["nGenJetNu"][ev]
    # Other 
        Muon_pt = branches["Muon_pt"][ev]
        Muon_eta = branches["Muon_eta"][ev]
        Muon_dxy = branches["Muon_dxy"][ev]
        Muon_dxyErr = branches["Muon_dxyErr"][ev]
        

        maskJets = (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4))
        if len(np.arange(nJet)[maskJets])<2:
            continue
        # Select Jets
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, nJet, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=1, Jet_pt=Jet_pt)

        if selected1==999:
            continue
        if selected2==999:
            assert False
        
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)

        jet1.SetPtEtaPhiM(Jet_pt[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        features_.append(np.float32(jet1.Pt()))
        features_.append(np.float32(jet1.Eta()))
        features_.append(np.float32(jet1.Phi()))
        features_.append(np.float32(jet1.M()))
        features_.append(np.float32(Jet_btagDeepFlavB[selected1]))
        features_.append(int(Jet_btagDeepFlavB[selected1]>0.71))
        features_.append(np.float32(jet1.DeltaR(dijet)))
        features_.append(int(Jet_nMuons[selected1]))
        
        #features_.append(int(Jet_btagDeepFlavB[selected1]>0.71))

        features_.append(np.float32(jet2.Pt()))
        features_.append(np.float32(jet2.Eta()))
        features_.append(np.float32(jet2.Phi()))
        features_.append(np.float32(jet2.M()))
        features_.append(np.float32(Jet_btagDeepFlavB[selected2]))
        features_.append(int(Jet_btagDeepFlavB[selected2]>0.71))
        features_.append(np.float32(jet2.DeltaR(dijet)))
        features_.append(int(Jet_nMuons[selected2]))

        assert Jet_jetId[selected1]==6
        assert Jet_jetId[selected2]==6

        selected3=-1
        for i in np.arange(nJet)[maskJets]:
            if ((i ==selected1) | (i==selected2)):
                continue
            else:
                jet3.SetPtEtaPhiM(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i])
                selected3=i
                assert Jet_jetId[i]==6
                break
        features_.append(np.float32(jet3.Pt()))
        features_.append(np.float32(jet3.Eta()))
        features_.append(np.float32(jet3.Phi()))
        features_.append(np.float32(jet3.M()))
        features_.append(np.float32(jet3.DeltaR(dijet)))
        features_.append(int(Jet_nMuons[selected3] if selected3 != -1 else 0))


        features_.append(Muon_pt[muonIdx1])
        features_.append(Muon_eta[muonIdx1])
        features_.append(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1])
        dijet = jet1+jet2
        features_.append(dijet.Pt())
        features_.append(dijet.M())
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        dijet = jet1+jet2
        features_.append(dijet.M())

        ht = 0
        for j in np.arange(nJet)[maskJets]:
            ht = ht+Jet_pt[j]
        features_.append(ht)
        features_.append(np.sum(maskJets))
        if 'GluGlu' in process:
            m = (       (GenPart_genPartIdxMother>-1) &                         # Particles With Mother
                        (GenPart_pdgId[GenPart_genPartIdxMother]==25) &         # Particles daughters of Higgs
                        (GenPart_statusFlags[GenPart_genPartIdxMother]>=8192))  # Count only Higgs last copy (to avoid H->H)
            m1 = (m) & (abs(GenPart_eta)<1000)                               # Eta acceptance
            m2 = (m1) & (GenPart_pt>0)                                     # pT acceptance
            mLast = (m2) & (abs(GenPart_pdgId)==5)                             # Flavor (redundant)
        elif ('QCD' in process) | ('TTTo' in process):
            m = (       (GenPart_genPartIdxMother>-1) &                         # Particles With Mother
                        (GenPart_statusFlags[GenPart_genPartIdxMother]>=8192))  # Count only Higgs last copy (to avoid H->H)
            m1 = (m) & (abs(GenPart_eta)<1000)                               # Eta acceptance
            m2 = (m1) & (GenPart_pt>0)                                     # pT acceptance
            mLast = (m2) & (abs(GenPart_pdgId)==5)                             # Flavor (redundant)


            if np.sum(mLast)<2:
                continue
        dEta_1 = GenPart_eta[mLast][0] - GenJetNu_eta
        dPhi_1 = (GenPart_phi[mLast][0] - GenJetNu_phi + np.pi) % (2 * np.pi) - np.pi
        deltaR_gen1 = np.sqrt(dEta_1**2 + dPhi_1**2)
        gen_true1 = np.argmin(deltaR_gen1)

        mGenJetNu_ambiguity = np.arange(nGenJetNu)==gen_true1
        dEta_2 = (GenPart_eta[mLast][1] - GenJetNu_eta)
        dPhi_2 = (GenPart_phi[mLast][1] - GenJetNu_phi)
        deltaR_gen2 = np.sqrt(dEta_2**2 + dPhi_2**2)  
        gen_true2 = np.argmin( deltaR_gen2 +   mGenJetNu_ambiguity*1000)

        features_.append(GenJetNu_pt[gen_true1])
        features_.append(GenJetNu_pt[gen_true2])
        
        GenJetNu1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        GenJetNu2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        
        GenJetNu1.SetPtEtaPhiM(GenJetNu_pt[gen_true1], GenJetNu_eta[gen_true1], GenJetNu_phi[gen_true1], GenJetNu_mass[gen_true1])
        GenJetNu2.SetPtEtaPhiM(GenJetNu_pt[gen_true2], GenJetNu_eta[gen_true2], GenJetNu_phi[gen_true2], GenJetNu_mass[gen_true2])
        GenDijetNu = GenJetNu1+GenJetNu2

        features_.append(GenDijetNu.M())
        features_.append(GenDijetNu.M()/dijet.M())


        fileData.append(features_)
    


    feature_names = [
    "jet1_pt",    "jet1_eta",    "jet1_phi",    "jet1_mass", "jet1_btagDeepFlavB","jet1_btagTight", "jet1_dR_dijet","jet1_nMuons",
    "jet2_pt",    "jet2_eta",    "jet2_phi",    "jet2_mass", "jet2_btagDeepFlavB","jet2_btagTight", "jet2_dR_dijet","jet2_nMuons",
    "jet3_pt",    "jet3_eta",    "jet3_phi",    "jet3_mass", "jet3_dR_dijet","jet3_nMuons",
    "muon_pt",    "muon_eta",    "muon_dxySig",
    "dijet_pt",    "dijet_mass", "dijet_mass_2018",
    "ht",    "nJets",
    "genJetNu_pt_1",    "genJetNu_pt_2",    "genDijetNu_mass",    "target"
]

    fileData=pd.DataFrame(fileData, columns=feature_names)
    fileData.to_parquet(outFolder+"/%s_GenMatched_%s.parquet"%(process, fileNumber))

    return 0


if __name__ == "__main__":
    nanoFileName            = (sys.argv[1]) 
    fileNumber            = int(sys.argv[2]) 
    process                 = (sys.argv[3])    #(z for Z boson, h for Higgs boson)
    main(   nanoFileName=nanoFileName, fileNumber=fileNumber, process=process)