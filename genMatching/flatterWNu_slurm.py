# %%
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
        
        GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
        GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
        GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
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

        Jet_PNetRegPtRawCorr              = branches["Jet_PNetRegPtRawCorr"][ev]
        Jet_PNetRegPtRawCorrNeutrino      = branches["Jet_PNetRegPtRawCorrNeutrino"][ev]
        Jet_PNetRegPtRawRes               = branches["Jet_PNetRegPtRawRes"][ev]
        Jet_ParTAK4RegPtRawCorr           = branches["Jet_ParTAK4RegPtRawCorr"][ev]
        Jet_UParTAK4RegPtRawCorrNeutrino  = branches["Jet_UParTAK4RegPtRawCorrNeutrino"][ev]
        Jet_UParTAK4RegPtRawRes           = branches["Jet_UParTAK4RegPtRawRes"][ev]

        GenJet_pt                   = branches["GenJet_pt"][ev]
        GenJet_eta                  = branches["GenJet_eta"][ev]
        GenJet_phi                  = branches["GenJet_phi"][ev]
        GenJet_mass                 = branches["GenJet_mass"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]
        Jet_qgl                     = branches["Jet_qgl"][ev]


        nGenJetNu                   = branches["nGenJetNu"][ev]
        GenJetNu_pt                 = branches["GenJetNu_pt"][ev]
        GenJetNu_eta                = branches["GenJetNu_eta"][ev]
        GenJetNu_phi                = branches["GenJetNu_phi"][ev]
        GenJetNu_mass               = branches["GenJetNu_mass"][ev]
        GenJetNu_partonMotherPdgId  = branches["GenJetNu_partonMotherPdgId"][ev]
        GenJetNu_partonMotherIdx    = branches["GenJetNu_partonMotherIdx"][ev]
        GenJetNu_partonFlavour      = branches["GenJetNu_partonFlavour"][ev]
        GenJetNu_hadronFlavour      = branches["GenJetNu_hadronFlavour"][ev]

        GenPart_pt                  = branches["GenPart_pt"][ev]
        GenPart_eta                 = branches["GenPart_eta"][ev]
        GenPart_mass                = branches["GenPart_phi"][ev]
        GenPart_phi                 = branches["GenPart_mass"][ev]

        #Muons
        nMuon                       = branches["nMuon"][ev]
        Muon_charge                 = branches["Muon_charge"][ev]
        
        # limit the data to events where 4 jets are gen matched to higgs daughers
        if nJet<2:
            continue
        
#Method 1
        #Jet_genJetNuIdx = []
        #Jet_genJetNu_dR = []
        #for j in range(nJet):
        #    minDeltaR = 999
        #    minDeltaR_idx = 999
        #    for gjnu in range(nGenJetNu):
        #        delta_eta = (Jet_eta[j] - GenJetNu_eta[gjnu])
        #        delta_phi = (Jet_phi[j] - GenJetNu_phi[gjnu])
        #        if delta_phi > np.pi:
        #            delta_phi -= 2 * np.pi
        #        elif delta_phi < -np.pi:
        #            delta_phi += 2 * np.pi
        #        deltaR = np.sqrt(delta_eta**2 + delta_phi**2)
        #        if (deltaR<0.1) & (deltaR<minDeltaR):
        #            minDeltaR = deltaR
        #            minDeltaR_idx = gjnu
        #    if minDeltaR == 999:
        #        Jet_genJetNuIdx.append(-1)
        #    else:
        #        Jet_genJetNuIdx.append(minDeltaR_idx)
        #    Jet_genJetNu_dR.append(minDeltaR)

#Method 2
        Jet_genJetNuIdx = []
        Jet_genJetNu_dR = []
        for j in range(nJet):
            if Jet_genJetIdx[j]==-1:
                Jet_genJetNuIdx.append(-1)
                Jet_genJetNu_dR.append(-1)
            else:
                eta = GenJet_eta[Jet_genJetIdx[j]]
                phi = GenJet_phi[Jet_genJetIdx[j]]
                minDeltaR = 999
                minDeltaR_idx = 999
                for gjnu in range(nGenJetNu):
                    delta_eta = (eta - GenJetNu_eta[gjnu])
                    delta_phi = (phi - GenJetNu_phi[gjnu])
                    if delta_phi > np.pi:
                        delta_phi -= 2 * np.pi
                    elif delta_phi < -np.pi:
                        delta_phi += 2 * np.pi
                    deltaR = np.sqrt(delta_eta**2 + delta_phi**2)
                    if (deltaR<minDeltaR):
                        minDeltaR = deltaR
                        minDeltaR_idx = gjnu
                if minDeltaR == 999:
                    Jet_genJetNuIdx.append(-1)
                else:
                    Jet_genJetNuIdx.append(minDeltaR_idx)
                Jet_genJetNu_dR.append(minDeltaR)


        





# Find the two JetsWithNu that are matched
        Jet_genJetNuIdx = np.array(Jet_genJetNuIdx)
        Jet_genJetNu_dR = np.array(Jet_genJetNu_dR)

        if process=='GluGluHToBB':
            m = (Jet_genJetNuIdx>-1) & (abs(GenJetNu_partonFlavour[Jet_genJetNuIdx])==5) & (GenJetNu_partonMotherPdgId[Jet_genJetNuIdx]==25)

        if np.sum(m)==2:
            selected1, selected2 = np.arange(nJet)[m][0], np.arange(nJet)[m][1]
        elif np.sum(m)==1:
            selected1, selected2 = np.arange(nJet)[m][0], -1
        else:
            continue
        

        

        features_.append(Jet_pt[selected1])
        features_.append(Jet_eta[selected1])
        features_.append(Jet_phi[selected1])
        features_.append(Jet_mass[selected1])
        features_.append(Jet_bReg2018[selected1])

        features_.append(Jet_rawFactor[selected1])
        features_.append(Jet_btagDeepFlavB[selected1])
        features_.append(Jet_btagPNetB[selected1])
        features_.append(Jet_tagUParTAK4B[selected1])
        features_.append(Jet_PNetRegPtRawCorr[selected1])
        features_.append(Jet_PNetRegPtRawCorrNeutrino[selected1])
        features_.append(Jet_PNetRegPtRawRes[selected1])
        features_.append(Jet_ParTAK4RegPtRawCorr[selected1])
        features_.append(Jet_UParTAK4RegPtRawCorrNeutrino[selected1])
        features_.append(Jet_UParTAK4RegPtRawRes[selected1])

        if selected2>-1:
            features_.append(Jet_pt[selected2])
            features_.append(Jet_eta[selected2])
            features_.append(Jet_phi[selected2])
            features_.append(Jet_mass[selected2])
            features_.append(Jet_bReg2018[selected2])
            features_.append(Jet_rawFactor[selected2])
            features_.append(Jet_btagDeepFlavB[selected2])
            features_.append(Jet_btagPNetB[selected2])
            features_.append(Jet_tagUParTAK4B[selected2])
            features_.append(Jet_PNetRegPtRawCorr[selected2])
            features_.append(Jet_PNetRegPtRawCorrNeutrino[selected2])
            features_.append(Jet_PNetRegPtRawRes[selected2])
            features_.append(Jet_ParTAK4RegPtRawCorr[selected2])
            features_.append(Jet_UParTAK4RegPtRawCorrNeutrino[selected2])
            features_.append(Jet_UParTAK4RegPtRawRes[selected2])
        else:
            for jdx in range(14):
                features_.append(-1)

        if selected2>-1:
# reco
            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet1.SetPtEtaPhiM(Jet_pt[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# breg 2018
            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1]*Jet_bReg2018[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2]*Jet_bReg2018[selected2])
            dijet = jet1 + jet2

            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# pnet no neutrinos
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])* Jet_PNetRegPtRawCorr[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])* Jet_PNetRegPtRawCorr[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2

            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())

# pnet with neutrinos
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])* Jet_PNetRegPtRawCorr[selected1]*Jet_PNetRegPtRawCorrNeutrino[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])* Jet_PNetRegPtRawCorr[selected2]*Jet_PNetRegPtRawCorrNeutrino[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2

            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# parT no neutrinos

            jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])* Jet_ParTAK4RegPtRawCorr[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])* Jet_ParTAK4RegPtRawCorr[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2

            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# parT with neutrinos
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])* Jet_ParTAK4RegPtRawCorr[selected1] * Jet_UParTAK4RegPtRawCorrNeutrino[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])* Jet_ParTAK4RegPtRawCorr[selected2] * Jet_UParTAK4RegPtRawCorrNeutrino[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2

            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
        
# genjet features
        else:
            for jdx in range(24):
                features_.append(-1.)
        
# gen matching and pt

        features_.append(GenJetNu_pt[Jet_genJetNuIdx[selected1]])
        features_.append(GenJetNu_eta[Jet_genJetNuIdx[selected1]])
        features_.append(GenJetNu_phi[Jet_genJetNuIdx[selected1]])
        features_.append(GenJetNu_mass[Jet_genJetNuIdx[selected1]])
        features_.append(Jet_genJetNu_dR[selected1])
        if selected2>-1:
            features_.append(GenJetNu_pt[Jet_genJetNuIdx[selected2]])
            features_.append(GenJetNu_eta[Jet_genJetNuIdx[selected2]])
            features_.append(GenJetNu_phi[Jet_genJetNuIdx[selected2]])
            features_.append(GenJetNu_mass[Jet_genJetNuIdx[selected2]])

            genJet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            genJet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            genJet1.SetPtEtaPhiM(GenJetNu_pt[Jet_genJetNuIdx[selected1]],
                                    GenJetNu_eta[Jet_genJetNuIdx[selected1]],
                                    GenJetNu_phi[Jet_genJetNuIdx[selected1]],
                                    GenJetNu_mass[Jet_genJetNuIdx[selected1]])
            genJet2.SetPtEtaPhiM(GenJetNu_pt[Jet_genJetNuIdx[selected2]],
                                    GenJetNu_eta[Jet_genJetNuIdx[selected2]],
                                    GenJetNu_phi[Jet_genJetNuIdx[selected2]],
                                    GenJetNu_mass[Jet_genJetNuIdx[selected2]])
            genDijet = genJet1 + genJet2
            features_.append(genDijet.M())
            features_.append(Jet_genJetNu_dR[selected2])
        else:
            features_.append(-1)
            features_.append(-1)
            features_.append(-1)
            features_.append(-1)
            features_.append(-1)
            features_.append(-1)

        fileData.append(features_)
    
    fileData=pd.DataFrame(fileData, columns=[
        'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_bReg2018',
        'jet1_rawFactor', 'jet1_btagDeepFlavB', 'jet1_btagPNetB', 'jet1_tagUParTAK4B', 'jet1_PNetRegPtRawCorr', 'jet1_PNetRegPtRawCorrNeutrino', 'jet1_PNetRegPtRawRes', 'jet1_ParTAK4RegPtRawCorr', 'jet1_UParTAK4RegPtRawCorrNeutrino', 'jet1_UParTAK4RegPtRawRes', 
        'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_bReg2018',
        'jet2_rawFactor', 'jet2_btagDeepFlavB', 'jet2_btagPNetB', 'jet2_tagUParTAK4B', 'jet2_PNetRegPtRawCorr', 'jet2_PNetRegPtRawCorrNeutrino', 'jet2_PNetRegPtRawRes', 'jet2_ParTAK4RegPtRawCorr', 'jet2_UParTAK4RegPtRawCorrNeutrino', 'jet2_UParTAK4RegPtRawRes', 
        'dijet_pt_reco', 'dijet_eta_reco', 'dijet_phi_reco', 'dijet_mass_reco',
        'dijet_pt_2018', 'dijet_eta_2018', 'dijet_phi_2018', 'dijet_mass_2018',
        'dijet_pt_pnet', 'dijet_eta_pnet', 'dijet_phi_pnet', 'dijet_mass_pnet',
        'dijet_pt_pnetNu', 'dijet_eta_pnetNu', 'dijet_phi_pnetNu', 'dijet_mass_pnetNu',
        'dijet_pt_parT', 'dijet_eta_parT', 'dijet_phi_parT', 'dijet_mass_parT',
        'dijet_pt_partTNu', 'dijet_eta_partTNu', 'dijet_phi_partTNu', 'dijet_mass_partTNu',
        'genJetNu1_pt', 'genJetNu1_eta', 'genJetNu1_phi', 'genJetNu1_mass', 'jet1_genJetNu_dR',
        'genJetNu2_pt', 'genJetNu2_eta', 'genJetNu2_phi', 'genJetNu2_mass', 'genDijet_mass', 'jet2_genJetNu_dR',
    ])
    fileData.to_parquet(outFolder+"/%s_GenMatched_%s.parquet"%(process, fileNumber))

    return 0


if __name__ == "__main__":
    nanoFileName            = (sys.argv[1]) 
    fileNumber            = int(sys.argv[2]) 
    process                 = (sys.argv[3])    #(z for Z boson, h for Higgs boson)
    main(   nanoFileName=nanoFileName, fileNumber=fileNumber, process=process)