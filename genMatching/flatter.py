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
from functions import getDfProcesses_v2
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
def saveMatchedJets(fileNames, path, prefix):

    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    
    for fileName in fileNames:
        fileData=[]        # to store mjj for the matched signals
        outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(outFolder +"/%s_GenMatched_%s.parquet"%(prefix, fileNumber)):
            # if you already saved this file skip
            print("%s_GenMatched_%s.parquet already present\n"%(prefix, fileNumber))
            continue
        
        f = uproot.open(fileName)
        tree = f['Events']
        print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName)+1, len(fileNames), fileName[len(path)+1:], tree.num_entries))
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
            Jet_jetId                   =branches["Jet_jetId"][ev]
            Jet_puId                    =branches["Jet_puId"][ev]

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
            GenJetNu_pt                 = branches["GenJetNu_pt"][ev]

            GenPart_pt                  = branches["GenPart_pt"][ev]
            GenPart_eta                 = branches["GenPart_eta"][ev]
            GenPart_mass                = branches["GenPart_phi"][ev]
            GenPart_phi                 = branches["GenPart_mass"][ev]

            #Muons
            nMuon                       = branches["nMuon"][ev]
            Muon_charge                 = branches["Muon_charge"][ev]
            #Jet_chargeP1                 = branches["Jet_chargeP1"][ev]
            #Jet_chargeP3                 = branches["Jet_chargeP3"][ev]
            #Jet_chargeP5                 = branches["Jet_chargeP5"][ev]
            #Jet_chargeP7                 = branches["Jet_chargeP7"][ev]
            #Jet_charge                 = branches["Jet_charge"][ev]
            
            # limit the data to events where 4 jets are gen matched to higgs daughers
            if nJet<2:
                continue
            if len(GenJet_pt)<2:
                continue
            if prefix=='GluGluHToBB':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==25)
            elif prefix=='EWKZJets':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==23)
            if np.sum(m)==2:
                pass
                #matchedEvents=matchedEvents+1
            
            #elif np.sum(m)==1:
            #    newM = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5)
            #    if np.sum(newM)==2:
            #        m=newM
            #        #matchedEvents=matchedEvents+1
            #    else:
            #        continue
            else:
                pass
            
            
            # choice of jets is done in any case (unless no muon in jets)
            jetsToCheck = np.min([4, nJet])
            selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_jetId, Jet_puId)
            if selected1==999:
                continue
            if selected2==999:
                print(muonIdx1, muonIdx2)
                assert False
            features_.append(Jet_pt[selected1])
            features_.append(Jet_eta[selected1])
            features_.append(Jet_phi[selected1])
            features_.append(Jet_mass[selected1])
            features_.append(Jet_bReg2018[selected1])
            features_.append(Jet_rawFactor[selected1])
            features_.append(Jet_btagPNetB[selected1])
            features_.append(Jet_tagUParTAK4B[selected1])
            features_.append(Jet_PNetRegPtRawCorr[selected1])
            features_.append(Jet_PNetRegPtRawCorrNeutrino[selected1])
            features_.append(Jet_PNetRegPtRawRes[selected1])
            features_.append(Jet_ParTAK4RegPtRawCorr[selected1])
            features_.append(Jet_UParTAK4RegPtRawCorrNeutrino[selected1])
            features_.append(Jet_UParTAK4RegPtRawRes[selected1])

            features_.append(Jet_pt[selected2])
            features_.append(Jet_eta[selected2])
            features_.append(Jet_phi[selected2])
            features_.append(Jet_mass[selected2])
            features_.append(Jet_bReg2018[selected2])
            features_.append(Jet_rawFactor[selected2])
            features_.append(Jet_btagPNetB[selected2])
            features_.append(Jet_tagUParTAK4B[selected2])
            features_.append(Jet_PNetRegPtRawCorr[selected2])
            features_.append(Jet_PNetRegPtRawCorrNeutrino[selected2])
            features_.append(Jet_PNetRegPtRawRes[selected2])
            features_.append(Jet_ParTAK4RegPtRawCorr[selected2])
            features_.append(Jet_UParTAK4RegPtRawCorrNeutrino[selected2])
            features_.append(Jet_UParTAK4RegPtRawRes[selected2])

            jet2_leptonicCharge = 0
            for mu in range(nMuon):
                if mu==muonIdx1:
                    # dont want the muon in the first jet
                    continue
                if (mu != Jet_muonIdx1[selected2]) & (mu != Jet_muonIdx2[selected2]):
                    continue
                else:
                    jet2_leptonicCharge = Muon_charge[mu]
                    break
            features_.append(Muon_charge[muonIdx1])
            if muonIdx2!=999:
                features_.append(Muon_charge[muonIdx2])
            else:
                features_.append(999)
            features_.append(jet2_leptonicCharge)

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
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_rawFactor[selected1]* Jet_PNetRegPtRawCorr[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_rawFactor[selected2]* Jet_PNetRegPtRawCorr[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2
            
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())

# pnet with neutrinos
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_rawFactor[selected1]* Jet_PNetRegPtRawCorr[selected1]*Jet_PNetRegPtRawCorrNeutrino[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_rawFactor[selected2]* Jet_PNetRegPtRawCorr[selected2]*Jet_PNetRegPtRawCorrNeutrino[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2
            
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# parT no neutrinos

            jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_rawFactor[selected1]* Jet_PNetRegPtRawCorr[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_rawFactor[selected2]* Jet_PNetRegPtRawCorr[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2
            
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
# parT with neutrinos
            jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_rawFactor[selected1]* Jet_PNetRegPtRawCorr[selected1] * Jet_UParTAK4RegPtRawCorrNeutrino[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
            jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_rawFactor[selected2]* Jet_PNetRegPtRawCorr[selected2] * Jet_UParTAK4RegPtRawCorrNeutrino[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
            dijet = jet1 + jet2
            
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
            
# genjet features

            jetsWereCorrect = False
            if np.sum(m)==2:
                if (np.arange(nJet)[m][0]==selected1) & (np.arange(nJet)[m][1]==selected2):
                    jetsWereCorrect = True
                if (np.arange(nJet)[m][0]==selected2) &  (np.arange(nJet)[m][1]==selected1):
                    jetsWereCorrect = True
#                else:
#                    print(np.arange(nJet)[m][0], np.arange(nJet)[m][1], selected1, selected2)
                features_.append(jetsWereCorrect)
                features_.append(True)
                # if two jets genmatched
                # -2 -> error happened
                # -1 no 2 jets genmatched
                # 0 genjet -> genPart problem happened
                if GenJet_partonMotherIdx[Jet_genJetIdx[m][0]]!=-1:
                    try:
                        features_.append(GenPart_pt[GenJet_partonMotherIdx[Jet_genJetIdx[m][0]]])
                        features_.append(GenPart_eta[GenJet_partonMotherIdx[Jet_genJetIdx[m][0]]])
                        features_.append(GenPart_phi[GenJet_partonMotherIdx[Jet_genJetIdx[m][0]]])
                        features_.append(GenPart_mass[GenJet_partonMotherIdx[Jet_genJetIdx[m][0]]])
                    except:
                        print("Error")
                        features_.append(-2.)
                        features_.append(-2.)
                        features_.append(-2.)
                        features_.append(-2.)    
                else:
                    features_.append(0.)
                    features_.append(0.)
                    features_.append(0.)
                    features_.append(0.)
                if GenJet_partonMotherIdx[Jet_genJetIdx[m][1]]!=-1:
                    try:
                        features_.append(GenPart_pt[GenJet_partonMotherIdx[Jet_genJetIdx[m][1]]])
                        features_.append(GenPart_eta[GenJet_partonMotherIdx[Jet_genJetIdx[m][1]]])
                        features_.append(GenPart_phi[GenJet_partonMotherIdx[Jet_genJetIdx[m][1]]])
                        features_.append(GenPart_mass[GenJet_partonMotherIdx[Jet_genJetIdx[m][1]]])
                    except:
                        print("Error")
                        features_.append(-2.)
                        features_.append(-2.)
                        features_.append(-2.)
                        features_.append(-2.)
                else:
                    features_.append(0.)
                    features_.append(0.)
                    features_.append(0.)
                    features_.append(0.)
            else:
                features_.append(jetsWereCorrect) # this always false when the next is false. when the next is true might be true or false
                features_.append(False)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)
                features_.append(-1.)


            fileData.append(features_)
        
        fileData=pd.DataFrame(fileData, columns=[
            'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_bReg2018',
            'jet1_rawFactor', 'jet1_btagPNetB', 'jet1_tagUParTAK4B', 'jet1_PNetRegPtRawCorr', 'jet1_PNetRegPtRawCorrNeutrino', 'jet1_PNetRegPtRawRes', 'jet1_ParTAK4RegPtRawCorr', 'jet1_UParTAK4RegPtRawCorrNeutrino', 'jet1_UParTAK4RegPtRawRes', 
            'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_bReg2018',
            'jet2_rawFactor', 'jet2_btagPNetB', 'jet2_tagUParTAK4B', 'jet2_PNetRegPtRawCorr', 'jet2_PNetRegPtRawCorrNeutrino', 'jet2_PNetRegPtRawRes', 'jet2_ParTAK4RegPtRawCorr', 'jet2_UParTAK4RegPtRawCorrNeutrino', 'jet2_UParTAK4RegPtRawRes', 
            #'jet2_chargeP1', 'jet2_chargeP3', 'jet2_chargeP5', 'jet2_chargeP7', 'jet2_charge',
            'muon_charge', 'muon2_charge', 'jet2_leptonicCharge',
            'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass',
            'dijet_pt_2018', 'dijet_eta_2018', 'dijet_phi_2018', 'dijet_mass_2018',
            'dijet_pt_pnet', 'dijet_eta_pnet', 'dijet_phi_pnet', 'dijet_mass_pnet',
            'dijet_pt_pnetNu', 'dijet_eta_pnetNu', 'dijet_phi_pnetNu', 'dijet_mass_pnetNu',
            'dijet_pt_parT', 'dijet_eta_parT', 'dijet_phi_parT', 'dijet_mass_parT',
            'dijet_pt_partTNu', 'dijet_eta_partTNu', 'dijet_phi_partTNu', 'dijet_mass_partTNu',
            'correctChoice',
            'twoJetsGenMatched',
            #'jet1Corr_pt', 'jet1Corr_mass',
            #'jet2Corr_pt', 'jet2Corr_mass',
            #'dijetCorr_pt', 'dijetCorr_eta', 'dijetCorr_phi', 'dijetCorr_mass',
            'genPart1_pt', 'genPart1_eta', 'genPart1_phi', 'genPart1_mass',
            'genPart2_pt', 'genPart2_eta', 'genPart2_phi', 'genPart2_mass',
        ])
        fileData.to_parquet(outFolder+"/%s_GenMatched_%s.parquet"%(prefix, fileNumber))

    return 0


def main(nFiles, particle):
    df=getDfProcesses_v2()[0]
    if particle == 'H':
        path = df.nanoPath[0]
        fileNames = glob.glob(path+'/**/*.root', recursive=True)
        print("Looking for files in ", path)
        prefix="GluGluHToBB"
    #elif particle=="Z":
    #    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/EWKZJets2024Mar15/EWKZ2Jets_ZToQQ_TuneCP5_13TeV-madgraph-pythia8/crab_EWKZ2Jets_ZToQQ/240315_141326/0000"
    #    fileNames = glob.glob(path+'/EWKZ*.root')
    #    prefix="EWKZJets"
    if (nFiles > len(fileNames)) | (nFiles == -1):
        nFiles=len(fileNames)
    else:
        fileNames = fileNames[:nFiles]
        

    print("nFiles                : ", nFiles)
    saveMatchedJets(fileNames, path=path, prefix=prefix)

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                   = (sys.argv[2]).upper()    #(z for Z boson, h for Higgs boson)
    main(nFiles=nFiles, particle=particle)