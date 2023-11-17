import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re
import random
'''
Open _nFiles_ of MC ggHbb events. Find the matched jets b-flavoured, sisters, from the Higgs and with a triggering muon inside.
Save the mjj mass


Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
'''


# Now open the file and use the previous distribution
def saveMatchedJets(fileNames):
    
    goodChoice = 0
    totalChoice = 0
    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    #path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000/Hbb_QCDBackground_Run2_mc_2023Nov01_99.root"
    #print("\nOpening path:", path, "...")
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000"
    for fileName in fileNames:
        fileData=[ ]        # to store mjj for the matched signals
        outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData"
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(outFolder +"/jjFeaturesTrue_%s.npy"%fileNumber):
            # if you already saved this file skip
            print("jjFeaturesTrue_%s.npy already present\n"%(fileNumber))
            continue
        
        f = uproot.open(fileName)
        tree = f['Events']
        print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName)+1, len(fileNames), fileName[len(path)+1:], tree.num_entries))
        branches = tree.arrays()
        maxEntries = tree.num_entries
    

        for ev in  range(maxEntries):
            features_ = []
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()
                pass
            nGenJet                     = branches["nGenJet"][ev]
            GenJet_pt                   = branches["GenJet_pt"][ev]
            GenJet_eta                  = branches["GenJet_eta"][ev]
            GenJet_phi                  = branches["GenJet_phi"][ev]
            GenJet_mass                 = branches["GenJet_mass"][ev]

            GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
            GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
            GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
        # Reco Jets
            nJet                        = branches["nJet"][ev]
            Jet_eta                     = branches["Jet_eta"][ev]
            Jet_pt                      = branches["Jet_pt"][ev]
            Jet_phi                     = branches["Jet_phi"][ev]
            Jet_mass                    = branches["Jet_mass"][ev]

            Jet_muEF                    = branches["Jet_muEF"][ev]
            Jet_chEmEF                  = branches["Jet_chEmEF"][ev]
            Jet_neEmEF                  = branches["Jet_neEmEF"][ev]
            Jet_chHEF                   = branches["Jet_chHEF"][ev]
            Jet_neHEF                   = branches["Jet_neHEF"][ev]
            Jet_nConstituents           = branches["Jet_nConstituents"][ev]
            Jet_area                    = branches["Jet_area"][ev]
            Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
            Jet_btagCMVA                = branches["Jet_btagCMVA"][ev]
            Jet_btagCSVV2               = branches["Jet_btagCSVV2"][ev]
            Jet_btagDeepB               = branches["Jet_btagDeepB"][ev]
            Jet_btagDeepC               = branches["Jet_btagDeepC"][ev]
            Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]
            Jet_qgl                     = branches["Jet_qgl"][ev]
            Jet_nMuons                  = branches["Jet_nMuons"][ev]
            Jet_nElectrons              = branches["Jet_nElectrons"][ev]
            Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
            Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
            Jet_bRegMVA                 = branches["Jet_bRegMVA"][ev]
            Jet_bRegNN                  = branches["Jet_bRegNN"][ev]
            Jet_bRegNN2                 = branches["Jet_bRegNN2"][ev]
            Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
        # Muons
            Muon_pt                     = branches["Muon_pt"][ev]
            Muon_eta                    = branches["Muon_eta"][ev]
            Muon_phi                    = branches["Muon_phi"][ev]
            Muon_mass                   = branches["Muon_mass"][ev]
            Muon_isTriggering           = branches["Muon_isTriggering"][ev]
            Muon_dxy                    = branches["Muon_dxy"][ev]
            Muon_dxyErr                 = branches["Muon_dxyErr"][ev]
            Muon_dz                     = branches["Muon_dz"][ev]
            Muon_dzErr                  = branches["Muon_dzErr"][ev]

        # SVs
            SV_chi2                     = branches["SV_chi2"][ev]
            SV_pt                       = branches["SV_pt"][ev]
            SV_eta                      = branches["SV_eta"][ev]
            SV_phi                      = branches["SV_phi"][ev]
            SV_mass                     = branches["SV_mass"][ev]


            nSV                         = branches["nSV"][ev]
            SV_dlen                     = branches["SV_dlen"][ev]
            SV_dlenSig                  = branches["SV_dlenSig"][ev]
            SV_dxy                      = branches["SV_dxy"][ev]
            SV_dxySig                   = branches["SV_dxySig"][ev]
            SV_pAngle                   = branches["SV_pAngle"][ev]
            SV_charge                   = branches["SV_charge"][ev]

            SV_ntracks                  = branches["SV_ntracks"][ev]
            SV_ndof                     = branches["SV_ndof"][ev]        

        # Gen Parts
            nGenPart                    = branches["nGenPart"][ev]
            GenPart_pt                  = branches["GenPart_pt"][ev]
            GenPart_eta                 = branches["GenPart_eta"][ev]
            GenPart_phi                 = branches["GenPart_phi"][ev]
            GenPart_mass                = branches["GenPart_mass"][ev]
            GenPart_genPartIdxMother    = branches["GenPart_genPartIdxMother"][ev]
            GenPart_pdgId               = branches["GenPart_pdgId"][ev]


            idxJet1, idxJet2 = -1, -1       # index of the first jet satisfying requirements
            numberOfGoodJets=0              # number of jets satisfying requirements per event
            #ht = 0

            for i in range(nJet):
            # Find the jets from the signal
                if (Jet_genJetIdx[i]>-1):                                           # jet is matched to gen
                    #if Jet_genJetIdx[i]<nGenJet:                                    # some events have jetGenIdx > nGenJet
                    if abs(GenJet_partonFlavour[Jet_genJetIdx[i]])==5:          # jet matched to genjet from b

                        if GenJet_partonMotherPdgId[Jet_genJetIdx[i]]==25:      # jet parton mother is higgs (b comes from h)
                            numberOfGoodJets=numberOfGoodJets+1
                            assert numberOfGoodJets<=2, "Error numberOfGoodJets = %d"%numberOfGoodJets                 # check there are no more than 2 jets from higgs
                            if idxJet1==-1:                                     # first match
                                idxJet1=i
                            elif GenJet_partonMotherIdx[Jet_genJetIdx[idxJet1]]==GenJet_partonMotherIdx[Jet_genJetIdx[i]]:  # second match. Also sisters
                                idxJet2=i    
            if ((idxJet1==-1) | (idxJet2==-1)):
                # take only events where there is the interested signal
                continue
            assert idxJet1>-0.01
            assert idxJet2>-0.01
            
            # Assert that at leasat one of the two jets has overlap with a triggering muon
            jetContainsTriggeringMuon_1 = False
            if Jet_muonIdx1[idxJet1]>-1:
                jetContainsTriggeringMuon_1 = bool(Muon_isTriggering[Jet_muonIdx1[idxJet1]])
            
            # else the jetContainsTriggeringMuon is already false. Check the second Muon inside the jet (if any)
            if Jet_muonIdx2[idxJet1]>-1:
                jetContainsTriggeringMuon_1 = jetContainsTriggeringMuon_1 | bool(Muon_isTriggering[Jet_muonIdx2[idxJet1]])
            
            # here if the jets have one muon that is triggering among the two considered inside the jets this leaf is true
            # do the same for the second jet of the pair
            
            jetContainsTriggeringMuon_2 = False
            if Jet_muonIdx1[idxJet2]>-1:
                jetContainsTriggeringMuon_2 = bool(Muon_isTriggering[Jet_muonIdx1[idxJet2]])
            
            if Jet_muonIdx2[idxJet2]>-1:
                jetContainsTriggeringMuon_2 = jetContainsTriggeringMuon_2 | bool(Muon_isTriggering[Jet_muonIdx2[idxJet2]])
            
            
            if (jetContainsTriggeringMuon_1|jetContainsTriggeringMuon_2):
                pass
            else:
                continue


            # idxJet1 and idxJet2 are the matched ones
            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet1.SetPtEtaPhiM(Jet_pt[idxJet1]*Jet_bRegNN2[idxJet1], Jet_eta[idxJet1], Jet_phi[idxJet1], Jet_mass[idxJet1])
            jet2.SetPtEtaPhiM(Jet_pt[idxJet2]*Jet_bRegNN2[idxJet2], Jet_eta[idxJet2], Jet_phi[idxJet2], Jet_mass[idxJet2])
            #features_.append((jet1 + jet2).M())

            features_.append(Jet_pt[idxJet1])                 #0
            features_.append(Jet_eta[idxJet1])                #1
            features_.append(Jet_phi[idxJet1])                #2
            features_.append(Jet_mass[idxJet1])               #3
            features_.append(Jet_nMuons[idxJet1])             #4
            features_.append(Jet_nElectrons[idxJet1])         #5
            features_.append(Jet_btagDeepFlavB[idxJet1])      #6
            #features_.append(jet1.Pt()/jet1.E())                #7

            features_.append(Jet_pt[idxJet2])
            features_.append(Jet_eta[idxJet2])
            features_.append(Jet_phi[idxJet2])
            features_.append(Jet_mass[idxJet2])
            features_.append(Jet_nMuons[idxJet2])
            features_.append(Jet_nElectrons[idxJet2])
            features_.append(Jet_btagDeepFlavB[idxJet2])
            #features_.append(jet2.Pt()/jet2.E())

            dijet = jet1 + jet2
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())
            features_.append(jet1.DeltaR(jet2))
            features_.append(abs(jet1.Eta() - jet2.Eta()))
            deltaPhi = jet1.Phi()-jet2.Phi()
            deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
            features_.append(abs(deltaPhi))     
            angVariable = np.pi - abs(deltaPhi) + abs(jet1.Eta() - jet2.Eta())
            features_.append(angVariable)
            tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
            features_.append(tau)
            ht = 0
            for idx in range(nJet):
                ht = ht+Jet_pt[idx]

            features_.append(ht)
            #if nSV>0:
            #    features_.append(nSV)
            #    features_.append(SV_chi2[0])
            #    features_.append(SV_pt[0])
            #    features_.append(SV_eta[0])
            #    features_.append(SV_phi[0])
            #    features_.append(SV_mass[0])
            #    features_.append(SV_dlen[0])
            #    features_.append(SV_dlenSig[0])
            #    features_.append(SV_dxy[0])
            #    features_.append(SV_dxySig[0])
            #    features_.append(SV_pAngle[0])
            #    features_.append(SV_charge[0])
            #else:
            #    features_.append(0)
            #    for z in range(11):
            #        features_.append(-999)
#
            #if nSV>1:
            #    features_.append(SV_chi2[1])
            #    features_.append(SV_pt[1])
            #    features_.append(SV_eta[1])
            #    features_.append(SV_phi[1])
            #    features_.append(SV_mass[1])
            #    features_.append(SV_dlen[1])
            #    features_.append(SV_dlenSig[1])
            #    features_.append(SV_dxy[1])
            #    features_.append(SV_dxySig[1])
            #    features_.append(SV_pAngle[1])
            #    features_.append(SV_charge[1])
            #else:
            #    for z in range(11):
            #        features_.append(-999)

            #GC To be Uncommented GCmuonIdx = 999
            #GC To be Uncommented GCwhichJetHasLeadingTrigMuon = -1
            #GC To be Uncommented GC# First jet
#GC To be Uncommented GC
            #GC To be Uncommented GCif Jet_muonIdx1[idxJet1]>-1:
            #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx1[idxJet1]]:
            #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx1[idxJet1] else Jet_muonIdx1[idxJet1]
            #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[idxJet1] else idxJet1
            #GC To be Uncommented GCif Jet_muonIdx2[idxJet1]>-1:
            #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx2[idxJet1]]:
            #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx2[idxJet1] else Jet_muonIdx2[idxJet1]
            #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[idxJet1] else idxJet1
            #GC To be Uncommented GC# Second jet
            #GC To be Uncommented GCif Jet_muonIdx1[idxJet2]>-1:
            #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx1[idxJet2]]:
            #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx1[idxJet2] else Jet_muonIdx1[idxJet2]
            #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[idxJet2] else idxJet2
            #GC To be Uncommented GCif Jet_muonIdx2[idxJet2]>-1:
            #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx2[idxJet2]]:
            #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx2[idxJet2] else Jet_muonIdx2[idxJet2]
            #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[idxJet2] else idxJet2
#GC To be Uncommented GC
#GC To be Uncommented GC
#GC To be Uncommented GC
            #GC To be Uncommented GCmuon = ROOT.TLorentzVector(0., 0., 0., 0.)
            #GC To be Uncommented GCmuon.SetPtEtaPhiM(Muon_pt[muonIdx], Muon_eta[muonIdx], Muon_phi[muonIdx], Muon_mass[muonIdx])
#GC To be Uncommented GC
#GC To be Uncommented GC
            #GC To be Uncommented GCfeatures_.append(muon.Pt())
            #GC To be Uncommented GCfeatures_.append(muon.Eta())
            #GC To be Uncommented GCif whichJetHasLeadingTrigMuon==idxJet1:
            #GC To be Uncommented GC    features_.append(muon.DeltaR(jet1) )
            #GC To be Uncommented GC    features_.append(muon.Pt()/jet1.Pt() )
            #GC To be Uncommented GCelif whichJetHasLeadingTrigMuon==idxJet2:
            #GC To be Uncommented GC    features_.append(muon.DeltaR(jet2) )
            #GC To be Uncommented GC    features_.append(muon.Pt()/jet2.Pt() )
            #GC To be Uncommented GCelse:
            #GC To be Uncommented GC    assert False
#GC To be Uncommented GC
            #GC To be Uncommented GCfeatures_.append(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])
            #GC To be Uncommented GCfeatures_.append(Muon_dz[muonIdx]/Muon_dzErr[muonIdx])
            #GC To be Uncommented GCassert Muon_isTriggering[muonIdx]
            fileData.append(features_)
        
        fileData=np.array(fileData)
        np.save(outFolder+"/jjFeaturesTrue_%s.npy"%fileNumber, fileData)

    return 0


def main(nFiles):
    
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000"
    fileNames = glob.glob(path+'/Hbb_QCDBackground_Run2_mc_2023Nov01*.root')
    random.shuffle(fileNames)
    if (nFiles > len(fileNames)) | (nFiles == -1):
        pass
    else:
        fileNames = fileNames[:nFiles]
        

    print("nFiles                : ", nFiles)
    saveMatchedJets(fileNames)

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    main(nFiles=nFiles)