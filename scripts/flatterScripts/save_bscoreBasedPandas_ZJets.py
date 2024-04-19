import numpy as np
import uproot
import glob
import sys
import ROOT
import os
import re
import pandas as pd
import random



'''
Givena a chosen criterion to select the candidates jets that are most likely to come from the Higgs, open the files from BParrking data and save them in flaat format
Select _nFiles_ files 
Consider the fist _maxJet_ in order of pT and choose the dijet that maximize the chosen criterion
Returns the percentage of selected pairs that match the correct pairs only when a complete matching is done.

Args:
    isMC 0:Data, 1:ggH, 2,3,4 : ttbar H, S, L, 5->16 QCD HTclass
    nFiles      = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    maxEntries  = int(sys.argv[2]) if len(sys.argv) > 1 else -1
    maxJet      = int(sys.argv[3]) if len(sys.argv) > 3 else 5

'''
def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB):
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if abs(Jet_eta[i])>2.5:     # exclude jets>2.5 from the jets with  muon group
            continue
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                jetsWithMuon.append(i)
                muonIdxs.append(Jet_muonIdx1[i])
                continue
        if (Jet_muonIdx2[i]>-1):
            if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                jetsWithMuon.append(i)
                muonIdxs.append(Jet_muonIdx2[i])
                continue
    assert len(muonIdxs)==len(jetsWithMuon)

# Now loop over these jets as first element of the pair

    
    for i in jetsWithMuon:
        for j in range(0, jetsToCheck):
            #print(i, j, Jet_eta[j])
            if i==j:
                continue
            if abs(Jet_eta[j])>2.5:
                continue

            currentScore = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[j]
            if currentScore>score:
                score=currentScore
                if j not in jetsWithMuon:  # if i has the muon only. jet1 is the jet with the muon.
                    selected1 = i
                    selected2 = j
                    muonIdx = muonIdxs[jetsWithMuon.index(i)]
                elif (i in jetsWithMuon) & (j in jetsWithMuon):
                    if muonIdxs[jetsWithMuon.index(i)] < muonIdxs[jetsWithMuon.index(j)]:
                        selected1 = i
                        selected2 = j
                        muonIdx = muonIdxs[jetsWithMuon.index(i)]
                    elif muonIdxs[jetsWithMuon.index(i)] > muonIdxs[jetsWithMuon.index(j)]:
                        selected1 = j
                        selected2 = i
                        muonIdx = muonIdxs[jetsWithMuon.index(j)]
                else:
                    assert False


    return selected1, selected2, muonIdx

def treeFlatten(fileName, maxEntries, maxJet, isMC):
    '''Require one muon in the dijet. Choose dijets based on their bscore. save all the features of the event append them in a list'''
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries))
    file_ =[]
    

    # open the file for the SF
    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    f = ROOT.TFile(histPath, "READ")
    hist = f.Get("hist_scale_factor")
    for ev in  range(maxEntries):
        
        features_ = []
        if (ev%(int(maxEntries/100))==0):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("%d%%"%(ev/maxEntries*100))
            sys.stdout.flush()
    
    # Reco Jets
        nJet                        = branches["nJet"][ev]
        Jet_eta                     = branches["Jet_eta"][ev]
        Jet_pt                      = branches["Jet_pt"][ev]
        Jet_phi                     = branches["Jet_phi"][ev]
        Jet_mass                    = branches["Jet_mass"][ev]

        #Jet_muEF                    = branches["Jet_muEF"][ev]
        #Jet_chEmEF                  = branches["Jet_chEmEF"][ev]
        #Jet_neEmEF                  = branches["Jet_neEmEF"][ev]
        #Jet_chHEF                   = branches["Jet_chHEF"][ev]
        #Jet_neHEF                   = branches["Jet_neHEF"][ev]
        #Jet_nConstituents           = branches["Jet_nConstituents"][ev]
        Jet_area                    = branches["Jet_area"][ev]
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
        #Jet_btagCMVA                = branches["Jet_btagCMVA"][ev]
        #Jet_btagCSVV2               = branches["Jet_btagCSVV2"][ev]
        #Jet_btagDeepB               = branches["Jet_btagDeepB"][ev]
        #Jet_btagDeepC               = branches["Jet_btagDeepC"][ev]
        #Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]
        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        #Jet_bRegMVA                 = branches["Jet_bRegMVA"][ev]
        #Jet_bRegNN2                  = branches["Jet_bRegNN2"][ev]
        Jet_bReg2018                 = branches["Jet_bReg2018"][ev]
        #Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
        #GenJet_partonFlavour         = branches["GenJet_partonFlavour"][ev]
        #GenJet_partonMotherIdx       = branches["GenJet_partonMotherIdx"][ev]
        #GenPart_genPartIdxMother     = branches["GenPart_genPartIdxMother"][ev]
        #GenPart_pdgId                = branches["GenPart_pdgId"][ev]
        #GenJet_partonMotherPdgId     = branches["GenJet_partonMotherPdgId"][ev]
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
        Muon_pfIsoId                = branches["Muon_pfIsoId"][ev]  # 1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight)
        Muon_pfRelIso03_all         = branches["Muon_pfRelIso03_all"][ev]
        Muon_ip3d                   = branches["Muon_ip3d"][ev]
        Muon_sip3d                  = branches["Muon_sip3d"][ev]
        #Muon_isGlobal               = branches["Muon_isGlobal"][ev]
        #Muon_isTracker              = branches["Muon_isTracker"][ev]
        #Muon_mediumId               = branches["Muon_mediumId"][ev]
        #Muon_looseId                = branches["Muon_looseId"][ev]
        #Muon_isPFcand               = branches["Muon_isPFcand"][ev]
        #Muon_softId                 = branches["Muon_softId"][ev]
        Muon_tightId                = branches["Muon_tightId"][ev]
        Muon_tkIsoId                = branches["Muon_tkIsoId"][ev]
        #Muon_triggerIdLoose         = branches["Muon_triggerIdLoose"][ev]
        PV_npvs                     = branches["PV_npvs"][ev]


    
    # SVs
        #SV_chi2                     = branches["SV_chi2"][ev]
        #SV_pt                       = branches["SV_pt"][ev]
        #SV_eta                      = branches["SV_eta"][ev]
        #SV_phi                      = branches["SV_phi"][ev]
        #SV_mass                     = branches["SV_mass"][ev]
#
        #
        #nSV                         = branches["nSV"][ev]
        #SV_dlen                     = branches["SV_dlen"][ev]
        #SV_dlenSig                  = branches["SV_dlenSig"][ev]
        #SV_dxy                      = branches["SV_dxy"][ev]
        #SV_dxySig                   = branches["SV_dxySig"][ev]
        #SV_pAngle                   = branches["SV_pAngle"][ev]
        #SV_charge                   = branches["SV_charge"][ev]
#
        #SV_ntracks                  = branches["SV_ntracks"][ev]
        #SV_ndof                     = branches["SV_ndof"][ev]        
# requires gen matching
        #if isMC==2:
        #    idxJet1, idxJet2 = -1, -1       # index of the first jet satisfying requirements
        #    numberOfGoodJets=0              # number of jets satisfying requirements per event
        #    #ht = 0
#
        #    for i in range(nJet):
        #    # Find the jets from the signal
        #        if (Jet_genJetIdx[i]>-1):                                           # jet is matched to gen
        #            if abs(GenJet_partonFlavour[Jet_genJetIdx[i]])==5:          # jet matched to genjet from b
        #                if GenJet_partonMotherPdgId[Jet_genJetIdx[i]]==23:      # jet parton mother is Z boson (b comes from Z)
        #                    numberOfGoodJets=numberOfGoodJets+1
        #                    assert numberOfGoodJets<=2, "Error numberOfGoodJets = %d"%numberOfGoodJets                 # check there are no more than 2 jets from higgs
        #                    if idxJet1==-1:                                     # first match
        #                        idxJet1=i
        #                    elif GenJet_partonMotherIdx[Jet_genJetIdx[idxJet1]]==GenJet_partonMotherIdx[Jet_genJetIdx[i]]:  # second match. Also sisters
        #                        idxJet2=i    
        #    if ((idxJet1==-1) | (idxJet2==-1)):
        #        continue



        jetsToCheck = np.min([maxJet, nJet])                                 # !!! max 4 jets to check   !!!
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        jetsToCheck = np.min([maxJet, nJet])
        
        selected1, selected2, muonIdx = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB)

        if selected1==999:
            #print("skipped")
            continue
        if selected2==999:
            assert False
        #print("Taken")
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        dijet = jet1 + jet2
        #if (nSV>0):
        #    if (SV_dlenSig[0]<1):
        #        assert False
        features_.append(Jet_pt[selected1])                 #0
        features_.append(Jet_eta[selected1])                #1
        features_.append(Jet_phi[selected1])                #2
        features_.append(Jet_mass[selected1])               #3
        features_.append(Jet_nMuons[selected1])             #4
        features_.append(Jet_nElectrons[selected1])         #5
        features_.append(Jet_btagDeepFlavB[selected1])      #6
        features_.append(Jet_area[selected1])
        features_.append(Jet_qgl[selected1])
        #features_.append(jet1.Pt()/jet1.E())                #7
        
        features_.append(Jet_pt[selected2])
        features_.append(Jet_eta[selected2])
        features_.append(Jet_phi[selected2])
        features_.append(Jet_mass[selected2])
        features_.append(Jet_nMuons[selected2])
        features_.append(Jet_nElectrons[selected2])
        features_.append(Jet_btagDeepFlavB[selected2])
        features_.append(Jet_area[selected2])
        features_.append(Jet_qgl[selected2])
        #features_.append(jet2.Pt()/jet2.E())

        if dijet.Pt()<1e-5:
            assert False
        features_.append(np.float32(dijet.Pt()))
        features_.append(np.float32(dijet.Eta()))
        features_.append(np.float32(dijet.Phi()))
        features_.append(np.float32(dijet.M()))
        features_.append(np.float32(jet1.DeltaR(jet2)))
        features_.append(np.float32(abs(jet1.Eta() - jet2.Eta())))
        deltaPhi = jet1.Phi()-jet2.Phi()
        deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
        features_.append(np.float32(abs(deltaPhi)))     
        angVariable = np.pi - abs(deltaPhi) + abs(jet1.Eta() - jet2.Eta())
        features_.append(np.float32(angVariable))
        tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
        features_.append(tau.astype(np.float32))
        features_.append(nJet)
        ht = 0
        for idx in range(nJet):
            ht = ht+Jet_pt[idx]

        features_.append(ht.astype(np.float32))
        #if nSV>0:
        #    features_.append(nSV)
        #    features_.append(SV_pt[0])
        
        newMuonIdx = 999
        whichJetHasLeadingTrigMuon = -1
        # First jet
        
        if Jet_muonIdx1[selected1]>-1:
            if Muon_isTriggering[Jet_muonIdx1[selected1]]:
                newMuonIdx = newMuonIdx if newMuonIdx < Jet_muonIdx1[selected1] else Jet_muonIdx1[selected1]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if newMuonIdx < Jet_muonIdx1[selected1] else selected1
        if Jet_muonIdx2[selected1]>-1:
            if Muon_isTriggering[Jet_muonIdx2[selected1]]:
                newMuonIdx = newMuonIdx if newMuonIdx < Jet_muonIdx2[selected1] else Jet_muonIdx2[selected1]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if newMuonIdx < Jet_muonIdx2[selected1] else selected1
        # Second jet
        if Jet_muonIdx1[selected2]>-1:
            if Muon_isTriggering[Jet_muonIdx1[selected2]]:
                newMuonIdx = newMuonIdx if newMuonIdx < Jet_muonIdx1[selected2] else Jet_muonIdx1[selected2]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if newMuonIdx < Jet_muonIdx1[selected2] else selected2
        if Jet_muonIdx2[selected2]>-1:
            if Muon_isTriggering[Jet_muonIdx2[selected2]]:
                newMuonIdx = newMuonIdx if newMuonIdx < Jet_muonIdx2[selected2] else Jet_muonIdx2[selected2]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if newMuonIdx < Jet_muonIdx2[selected2] else selected2

        assert whichJetHasLeadingTrigMuon == selected1
        assert newMuonIdx == muonIdx
        
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx], Muon_eta[muonIdx], Muon_phi[muonIdx], Muon_mass[muonIdx])


        features_.append(np.float32(muon.Pt()))
        features_.append(np.float32(muon.Eta()))
        #if whichJetHasLeadingTrigMuon==selected1:
        #    features_.append(muon.DeltaR(jet1) )
        #    features_.append(muon.Pt()/jet1.Pt() )
        #elif whichJetHasLeadingTrigMuon==selected2:
        #    features_.append(muon.DeltaR(jet2) )
        #    features_.append(muon.Pt()/jet2.Pt() )
        #else:
        #    assert False
        
        features_.append(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])
        features_.append(Muon_dz[muonIdx]/Muon_dzErr[muonIdx])
        features_.append(Muon_ip3d[muonIdx])
        features_.append(Muon_sip3d[muonIdx])
        
        #features_.append(Muon_looseId[muonIdx])
        #features_.append(Muon_mediumId[muonIdx])
        features_.append(Muon_tightId[muonIdx])
        #features_.append(Muon_softId[muonIdx])
        #features_.append(Muon_isPFcand[muonIdx])
        
        #features_.append(Muon_isGlobal[muonIdx])
        #features_.append(Muon_isTracker[muonIdx])
        features_.append(Muon_pfRelIso03_all[muonIdx])

        features_.append(Muon_tkIsoId[muonIdx])
        features_.append(PV_npvs)
        #features_.append(Muon_triggerIdLoose[muonIdx])
        if not isMC:
            features_.append(1)
        else:
            features_.append(np.float32(hist.GetBinContent(hist.GetXaxis().FindBin(Muon_pt[muonIdx]),hist.GetYaxis().FindBin(abs(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])))))

        assert Muon_isTriggering[muonIdx]

                
        
        file_.append(features_)
    
    return file_



def saveData(isMC, nFiles, maxEntries, maxJet):

# Use Data For Bkg estimation
    
    path = ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB_20UL18",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-15To20_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-20To30_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-30To50_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-50To80_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-80To120_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-120To170_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-170To300_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-300To470_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-470To600_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-600To800_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-800To1000_MuEnrichedPt5",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/crab_QCD_Pt-1000_MuEnrichedPt5"][isMC]
    names = "**/*.root"
    fileNames = glob.glob(path+"/"+names, recursive=True)
    
        
    random.shuffle(fileNames)
    if nFiles > len(fileNames):
        print("nFiles > len(fileNames)\nSetting nFiles = len(fileNames) = %d" %len(fileNames))
        nFiles = len(fileNames)
    elif nFiles == -1:
        print("nFiles = -1\nSetting nFiles = len(fileNames) = %d" %len(fileNames))
        nFiles = len(fileNames)
    
    #print("Taking only the first %d files" %nFiles)
    #fileNames = fileNames[:nFiles]
    doneFiles = 0
    
    for fileName in fileNames:
        if doneFiles == nFiles:
            sys.exit("Reached %d files\nExit"%nFiles)  
        outFolder = ["/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/data",
                 "/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/ggHTrue",
                 ][int(bool(isMC))]
        T3Folder = ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbarHadronic",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbarSemiLeptonic",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbar2L2Nu",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt15To20",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt20To30",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt30To50",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt50To80",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt80To120",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt120To170",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt170To300",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt300To470",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt470To600",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt600To800",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt800To1000",
                    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt1000ToInf",
                    
                ][isMC]
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        outName = [ "/BParkingDataRun20181A_%s.parquet"%(fileNumber),
                    "/GluGluHToBB_%s.parquet"%(fileNumber),
                    #"/ZJetsToQQ_%s.parquet"%(fileNumber),
                    "/TTToHadronic_%s.parquet"%fileNumber,
                    "/TTToSemiLeptonic_%s.parquet"%fileNumber,
                    "/TTTo2L2Nu_%s.parquet"%fileNumber,
                    "/QCD_Pt15To20_%s.parquet"%fileNumber,
                    "/QCD_Pt20To30_%s.parquet"%fileNumber,
                    "/QCD_Pt30To50_%s.parquet"%fileNumber,
                    "/QCD_Pt50To80_%s.parquet"%fileNumber,
                    "/QCD_Pt80To120_%s.parquet"%fileNumber,
                    "/QCD_Pt120To170_%s.parquet"%fileNumber,
                    "/QCD_Pt170To300_%s.parquet"%fileNumber,
                    "/QCD_Pt300To470_%s.parquet"%fileNumber,
                    "/QCD_Pt600To800_%s.parquet"%fileNumber,
                    "/QCD_Pt470To600_%s.parquet"%fileNumber,
                    "/QCD_Pt600To800_%s.parquet"%fileNumber,
                    "/QCD_Pt1000ToInf_%s.parquet"%fileNumber,
                    "/QCD_Pt800To1000_%s.parquet"%fileNumber,
                   ][isMC]
        if (os.path.exists(T3Folder+"/others/" + outName)) | (os.path.exists(T3Folder+"/training/" + outName)) | (os.path.exists(T3Folder+ outName)):
            # if you already saved this file skip
            print("File %s already present in T3\n" %fileNumber)
            continue
        if os.path.exists(outFolder + outName):
            # if you already saved this file skip
            print("File %s already present in bbar_analysis\n" %fileNumber)
            continue
        
        print("\nOpening ", (fileNames.index(fileName)+1), "/", len(fileNames), " path:", fileName, "...")
        df = pd.DataFrame()
        try:
            df.to_csv(T3Folder+outName, index=False, header=False)
            
        except:
            T3Folder="/scratch"
            df.to_csv(outFolder+outName, index=False, header=False)
        fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, isMC=isMC)
        print("Saving entries %d" %len(fileData))
        if T3Folder=="/scratch":
            print("Trying to remove ", outFolder+outName)
            os.remove(outFolder+outName)
            with open("/t3home/gcelotto/slurm/output/files.txt", 'a') as file:
                file.write(outName[1:]+"\n")
                print("Written %s in the file"%outName[1:])
        
        df=pd.DataFrame(fileData)
        featureNames = [
'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB',
'jet2_area', 'jet2_qgl', 'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable',
'dijet_twist', 'nJets', 'ht', 'muon_pt', 'muon_eta',  'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_tkIsoId', 
'PV_npvs', 'sf']
        df.columns = featureNames
        df.to_parquet(T3Folder + outName)
        doneFiles = doneFiles +1
        #np.save(, np.array(fileData, np.float32))
        print("Saved in T3", T3Folder+ outName)




    return 0


if __name__=="__main__":
    isMC        = int(sys.argv[1]) if len(sys.argv) > 1 else 1 # 0 = Data , 1 = ggHbb, 2 = ZJets
    nFiles      = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    maxEntries  = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    maxJet      = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    saveData(isMC, nFiles, maxEntries, maxJet)


def sliceJetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB):
    # need to work on this
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    muon_mask = (np.abs(Jet_eta) <= 2.5)
    muon_mask1 = (Jet_muonIdx1 >= 0) & Muon_isTriggering[Jet_muonIdx1]
    muon_mask2 = (Jet_muonIdx2 >= 0) & Muon_isTriggering[Jet_muonIdx2]

    jetsWithMuon = np.where(muon_mask & (muon_mask1 | muon_mask2))[0]
    muonIdxs = np.where(muon_mask1[jetsWithMuon], Jet_muonIdx1[jetsWithMuon], Jet_muonIdx2[jetsWithMuon])
    assert len(muonIdxs) == len(jetsWithMuon)

    # Initialize variables
    score = -999
    selected1, selected2, muonIdx = 999, 999, 999
    for i in jetsWithMuon:
        # Create a mask for valid jets to check
        valid_jet_mask = (np.abs(Jet_eta) <= 2.5)[:jetsToCheck] & (np.arange(jetsToCheck) != i)

        # Calculate scores for all valid pairs
        currentScores = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[valid_jet_mask]

        # Find the index of the maximum score
        j = np.argmax(currentScores)
        currentScore = currentScores[j]

        # Update variables if currentScore is greater than the current score
        if currentScore > score:
            score = currentScore
            selected1, selected2 = i, valid_jet_mask[j]

            # Determine muon index based on the selected indices
            if selected1 in jetsWithMuon and selected2 in jetsWithMuon:
                muonIdx = muonIdxs[jetsWithMuon == selected1][0] if muonIdxs[jetsWithMuon == selected1][0] < muonIdxs[jetsWithMuon == selected2][0] else muonIdxs[jetsWithMuon == selected2][0]
                selected1, selected2 = [selected1, selected2] if muonIdxs[jetsWithMuon == selected1][0] < muonIdxs[jetsWithMuon == selected2][0] else [selected2, selected1]
            elif selected1 in jetsWithMuon:
                muonIdx = muonIdxs[jetsWithMuon == selected1][0]
            elif selected2 in jetsWithMuon:
                muonIdx = muonIdxs[jetsWithMuon == selected2][0]
                selected1, selected2 = [selected2, selected1]
            else:
                muonIdx = -1

    return selected1, selected2, muonIdx
