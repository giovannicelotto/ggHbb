import numpy as np
import uproot
import glob
import sys
import ROOT
import os
import re
import random



'''
Givena a chosen criterion to select the candidates jets that are most likely to come from the Higgs, open the files from BParrking data and save them in flaat format
Select _nFiles_ files 
Consider the fist _maxJet_ in order of pT and choose the dijet that maximize the chosen criterion
Returns the percentage of selected pairs that match the correct pairs only when a complete matching is done.

Args:
    nFiles      = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    maxEntries  = int(sys.argv[2]) if len(sys.argv) > 1 else -1
    maxJet      = int(sys.argv[3]) if len(sys.argv) > 3 else 5

'''
def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB):
    score=-999
    selected1 = 999
    selected2 = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon = []
    for i in range(nJet): # exclude the last jet because we are looking for pairs
        if abs(Jet_eta[i])>2.5:     # exclude jets>2.5 from the jets with  muon group
            continue
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                jetsWithMuon.append(i)
                continue
        if (Jet_muonIdx2[i]>-1):
            if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                jetsWithMuon.append(i)
                continue

# Now loop over these jets as first element of the pair
    #if len(jetsWithMuon)==0:
    #    continue
    for i in jetsWithMuon:
        for j in range(0, jetsToCheck):
            if i==j:
                continue
            if abs(Jet_eta[j])>2.5:
                continue

            
            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            #jet1.SetPtEtaPhiM(Jet_pt[i]*Jet_bReg2018[i], Jet_eta[i], Jet_phi[i], Jet_mass[i])
            #jet2.SetPtEtaPhiM(Jet_pt[j]*Jet_bReg2018[j], Jet_eta[j], Jet_phi[j], Jet_mass[j])
            # massDr criterion
            #deltaPhi = jet1.Phi()-jet2.Phi()
            #deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
            #tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))

            currentScore = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[j]
            if currentScore>score:
                score=currentScore
                selected1 = min(i, j)
                selected2 = max(i, j)
    return selected1, selected2 

def treeFlatten(fileName, maxEntries, maxJet, isMC, fileNumber):
    '''Require one muon in the dijet. Choose dijets based on their bscore. save all the features of the event append them in a list'''

    f = uproot.open(fileName)
    oldTree = f['Events']
    branches = oldTree.arrays()
    maxEntries = oldTree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries))

    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    f = ROOT.TFile(histPath, "READ")
    hist = f.Get("hist_scale_factor")
    
    file = ROOT.TFile("newTree_%s.root"%fileNumber, "RECREATE")
    newTree = ROOT.TTree("Events", "variables")
    jet1_pt             = np.zeros(1, dtype=np.float32)
    jet1_eta            = np.zeros(1, dtype=np.float32)
    jet1_phi            = np.zeros(1, dtype=np.float32)
    jet1_mass           = np.zeros(1, dtype=np.float32)
    jet1_nMuons         = np.zeros(1, dtype=int)
    jet1_nElectrons     = np.zeros(1, dtype=int)    
    jet1_btagDeepFlavB  = np.zeros(1, dtype=np.float32)
    jet1_area           = np.zeros(1, dtype=np.float32)
    jet1_qgl            = np.zeros(1, dtype=np.float32)
    jet2_pt             = np.zeros(1, dtype=np.float32)
    jet2_eta            = np.zeros(1, dtype=np.float32)
    jet2_phi            = np.zeros(1, dtype=np.float32)
    jet2_mass           = np.zeros(1, dtype=np.float32)
    jet2_nMuons         = np.zeros(1, dtype=int)    
    jet2_nElectrons     = np.zeros(1, dtype=int)
    jet2_btagDeepFlavB  = np.zeros(1, dtype=np.float32)
    jet2_area           = np.zeros(1, dtype=np.float32)
    jet2_qgl            = np.zeros(1, dtype=np.float32)
    dijet_Pt            = np.zeros(1, dtype=np.float32)
    dijet_Eta           = np.zeros(1, dtype=np.float32)
    dijet_Phi           = np.zeros(1, dtype=np.float32)
    dijet_M             = np.zeros(1, dtype=np.float32)
    dijet_dR            = np.zeros(1, dtype=np.float32)
    dijet_dEta          = np.zeros(1, dtype=np.float32)
    dijet_dPhi          = np.zeros(1, dtype=np.float32)
    dijet_angVariable   = np.zeros(1, dtype=np.float32)
    dijet_twist         = np.zeros(1, dtype=np.float32)
    nJets               = np.zeros(1, dtype=int)
    ht                  = np.zeros(1, dtype=np.float32)
    max_Muon_pfIsoId    = np.zeros(1, dtype=int)
    muon_pt             = np.zeros(1, dtype=np.float32)
    muon_eta            = np.zeros(1, dtype=np.float32)
    muonJet_dR          = np.zeros(1, dtype=np.float32)
    muonOverJet_pt      = np.zeros(1, dtype=np.float32)
    muon_dxySig         = np.zeros(1, dtype=np.float32)
    muon_dzSig          = np.zeros(1, dtype=np.float32)
    muon_IP3d           = np.zeros(1, dtype=np.float32)
    muon_sIP3d          = np.zeros(1, dtype=np.float32)
    muon_tightId        = np.zeros(1, dtype=bool)
    muon_pfIsoId        = np.zeros(1, dtype=int)
    muon_tkIsoId        = np.zeros(1, dtype=int)
    sf                  = np.zeros(1, dtype=np.float32)


    newTree.Branch("jet1_pt", jet1_pt, "jet1_pt/F")
    newTree.Branch("jet1_eta", jet1_eta, "jet1_eta/F")
    newTree.Branch("jet1_phi", jet1_phi, "jet1_phi/F")
    newTree.Branch("jet1_mass", jet1_mass, "jet1_mass/F")
    newTree.Branch("jet1_nMuons", jet1_nMuons, "jet1_nMuons/I")
    newTree.Branch("jet1_nElectrons", jet1_nElectrons, "jet1_nElectrons/I")
    newTree.Branch("jet1_btagDeepFlavB", jet1_btagDeepFlavB, "jet1_btagDeepFlavB/F")
    newTree.Branch("jet1_area", jet1_area, "jet1_area/F")
    newTree.Branch("jet1_qgl", jet1_qgl, "jet1_qgl/F")
    newTree.Branch("jet2_pt", jet2_pt, "jet2_pt/F")
    newTree.Branch("jet2_eta", jet2_eta, "jet2_eta/F")
    newTree.Branch("jet2_phi", jet2_phi, "jet2_phi/F")
    newTree.Branch("jet2_mass", jet2_mass, "jet2_mass/F")
    newTree.Branch("jet2_nMuons", jet2_nMuons, "jet2_nMuons/I")
    newTree.Branch("jet2_nElectrons", jet2_nElectrons, "jet2_nElectrons/I")
    newTree.Branch("jet2_btagDeepFlavB", jet2_btagDeepFlavB, "jet2_btagDeepFlavB/F")
    newTree.Branch("jet2_area", jet2_area, "jet2_area/F")
    newTree.Branch("jet2_qgl", jet2_qgl, "jet2_qgl/F")
    newTree.Branch("dijet_Pt", dijet_Pt, "dijet_Pt/F")
    newTree.Branch("dijet_Eta", dijet_Eta, "dijet_Eta/F")
    newTree.Branch("dijet_Phi", dijet_Phi, "dijet_Phi/F")
    newTree.Branch("dijet_M", dijet_M, "dijet_M/F")
    newTree.Branch("dijet_dR", dijet_dR, "dijet_dR/F")
    newTree.Branch("dijet_dEta", dijet_dEta, "dijet_dEta/F")
    newTree.Branch("dijet_dPhi", dijet_dPhi, "dijet_dPhi/F")
    newTree.Branch("dijet_angVariable", dijet_angVariable, "dijet_twist/F")
    newTree.Branch("dijet_twist", dijet_twist, "dijet_twist/F")
    newTree.Branch("nJets", nJets, "nJets/I")
    newTree.Branch("ht", ht, "ht/F")
    newTree.Branch("max_Muon_pfIsoId", max_Muon_pfIsoId, "max_Muon_pfIsoId/I")
    newTree.Branch("muon_pt", muon_pt, "muon_pt/F")
    newTree.Branch("muon_eta", muon_eta, "muon_eta/F")
    newTree.Branch("muonJet_dR", muonJet_dR, "muonJet_dR/F")
    newTree.Branch("muonOverJet_pt", muonOverJet_pt, "muonOverJet_pt/F")
    newTree.Branch("muon_dxySig", muon_dxySig, "muon_dxySig/F")
    newTree.Branch("muon_dzSig", muon_dzSig, "muon_dzSig/F")
    newTree.Branch("muon_IP3d", muon_IP3d, "muon_IP3d/F")
    newTree.Branch("muon_sIP3d", muon_sIP3d, "muon_sIP3d/F")
    newTree.Branch("muon_tightId", muon_tightId, "Muon_tightId/I")
    newTree.Branch("muon_pfIsoId", muon_pfIsoId, "Muon_pfIsoId/O")
    newTree.Branch("muon_tkIsoId", muon_tkIsoId, "Muon_tkIsoId/O")
    newTree.Branch("sf", sf, "sf/F")


    
    for ev in  range(maxEntries):
        
        
        if (ev%(int(maxEntries/100))==0):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("%d%%"%(ev/maxEntries*100))
            sys.stdout.flush()
    
    # Reco Jets
        nJets[0]                     = branches["nJet"][ev]
        Jet_eta                     = branches["Jet_eta"][ev]
        Jet_pt                      = branches["Jet_pt"][ev]
        Jet_phi                     = branches["Jet_phi"][ev]
        Jet_mass                    = branches["Jet_mass"][ev]
        Jet_area                    = branches["Jet_area"][ev]
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
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
        Muon_ip3d                   = branches["Muon_ip3d"][ev]
        Muon_sip3d                  = branches["Muon_sip3d"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        Muon_tightId                = branches["Muon_tightId"][ev]
        Muon_tkIsoId                = branches["Muon_tkIsoId"][ev]
        Jet_bReg2018                 = branches["Jet_bReg2018"][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]




        jetsToCheck = np.min([maxJet, nJets[0]])                                 # !!! max 4 jets to check   !!!
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        

        # Selection happens HERE
        selected1, selected2, = jetsSelector(nJets[0], Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB)
        # end of selection

        if selected1==999:
            continue
        if selected2==999:
            assert False
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        dijet = jet1 + jet2
        
        
        jet1_pt[0]      =   Jet_pt[selected1]
        jet1_eta[0]     =   Jet_eta[selected1]
        jet1_phi[0]     =   Jet_phi[selected1]
        jet1_mass[0]    =   Jet_mass[selected1]
        jet1_nMuons[0]          = Jet_nMuons[selected1]
        jet1_nElectrons[0]      = Jet_nElectrons[selected1]
        jet1_btagDeepFlavB[0]   = Jet_btagDeepFlavB[selected1]
        jet1_area[0]            = Jet_area[selected1]
        jet1_qgl[0]             = Jet_qgl[selected1]
        
        jet2_pt[0]      =       Jet_pt[selected2]
        jet2_eta[0]     =       Jet_eta[selected2]
        jet2_phi[0]     =       Jet_phi[selected2]
        jet2_mass[0]    =       Jet_mass[selected2]
        jet2_nMuons[0]  =       Jet_nMuons[selected2]
        jet2_nElectrons[0]      =   Jet_nElectrons[selected2]
        jet2_btagDeepFlavB[0]   =   Jet_btagDeepFlavB[selected2]
        jet2_area[0]            =   Jet_area[selected2]
        jet2_qgl[0]             =   Jet_qgl[selected2]
        

        if dijet.Pt()<1e-5:
            assert False
        dijet_Pt[0] =   dijet.Pt()
        dijet_Eta[0] =  dijet.Eta()
        dijet_Phi[0] =  dijet.Phi()
        dijet_M[0] =    dijet.M()
        
        
        
        
        dijet_dR[0] =  jet1.DeltaR(jet2)
        dijet_dEta[0] =  abs(jet1.Eta() - jet2.Eta())
        deltaPhi = jet1.Phi()-jet2.Phi()
        deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
        dijet_dPhi[0] = abs(deltaPhi)     
        angVariable = np.pi - abs(deltaPhi) + abs(jet1.Eta() - jet2.Eta())
        dijet_angVariable[0] = angVariable
        tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
        dijet_twist[0] = (tau)

        ht[0] = 0
        for idx in range(nJets[0]):
            ht[0] = ht[0]+Jet_pt[idx]
        
        max_Muon_pfIsoId[0] = np.max(Muon_pfIsoId)
        
        muonIdx = 999
        whichJetHasLeadingTrigMuon = -1
        # First jet
        
        if Jet_muonIdx1[selected1]>-1:
            if Muon_isTriggering[Jet_muonIdx1[selected1]]:
                muonIdx = muonIdx if muonIdx < Jet_muonIdx1[selected1] else Jet_muonIdx1[selected1]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[selected1] else selected1
        if Jet_muonIdx2[selected1]>-1:
            if Muon_isTriggering[Jet_muonIdx2[selected1]]:
                muonIdx = muonIdx if muonIdx < Jet_muonIdx2[selected1] else Jet_muonIdx2[selected1]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[selected1] else selected1
        # Second jet
        if Jet_muonIdx1[selected2]>-1:
            if Muon_isTriggering[Jet_muonIdx1[selected2]]:
                muonIdx = muonIdx if muonIdx < Jet_muonIdx1[selected2] else Jet_muonIdx1[selected2]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[selected2] else selected2
        if Jet_muonIdx2[selected2]>-1:
            if Muon_isTriggering[Jet_muonIdx2[selected2]]:
                muonIdx = muonIdx if muonIdx < Jet_muonIdx2[selected2] else Jet_muonIdx2[selected2]
                whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[selected2] else selected2

        assert muonIdx!=999
        
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx], Muon_eta[muonIdx], Muon_phi[muonIdx], Muon_mass[muonIdx])

        muon_pt[0] = muon.Pt()
        muon_eta[0] = muon.Eta()
        
        if whichJetHasLeadingTrigMuon==selected1:
            muonJet_dR[0] = muon.DeltaR(jet1)
            muonOverJet_pt[0] = muon.Pt()/jet1.Pt()
        elif whichJetHasLeadingTrigMuon==selected2:
            muonJet_dR[0] = muon.DeltaR(jet1)
            muonOverJet_pt[0] = muon.Pt()/jet2.Pt()
        else:
            assert False
        
        muon_dxySig[0] = Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx]
        muon_dzSig[0] = Muon_dz[muonIdx]/Muon_dzErr[muonIdx]
        muon_IP3d[0] = Muon_ip3d[muonIdx]
        muon_sIP3d[0] = Muon_sip3d[muonIdx]
        
        #features_.append(Muon_looseId[muonIdx])
        #features_.append(Muon_mediumId[muonIdx])
        muon_tightId[0] = Muon_tightId[muonIdx]
        #features_.append(Muon_softId[muonIdx])
        #features_.append(Muon_isPFcand[muonIdx])
        
        #features_.append(Muon_isGlobal[muonIdx])
        #features_.append(Muon_isTracker[muonIdx])
        muon_pfIsoId[0] = Muon_pfIsoId[muonIdx]

        muon_tkIsoId[0] = Muon_tkIsoId[muonIdx]
        
        if not isMC:
            sf[0] = 1
        else:
            sf[0] = hist.GetBinContent(hist.GetXaxis().FindBin(Muon_pt[muonIdx]),hist.GetYaxis().FindBin(abs(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])))

        assert Muon_isTriggering[muonIdx]
        newTree.Fill()
        
    
    
    newTree.Write()
    file.Close()
    return newTree



def saveData(isMC, nFiles, maxEntries, maxJet, criterionTag):

# Use Data For Bkg estimation
    outFolderBkg = "/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/data"
    T3FolderBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatData"#/scratch"
    pathBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/000*"
    fileNamesBkg = glob.glob(pathBkg+"/Data20181A__Run2_data_2023Nov30_*.root")
    
    outFolderSignal = "/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/ggHTrue"
    T3FolderSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatData"#/scratch"
    pathSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/0000" #"/t3home/gcelotto/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test"
    fileNamesSignal = glob.glob(pathSignal+"/ggH*.root")
    
    outFolder = outFolderSignal if isMC else outFolderBkg
    fileNames = fileNamesSignal if isMC else fileNamesBkg
    T3Folder = T3FolderSignal   if isMC else T3FolderBkg
    
    #random.shuffle(fileNames)
    if nFiles > len(fileNames):
        print("nFiles > len(fileNames)\nSetting nFiles = len(fileNames) = %d" %len(fileNames))
        nFiles = len(fileNames)
    elif nFiles == -1:
        print("nFiles = -1\nSetting nFiles = len(fileNames) = %d" %len(fileNames))
        nFiles = len(fileNames)
    
    print("Taking only the first %d files" %nFiles)
    fileNames = fileNames[:nFiles]

    
    for fileName in fileNames:
        
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        outName = "/ggHbb_%s_%s.root"%(criterionTag, fileNumber) if isMC else "/BParkingDataRun20181A_%s_%s.root"%(criterionTag, fileNumber)
        #outFile = outFolder + outName
        if os.path.exists(T3Folder +outName):
            # if you already saved this file skip
            print("File %s already present in T3\n" %fileNumber)
            continue
        if os.path.exists(outFolder + outName):
            # if you already saved this file skip
            print("File %s already present in bbar_analysis\n" %fileNumber)
            continue
        
        print("\nOpening ", (fileNames.index(fileName)+1), "/", len(fileNames), " path:", fileName, "...")
        np.save(outFolder+outName, np.array([1]))  # write a dummy file so that if other jobs look for this file while treeFlatten is working they ignore it
        newTree = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, isMC=isMC, fileNumber=fileNumber)
        
        
    return 0


if __name__=="__main__":
    isMC        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nFiles      = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    maxEntries  = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    maxJet      = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    #criterionTag= (sys.argv[4]) if len(sys.argv) > 4 else criterionTag
    criterionTag = 'bScoreBased%d'%maxJet
    saveData(isMC, nFiles, maxEntries, maxJet, criterionTag)