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


def treeFlatten(fileName, maxEntries, maxJet):
    '''Require one muon in the dijet. Choose dijets based on their bscore. save all the features of the event append them in a list'''
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries))
    file_ =[]
    
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
        #Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
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


        jetsToCheck = np.min([maxJet, nJet])                                 # !!! max 4 jets to check   !!!
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        jetsToCheck = np.min([maxJet, nJet])
        score=-999

        # criterion 1:
        selected1 = 999
        selected2 = 999

# Jets With Muon is the list of jets with a muon that triggers inside their cone
        jetsWithMuon = []
        for i in range(nJet): # exclude the last jet because we are looking for pairs
            if abs(Jet_eta[i])>2.5:     # exclude jets>2.5 from the jets with  muon group
                continue
            if (Jet_muonIdx1[i]>-1): #if there is a muon
                if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                    jetsWithMuon.append(i)
                    continue
            if (Jet_muonIdx2[i]>-1):
                if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                    jetsWithMuon.append(i)
                    continue

# Now loop over these jets as first element of the pair
        if len(jetsWithMuon)==0:          
            continue
        for i in jetsWithMuon:
            for j in range(0, jetsToCheck):
                if i==j:
                    continue
                if abs(Jet_eta[j])>2.5:
                    continue

                
                jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
                jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
                jet1.SetPtEtaPhiM(Jet_pt[i]*Jet_bRegNN2[i], Jet_eta[i], Jet_phi[i], Jet_mass[i])
                jet2.SetPtEtaPhiM(Jet_pt[j]*Jet_bRegNN2[j], Jet_eta[j], Jet_phi[j], Jet_mass[j])
                # massDr criterion
                #deltaPhi = jet1.Phi()-jet2.Phi()
                #deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
                #tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))

                currentScore = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[j]
                if currentScore>score:
                    score=currentScore
                    selected1 = min(i, j)
                    selected2 = max(i, j)

        if selected1==999:
            continue
        if selected2==999:
            assert False
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bRegNN2[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bRegNN2[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        dijet = jet1 + jet2
        if (nSV>0):
            if (SV_dlenSig[0]<1):
                assert False
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
        #GC To be Uncommented GCif Jet_muonIdx1[selected1]>-1:
        #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx1[selected1]]:
        #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx1[selected1] else Jet_muonIdx1[selected1]
        #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[selected1] else selected1
        #GC To be Uncommented GCif Jet_muonIdx2[selected1]>-1:
        #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx2[selected1]]:
        #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx2[selected1] else Jet_muonIdx2[selected1]
        #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[selected1] else selected1
        #GC To be Uncommented GC# Second jet
        #GC To be Uncommented GCif Jet_muonIdx1[selected2]>-1:
        #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx1[selected2]]:
        #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx1[selected2] else Jet_muonIdx1[selected2]
        #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx1[selected2] else selected2
        #GC To be Uncommented GCif Jet_muonIdx2[selected2]>-1:
        #GC To be Uncommented GC    if Muon_isTriggering[Jet_muonIdx2[selected2]]:
        #GC To be Uncommented GC        muonIdx = muonIdx if muonIdx < Jet_muonIdx2[selected2] else Jet_muonIdx2[selected2]
        #GC To be Uncommented GC        whichJetHasLeadingTrigMuon = whichJetHasLeadingTrigMuon if muonIdx < Jet_muonIdx2[selected2] else selected2
#GC To be Uncommented GC
#GC To be Uncommented GC
        #GC To be Uncommented GC
        #GC To be Uncommented GCmuon = ROOT.TLorentzVector(0., 0., 0., 0.)
        #GC To be Uncommented GCmuon.SetPtEtaPhiM(Muon_pt[muonIdx], Muon_eta[muonIdx], Muon_phi[muonIdx], Muon_mass[muonIdx])
#GC To be Uncommented GC
#GC To be Uncommented GC
        #GC To be Uncommented GCfeatures_.append(muon.Pt())
        #GC To be Uncommented GCfeatures_.append(muon.Eta())
        #GC To be Uncommented GCif whichJetHasLeadingTrigMuon==selected1:
        #GC To be Uncommented GC    features_.append(muon.DeltaR(jet1) )
        #GC To be Uncommented GC    features_.append(muon.Pt()/jet1.Pt() )
        #GC To be Uncommented GCelif whichJetHasLeadingTrigMuon==selected2:
        #GC To be Uncommented GC    features_.append(muon.DeltaR(jet2) )
        #GC To be Uncommented GC    features_.append(muon.Pt()/jet2.Pt() )
        #GC To be Uncommented GCelse:
        #GC To be Uncommented GC    assert False
        #GC To be Uncommented GC
        #GC To be Uncommented GCfeatures_.append(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])
        #GC To be Uncommented GCfeatures_.append(Muon_dz[muonIdx]/Muon_dzErr[muonIdx])
        #GC To be Uncommented GCassert Muon_isTriggering[muonIdx]

        file_.append(features_)
    
    return file_



def saveData(nFiles, maxEntries, maxJet, criterionTag):
    signal = True

# Use Data For Bkg estimation
    outFolderBkg = "/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/data"
    T3FolderBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"
    pathBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/000*"
    fileNamesBkg = glob.glob(pathBkg+"/Data20181A_Run2_data_2023Nov0*.root")
    
    outFolderSignal = "/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/ggHTrue"
    T3FolderSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
    pathSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000"
    fileNamesSignal = glob.glob(pathSignal+"/Hbb*.root")
    
    outFolder = outFolderSignal if signal else outFolderBkg
    fileNames = fileNamesSignal if signal else fileNamesBkg
    T3Folder = T3FolderSignal   if signal else T3FolderBkg
    
    random.shuffle(fileNames)
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
        outName = "/ggHbb_%s_%s.npy"%(criterionTag, fileNumber) if signal else "/BParkingDataRun2018_1A_Cand_%s_%s.npy"%(criterionTag, fileNumber)
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
        fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet)
        print("Saving in " + outFolder+outName)
        try:
            print("Saved in T3")
            np.save(T3Folder +outName, fileData)
        except:
            print("Failed to save in T3")
            np.save(outFolder+outName, fileData)


    return 0


if __name__=="__main__":
    nFiles      = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    maxEntries  = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    maxJet      = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    #criterionTag= (sys.argv[4]) if len(sys.argv) > 4 else criterionTag
    criterionTag = 'bScoreBased%d'%maxJet
    saveData(nFiles, maxEntries, maxJet, criterionTag)