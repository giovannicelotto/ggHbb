import sys, re
import pandas as pd
import numpy as np
import ROOT
import uproot

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

        Jet_area                    = branches["Jet_area"][ev]
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]

        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]

        Jet_bReg2018                 = branches["Jet_bReg2018"][ev]

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

        Muon_tightId                = branches["Muon_tightId"][ev]
        Muon_tkIsoId                = branches["Muon_tkIsoId"][ev]

        Muon_fired_HLT_Mu10p5_IP3p5 =   branches["Muon_fired_HLT_Mu10p5_IP3p5"][ev]
        Muon_fired_HLT_Mu12_IP6 =       branches["Muon_fired_HLT_Mu12_IP6"][ev]
        Muon_fired_HLT_Mu7_IP4 =        branches["Muon_fired_HLT_Mu7_IP4"][ev]
        Muon_fired_HLT_Mu8_IP3 =        branches["Muon_fired_HLT_Mu8_IP3"][ev]
        Muon_fired_HLT_Mu8_IP5 =        branches["Muon_fired_HLT_Mu8_IP5"][ev]
        Muon_fired_HLT_Mu8_IP6 =        branches["Muon_fired_HLT_Mu8_IP6"][ev]
        Muon_fired_HLT_Mu8p5_IP3p5 =    branches["Muon_fired_HLT_Mu8p5_IP3p5"][ev]
        Muon_fired_HLT_Mu9_IP4 =        branches["Muon_fired_HLT_Mu9_IP4"][ev]
        Muon_fired_HLT_Mu9_IP5 =        branches["Muon_fired_HLT_Mu9_IP5"][ev]
        Muon_fired_HLT_Mu9_IP6 =        branches["Muon_fired_HLT_Mu9_IP6"][ev]
        PV_npvs                =        branches["PV_npvs"][ev]

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
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1]*Jet_bReg2018[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2]*Jet_bReg2018[selected2])
        dijet = jet1 + jet2

        features_.append(jet1.Pt())                         
        features_.append(Jet_eta[selected1])                
        features_.append(Jet_phi[selected1])                
        features_.append(jet1.M())                          
        features_.append(Jet_nMuons[selected1])             
        features_.append(Jet_nElectrons[selected1])         
        features_.append(Jet_btagDeepFlavB[selected1])      
        features_.append(Jet_area[selected1])
        features_.append(Jet_qgl[selected1])

        features_.append(jet2.Pt())
        features_.append(Jet_eta[selected2])
        features_.append(Jet_phi[selected2])
        features_.append(jet2.M())
        features_.append(Jet_nMuons[selected2])
        features_.append(Jet_nElectrons[selected2])
        features_.append(Jet_btagDeepFlavB[selected2])
        features_.append(Jet_area[selected2])
        features_.append(Jet_qgl[selected2])

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
        features_.append(2*((jet1.Pz()*jet2.E() - jet2.Pz()*jet1.E())/(dijet.M()*np.sqrt(dijet.M()**2+dijet.Pt()**2))))

        features_.append(nJet)
        features_.append(int(np.sum(Jet_pt>20)))
        ht = 0
        for idx in range(nJet):
            ht = ht+Jet_pt[idx]

        features_.append(ht.astype(np.float32))
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
        features_.append(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])
        features_.append(Muon_dz[muonIdx]/Muon_dzErr[muonIdx])
        features_.append(Muon_ip3d[muonIdx])
        features_.append(Muon_sip3d[muonIdx])
        features_.append(Muon_tightId[muonIdx])
        features_.append(Muon_pfRelIso03_all[muonIdx])
        features_.append(Muon_tkIsoId[muonIdx])
        features_.append(bool(Muon_fired_HLT_Mu10p5_IP3p5[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu12_IP6[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu7_IP4[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu8_IP3[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu8_IP5[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu8_IP6[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu8p5_IP3p5[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu9_IP4[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu9_IP5[muonIdx]))
        features_.append(bool(Muon_fired_HLT_Mu9_IP6[muonIdx]))
        features_.append(PV_npvs)
        if not isMC:
            features_.append(1)
        else:
            features_.append(np.float32(hist.GetBinContent(hist.GetXaxis().FindBin(Muon_pt[muonIdx]),hist.GetYaxis().FindBin(abs(Muon_dxy[muonIdx]/Muon_dxyErr[muonIdx])))))
        assert Muon_isTriggering[muonIdx]
        file_.append(features_)
    
    return file_
def main(fileName, process):
    print("FileName", fileName)
    print("Process", process)
    fileData = treeFlatten(fileName=fileName, maxEntries=-1, maxJet=4, isMC=False if process=='BParkingDataRun20181A' else True)
    df=pd.DataFrame(fileData)
    featureNames = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
                    'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB',
                    'jet2_area', 'jet2_qgl', 'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable',
                    'dijet_twist', 'dijet_cs', 'nJets', 'nJets_20GeV', 'ht', 'muon_pt', 'muon_eta',  'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_tkIsoId', 
                    'Muon_fired_HLT_Mu10p5_IP3p5',  'Muon_fired_HLT_Mu12_IP6',  'Muon_fired_HLT_Mu7_IP4',   'Muon_fired_HLT_Mu8_IP3',  'Muon_fired_HLT_Mu8_IP5',    'Muon_fired_HLT_Mu8_IP6',
                    'Muon_fired_HLT_Mu8p5_IP3p5',   'Muon_fired_HLT_Mu9_IP4',   'Muon_fired_HLT_Mu9_IP5',   'Muon_fired_HLT_Mu9_IP6',    'PV_npvs','sf']
    df.columns = featureNames
    fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
    df.to_parquet('/scratch/' +process+"_%s.parquet"%fileNumber )
    print("Saving in " + '/scratch/' +process+"_%s.parquet"%fileNumber )


if __name__ == "__main__":
    fileName = sys.argv[1]
    process = sys.argv[2] 
    
    main(fileName, process)


