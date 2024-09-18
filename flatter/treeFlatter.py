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
    muonIdx2 = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if abs(Jet_eta[i])>2.5:     # exclude jets>2.5 from the jets with  muon group
            continue
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                if i not in jetsWithMuon:
                    jetsWithMuon.append(i)
                    muonIdxs.append(Jet_muonIdx1[i])
                    continue
        if (Jet_muonIdx2[i]>-1):
            if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                if i not in jetsWithMuon:
                    jetsWithMuon.append(i)
                    muonIdxs.append(Jet_muonIdx2[i])
                    continue
    assert len(muonIdxs)==len(jetsWithMuon)
# Now loop over these jets as first element of the pair
    # if two jets
    if len(muonIdxs)>=2:
        selected1=jetsWithMuon[0]
        selected2=jetsWithMuon[1]
        muonIdx=muonIdxs[0]
        muonIdx2=muonIdxs[1]
    elif len(muonIdxs)==0:
        selected1=999
        selected2=999
        muonIdx=999
        muonIdx2=999
    elif len(muonIdxs)==1:
        selected1 = jetsWithMuon[0]
        muonIdx = muonIdxs[0]
        for j in range(0, jetsToCheck):
            if j==jetsWithMuon[0]:
                continue
            if abs(Jet_eta[j])>2.5:
                continue
            currentScore = Jet_btagDeepFlavB[j]
            if currentScore>score:
                score=currentScore
                selected2 = j
                muonIdx2 = 999
        if selected2 == 999: # case there are not 2 jets in the acceptance set also the first to 999 to say there is no pair chosen
            selected1 = 999
    else:
        assert False
    return selected1, selected2, muonIdx, muonIdx2


def treeFlatten(fileName, maxEntries, maxJet, isMC):
    processName = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv").process[isMC]
    maxEntries=int(maxEntries)
    maxJet=int(maxJet)
    isMC=int(isMC)
    print("fileName", fileName)
    print("maxEntries", maxEntries)
    print("isMC", isMC)
    print("maxJet", maxJet)
    '''Require one muon in the dijet. Choose dijets based on their bscore. save all the features of the event append them in a list'''
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries))
    file_ =[]
    

    #open the PU SF
    #df_PU = pd.read_csv("/t3home/gcelotto/ggHbb/PU_reweighting/output/pu_sfs.csv")
    # open the file for the SF
    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    f = ROOT.TFile(histPath, "READ")
    hist = f.Get("hist_scale_factor")
    for ev in  range(maxEntries):
        
        features_ = []
        if maxEntries>100:
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
        Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]

        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]

        Jet_bReg2018                 = branches["Jet_bReg2018"][ev]

    # Muons
        nMuon                       = branches["nMuon"][ev]
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
        Muon_charge                 = branches["Muon_charge"][ev]
        Muon_tightId                = branches["Muon_tightId"][ev]
        Muon_tkIsoId                = branches["Muon_tkIsoId"][ev]
        Muon_pfRelIso04_all         = branches["Muon_pfRelIso04_all"][ev]


        Muon_fired_HLT_Mu12_IP6 =       branches["Muon_fired_HLT_Mu12_IP6"][ev]
        Muon_fired_HLT_Mu7_IP4 =        branches["Muon_fired_HLT_Mu7_IP4"][ev]
        Muon_fired_HLT_Mu8_IP3 =        branches["Muon_fired_HLT_Mu8_IP3"][ev]
        Muon_fired_HLT_Mu8_IP5 =        branches["Muon_fired_HLT_Mu8_IP5"][ev]
        Muon_fired_HLT_Mu8_IP6 =        branches["Muon_fired_HLT_Mu8_IP6"][ev]

        Muon_fired_HLT_Mu9_IP4 =        branches["Muon_fired_HLT_Mu9_IP4"][ev]
        Muon_fired_HLT_Mu9_IP5 =        branches["Muon_fired_HLT_Mu9_IP5"][ev]
        Muon_fired_HLT_Mu9_IP6 =        branches["Muon_fired_HLT_Mu9_IP6"][ev]
        nSV                    =        branches["nSV"][ev]
        PV_npvs                =        branches["PV_npvs"][ev]
        #Jet_chargeK1           =        branches["Jet_chargeK1"][ev]
        #Jet_chargeKp1          =        branches["Jet_chargeKp1"][ev]
        #Jet_chargeKp3          =        branches["Jet_chargeKp3"][ev]
        #Jet_chargeKp5          =        branches["Jet_chargeKp5"][ev]
        #Jet_chargeKp7          =        branches["Jet_chargeKp7"][ev]
        #Jet_chargeUnweighted   =        branches["Jet_chargeUnweighted"][ev]

        if (isMC!=0) & (isMC!=39):
            Pileup_nTrueInt         =       branches["Pileup_nTrueInt"][ev]
        else:
            Pileup_nTrueInt = 0

        jetsToCheck = np.min([maxJet, nJet])                                 # !!! max 4 jets to check   !!!
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        jetsToCheck = np.min([maxJet, nJet])
        
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB)

        if selected1==999:
            #print("skipped")
            continue
        if selected2==999:
            assert False

        

            
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1]*Jet_bReg2018[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2]*Jet_bReg2018[selected2])
        dijet = jet1 + jet2

        if jet2.M() < 0:
            print("Negative mass for jet2")
            print("Unorrected pt ", Jet_pt[selected2])
            print("Unorrected mass ", Jet_mass[selected2])
            print("corrected pt ", Jet_pt[selected2]*Jet_bReg2018[selected2])
            print("corrected mass ", Jet_mass[selected2]*Jet_bReg2018[selected2])
            print("computed pt ", jet2.Pt())
            print("computed mass ", jet2.M())

        features_.append(jet1.Pt())                         
        features_.append(Jet_eta[selected1])                
        features_.append(Jet_phi[selected1])                
        features_.append(jet1.M())                          
        features_.append(Jet_nMuons[selected1])             
        # add jet_nmuons tight
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected1])**2 + (Muon_phi[muIdx]-Jet_phi[selected1])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_.append(counterMuTight)                 
        features_.append(Jet_nElectrons[selected1])         
        features_.append(Jet_btagDeepFlavB[selected1])      
        features_.append(Jet_area[selected1])
        features_.append(Jet_qgl[selected1])
        features_.append(selected1)
        #features_.append(Jet_chargeK1[selected1])
        #features_.append(Jet_chargeKp1[selected1])
        #features_.append(Jet_chargeKp3[selected1])
        #features_.append(Jet_chargeKp5[selected1])
        #features_.append(Jet_chargeKp7[selected1])
        #features_.append(Jet_chargeUnweighted[selected1])

        features_.append(jet2.Pt())
        features_.append(Jet_eta[selected2])
        features_.append(Jet_phi[selected2])
        features_.append(jet2.M())
        features_.append(Jet_nMuons[selected2])
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected2])**2 + (Muon_phi[muIdx]-Jet_phi[selected2])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_.append(counterMuTight)   
        features_.append(Jet_nElectrons[selected2])
        features_.append(Jet_btagDeepFlavB[selected2])
        features_.append(Jet_area[selected2])
        features_.append(Jet_qgl[selected2])
        features_.append(selected2)
        #features_.append(Jet_chargeK1[selected2])
        #features_.append(Jet_chargeKp1[selected2])
        #features_.append(Jet_chargeKp3[selected2])
        #features_.append(Jet_chargeKp5[selected2])
        #features_.append(Jet_chargeKp7[selected2])
        #features_.append(Jet_chargeUnweighted[selected2])
        if len(Jet_pt)>2:
            for i in range(len(Jet_pt)):
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    #first jet besides the tagged ones
                    features_.append(Jet_pt[i])
                    features_.append(Jet_eta[i])
                    features_.append(Jet_phi[i])
                    features_.append(Jet_mass[i])
                    counterMuTight=0
                    for muIdx in range(len(Muon_pt)):
                        if (np.sqrt((Muon_eta[muIdx]-Jet_eta[i])**2 + (Muon_phi[muIdx]-Jet_phi[i])**2)<0.4) & (Muon_tightId[muIdx]):
                            counterMuTight=counterMuTight+1
                    features_.append(counterMuTight)   
                    features_.append(Jet_btagDeepFlavB[i])
                    features_.append(Jet_btagDeepFlavC[i])
                    features_.append(Jet_qgl[i])
                    jet3.SetPtEtaPhiM(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i])
                    features_.append(jet3.DeltaR(dijet))
                    break
        else:
            features_.append(0)
            features_.append(0)
            features_.append(0)
            features_.append(0)
            features_.append(0)
            features_.append(0)   
            features_.append(0)
            features_.append(0)
            features_.append(0)


# Dijet
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
        features_.append(dijet.Pt()/(jet1.Pt()+jet2.Pt()))
# nJets
        features_.append(nJet)
        features_.append(int(np.sum(Jet_pt>20)))
        ht = 0
        for idx in range(nJet):
            ht = ht+Jet_pt[idx]
        features_.append(ht.astype(np.float32))
# SV
        features_.append(nSV)
# Trig Muon
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx1], Muon_eta[muonIdx1], Muon_phi[muonIdx1], Muon_mass[muonIdx1])
        features_.append(np.float32(muon.Pt()))
        features_.append(np.float32(muon.Eta()))
        features_.append(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1])
        features_.append(Muon_dz[muonIdx1]/Muon_dzErr[muonIdx1])
        features_.append(Muon_ip3d[muonIdx1])
        features_.append(Muon_sip3d[muonIdx1])
        features_.append(Muon_tightId[muonIdx1])
        features_.append(Muon_pfRelIso03_all[muonIdx1])
        features_.append(Muon_pfRelIso04_all[muonIdx1])
        features_.append(int(Muon_tkIsoId[muonIdx1]))
        features_.append(Muon_charge[muonIdx1])

        leptonClass = 3
        # R1
        if muonIdx2 != 999:
            leptonClass = 1
            muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
            muon2.SetPtEtaPhiM(Muon_pt[muonIdx2], Muon_eta[muonIdx2], Muon_phi[muonIdx2], Muon_mass[muonIdx2])
            features_.append(leptonClass) #R1
            features_.append(np.float32(muon2.Pt()))
            features_.append(np.float32(muon2.Eta()))
            features_.append(Muon_dxy[muonIdx2]/Muon_dxyErr[muonIdx2])
            features_.append(Muon_dz[muonIdx2]/Muon_dzErr[muonIdx2])
            features_.append(Muon_ip3d[muonIdx2])
            features_.append(Muon_sip3d[muonIdx2])
            features_.append(Muon_tightId[muonIdx2])
            features_.append(Muon_pfRelIso03_all[muonIdx2])
            features_.append(Muon_pfRelIso04_all[muonIdx2])
            features_.append(int(Muon_tkIsoId[muonIdx2]))
            features_.append(Muon_charge[muonIdx2])
            features_.append((muon+muon2).M())
        else:
            # R2 or R3
            # find leptonic charge in the second jet
            for mu in range(nMuon):
                if mu==muonIdx1:
                    # dont want the muon in the first jet
                    continue
                if (mu != Jet_muonIdx1[selected2]) & (mu != Jet_muonIdx2[selected2]):
                    continue
                else:
                    leptonClass = 2
                    muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
                    muon2.SetPtEtaPhiM(Muon_pt[mu], Muon_eta[mu], Muon_phi[mu], Muon_mass[mu])
                    features_.append(2) #R1
                    features_.append(np.float32(muon2.Pt()))
                    features_.append(np.float32(muon2.Eta()))
                    features_.append(Muon_dxy[mu]/Muon_dxyErr[mu])
                    features_.append(Muon_dz[mu]/Muon_dzErr[mu])
                    features_.append(Muon_ip3d[mu])
                    features_.append(Muon_sip3d[mu])
                    features_.append(Muon_tightId[mu])
                    features_.append(Muon_pfRelIso03_all[mu])
                    features_.append(Muon_pfRelIso04_all[mu])
                    features_.append(int(Muon_tkIsoId[mu]))
                    features_.append(Muon_charge[mu])
                    features_.append((muon+muon2).M())
                    break
        # R3
        if leptonClass == 3:
            features_.append(leptonClass)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
            features_.append(-999)
# Trigger
        features_.append(int(bool(Muon_fired_HLT_Mu12_IP6[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu7_IP4[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu8_IP3[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu8_IP5[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu8_IP6[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu9_IP4[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu9_IP5[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu9_IP6[muonIdx1])))
# PV
        features_.append(PV_npvs)
        features_.append(Pileup_nTrueInt)
        #features_.append(Muon_vx[muonIdx])
        #features_.append(Muon_vy[muonIdx])
        #features_.append(Muon_vz[muonIdx])
# SF
        #if isMC==0:
        if 'Data' in processName:
            features_.append(1)
        else:
            xbin = hist.GetXaxis().FindBin(Muon_pt[muonIdx1])
            ybin = hist.GetYaxis().FindBin(abs(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1]))
            # overflow gets the same triggerSF as the last bin
            if xbin == hist.GetNbinsX()+1:
                xbin=xbin-1
            if ybin == hist.GetNbinsY()+1:
                ybin=ybin-1
            features_.append(np.float32(hist.GetBinContent(xbin,ybin)))
        assert Muon_isTriggering[muonIdx1]
        file_.append(features_)
    
    return file_
def main(fileName, maxEntries, maxJet, isMC, process):
    print("FileName", fileName)
    print("Process", process)
    assert maxJet==4
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, isMC=isMC)
    df=pd.DataFrame(fileData)
    featureNames = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
                    'jet1_nMuons','jet1_nTightMuons','jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl', 'jet1_idx',
                    #'jet1_chargeK1',
                    #'jet1_chargeKp1',
                    #'jet1_chargeKp3',
                    #'jet1_chargeKp5',
                    #'jet1_chargeKp7',
                    #'jet1_chargeUnweighted',
                    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons', 'jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB',
                    'jet2_area', 'jet2_qgl', 'jet2_idx',
                    #'jet2_chargeK1',
                    #'jet2_chargeKp1',
                    #'jet2_chargeKp3',
                    #'jet2_chargeKp5',
                    #'jet2_chargeKp7',
                    #'jet2_chargeUnweighted',
                    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons', 'jet3_btagDeepFlavB', 'jet3_btagDeepFlavC', 'jet3_qgl', 'dR_jet3_dijet',
                    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable',
                    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 'nJets', 'nJets_20GeV', 'ht', 'nSV',
                    'muon_pt', 'muon_eta',  'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', 'muon_charge',
                    'leptonClass',
                    'muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
                    'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge', 'dimuon_mass',
                    'Muon_fired_HLT_Mu12_IP6',  'Muon_fired_HLT_Mu7_IP4',   'Muon_fired_HLT_Mu8_IP3',  'Muon_fired_HLT_Mu8_IP5',    'Muon_fired_HLT_Mu8_IP6',
                    'Muon_fired_HLT_Mu9_IP4',   'Muon_fired_HLT_Mu9_IP5',   'Muon_fired_HLT_Mu9_IP6',    'PV_npvs', 'Pileup_nTrueInt',
                    'sf']#, #'PU_sf']
    df.columns = featureNames
    try:
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
    except:
        print("filenumber not found in ", fileName)
        try:
            fileNumber = re.search(r'200_(\d+)_Run2', fileName).group(1)
            print("This is ZJets100To200")
        except:
            sys.exit()
    df.to_parquet('/scratch/' +process+"_%s.parquet"%fileNumber )
    print("Saving in " + '/scratch/' +process+"_%s.parquet"%fileNumber )


if __name__ == "__main__":
    fileName    = sys.argv[1]
    maxEntries  = int(sys.argv[2])
    maxJet      = int(sys.argv[3])
    isMC        = int(sys.argv[4] )
    process     = sys.argv[5] 
    
    main(fileName, maxEntries, maxJet, isMC, process)