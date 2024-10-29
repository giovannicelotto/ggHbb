import sys, re
import pandas as pd
import numpy as np
import ROOT
import uproot
from functions import load_mapping_dict
def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId):
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
    muonIdx2 = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if (abs(Jet_eta[i])>2.5) | (Jet_puId[i]<4) | (Jet_jetId[i]<6):     # exclude jets>2.5 from the jets with  muon group
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
            if (abs(Jet_eta[j])>2.5) | (Jet_puId[j]<4) | (Jet_jetId[j]<6):
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

        Jet_jetId                   = branches["Jet_jetId"][ev]
        Jet_puId                    = branches["Jet_puId"][ev]

        Jet_area                    = branches["Jet_area"][ev]
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
        Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]
        Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
        Jet_PNetRegPtRawCorr        = branches["Jet_PNetRegPtRawCorr"][ev]
        Jet_PNetRegPtRawCorrNeutrino= branches["Jet_PNetRegPtRawCorrNeutrino"][ev]
        Jet_PNetRegPtRawRes         = branches["Jet_PNetRegPtRawRes"][ev]
        Jet_rawFactor               = branches["Jet_rawFactor"][ev]


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
        
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagPNetB, Jet_puId, Jet_jetId)

        if selected1==999:
            #print("skipped")
            continue
        if selected2==999:
            assert False

        

            
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])*Jet_PNetRegPtRawCorr[selected1]*Jet_PNetRegPtRawCorrNeutrino[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])*Jet_PNetRegPtRawCorr[selected2]*Jet_PNetRegPtRawCorrNeutrino[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        dijet = jet1 + jet2

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
        features_.append(Jet_btagPNetB[selected1])
        features_.append(selected1)
        features_.append(Jet_rawFactor[selected1])
        features_.append(Jet_PNetRegPtRawCorr[selected1])
        features_.append(Jet_PNetRegPtRawCorrNeutrino[selected1])
        
        features_.append(Jet_jetId[selected1])
        features_.append(Jet_puId[selected1])


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
        features_.append(Jet_btagPNetB[selected2])
        features_.append(selected2)
        features_.append(Jet_rawFactor[selected2])
        features_.append(Jet_PNetRegPtRawCorr[selected2])
        features_.append(Jet_PNetRegPtRawCorrNeutrino[selected2])
        features_.append(Jet_jetId[selected2])
        features_.append(Jet_puId[selected2])
        


        if len(Jet_pt)>2:
            for i in range(len(Jet_pt)):
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    jet3.SetPtEtaPhiM(Jet_pt[i]*(1-Jet_rawFactor[i]*Jet_PNetRegPtRawCorr[i]),Jet_eta[i],Jet_phi[i],Jet_mass[i])
                    features_.append(np.float32(jet3.Pt()))
                    features_.append(np.float32(Jet_eta[i]))
                    features_.append(np.float32(Jet_phi[i]))
                    features_.append(np.float32(Jet_mass[i]))
                    counterMuTight=0
                    for muIdx in range(len(Muon_pt)):
                        if (np.sqrt((Muon_eta[muIdx]-Jet_eta[i])**2 + (Muon_phi[muIdx]-Jet_phi[i])**2)<0.4) & (Muon_tightId[muIdx]):
                            counterMuTight=counterMuTight+1
                    features_.append(int(counterMuTight))   
                    features_.append(np.float32(Jet_btagPNetB[i]))
                    features_.append(jet3.DeltaR(dijet))
                    break
        else:
            features_.append(np.float32(0)) #pt
            features_.append(np.float32(0))
            features_.append(np.float32(0))
            features_.append(np.float32(0)) #mass
            features_.append(int(0))                
            features_.append(np.float32(0))         # pnet
            features_.append(np.float32(0))         # deltaR


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
        
        tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
        features_.append(np.float32(tau))

        cs_angle = 2*((jet1.Pz()*jet2.E() - jet2.Pz()*jet1.E())/(dijet.M()*np.sqrt(dijet.M()**2+dijet.Pt()**2)))
        features_.append(np.float32(cs_angle))
        features_.append(dijet.Pt()/(jet1.Pt()+jet2.Pt()))

    # uncorrected quantities
        #jet1_unc  = ROOT.TLorentzVector(0.,0.,0.,0.)
        #jet2_unc  = ROOT.TLorentzVector(0.,0.,0.,0.)
        #jet1_unc.SetPtEtaPhiM(Jet_pt[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1])
        #jet2_unc.SetPtEtaPhiM(Jet_pt[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2])
        #dijet_unc = jet1_unc + jet2_unc

        #features_.append(np.float32(jet1_unc.Pt()))
        #features_.append(np.float32(jet1_unc.M()))
        #features_.append(np.float32(jet2_unc.Pt()))
        #features_.append(np.float32(jet2_unc.M()))
        #features_.append(np.float32(dijet_unc.Pt()))
        #features_.append(np.float32(dijet_unc.M()))
# Event variables
# nJets
        features_.append(int(nJet))
        features_.append(int(np.sum(Jet_pt>20)))
        ht = 0
        for idx in range(nJet):
            ht = ht+Jet_pt[idx]
        features_.append((np.float32(ht)))
        #nJets with pTT > 30 and btag < 0.2
        nJets_30_p2 = np.sum((Jet_pt>30) & (Jet_btagPNetB<0.2) & (abs(Jet_eta)<2.5))
        ttbar_tag = (np.sum(Jet_pt>45)>=4) & (np.sum((Jet_pt>45) & (Jet_btagPNetB<0.2))<=3) & (np.sum(Jet_btagPNetB>0.9)==2)
        features_.append(nJets_30_p2)
        features_.append(ttbar_tag)
# SV
        features_.append(int(nSV))

# Trig Muon
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx1], Muon_eta[muonIdx1], Muon_phi[muonIdx1], Muon_mass[muonIdx1])
        features_.append(np.float32(muon.Pt()))
        features_.append(np.float32(muon.Eta()))
        features_.append(np.float32(muon.Perp(jet1.Vect())))
        features_.append(np.float32(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1]))
        features_.append(np.float32(Muon_dz[muonIdx1]/Muon_dzErr[muonIdx1]))
        features_.append(np.float32(Muon_ip3d[muonIdx1]))
        features_.append(np.float32(Muon_sip3d[muonIdx1]))
        features_.append(bool(Muon_tightId[muonIdx1]))
        features_.append(np.float32(Muon_pfRelIso03_all[muonIdx1]))
        features_.append(np.float32(Muon_pfRelIso04_all[muonIdx1]))
        features_.append(int(Muon_tkIsoId[muonIdx1]))
        features_.append(int(Muon_charge[muonIdx1]))

        leptonClass = 3
        # R1
        if muonIdx2 != 999:
            leptonClass = 1
            muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
            muon2.SetPtEtaPhiM(Muon_pt[muonIdx2], Muon_eta[muonIdx2], Muon_phi[muonIdx2], Muon_mass[muonIdx2])
            features_.append(int(leptonClass)) #R1
            features_.append(np.float32(muon2.Pt()))
            features_.append(np.float32(muon2.Eta()))
            features_.append(np.float32(Muon_dxy[muonIdx2]/Muon_dxyErr[muonIdx2]))
            features_.append(np.float32(Muon_dz[muonIdx2]/Muon_dzErr[muonIdx2]))
            features_.append(np.float32(Muon_ip3d[muonIdx2]))
            features_.append(np.float32(Muon_sip3d[muonIdx2]))
            features_.append(bool(Muon_tightId[muonIdx2]))
            features_.append(np.float32(Muon_pfRelIso03_all[muonIdx2]))
            features_.append(np.float32(Muon_pfRelIso04_all[muonIdx2]))
            features_.append(int(Muon_tkIsoId[muonIdx2]))
            features_.append(int(Muon_charge[muonIdx2]))
            features_.append(np.float32((muon+muon2).M()))
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
                    features_.append(int(leptonClass)) #R1
                    features_.append(np.float32(muon2.Pt()))
                    features_.append(np.float32(muon2.Eta()))
                    features_.append(np.float32(Muon_dxy[mu]/Muon_dxyErr[mu]))
                    features_.append(np.float32(Muon_dz[mu]/Muon_dzErr[mu]))
                    features_.append(np.float32(Muon_ip3d[mu]))
                    features_.append(np.float32(Muon_sip3d[mu]))
                    features_.append(bool(Muon_tightId[mu]))
                    features_.append(np.float32(Muon_pfRelIso03_all[mu]))
                    features_.append(np.float32(Muon_pfRelIso04_all[mu]))
                    features_.append(int(int(Muon_tkIsoId[mu])))
                    features_.append(int(Muon_charge[mu]))
                    features_.append(np.float32((muon+muon2).M()))
                    break
        # R3
        if leptonClass == 3:
            features_.append(int(leptonClass))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(bool(-999))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(int(-999))
            features_.append(int(-999))
            features_.append(np.float32(-999))
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
        features_.append(int(PV_npvs))
        features_.append(np.float32(Pileup_nTrueInt))
        #features_.append(Muon_vx[muonIdx])
        #features_.append(Muon_vy[muonIdx])
        #features_.append(Muon_vz[muonIdx])
# SF
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
            # if underflow gets the same triggerSF as the first bin
            if xbin == 0:
                xbin=1
            if ybin == 0:
                ybin=1
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
    featureNames = [
                # Jet 1
                    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
                    'jet1_nMuons','jet1_nTightMuons','jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_btagPNetB', 'jet1_idx',
                    'jet1_rawFactor', 'jet1_PNetRegPtRawCorr', 'jet1_PNetRegPtRawCorrNeutrino',
                    'jet1_id', 'jet1_puId',
                # Jet 2
                    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
                    'jet2_nMuons', 'jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB','jet2_btagPNetB','jet2_idx',
                    'jet2_rawFactor', 'jet2_PNetRegPtRawCorr', 'jet2_PNetRegPtRawCorrNeutrino',
                    'jet2_id', 'jet2_puId',
                # Jet 3
                    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
                    'jet3_btagPNetB', 'dR_jet3_dijet',
                # Dijet
                    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
                    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 
                # Event Variables
                    'nJets', 'nJets_20GeV', 'ht', 'nJets_pt30_btag0p2', 'ttbar_tag', 'nSV',  # Error
                # Trig Muon
                    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
                    'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', 'muon_charge',
                # Trig Muon 2
                'leptonClass',
                    'muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
                    'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge',
                    'dimuon_mass',
                # Trigger Paths
                    'Muon_fired_HLT_Mu12_IP6',  'Muon_fired_HLT_Mu7_IP4',   'Muon_fired_HLT_Mu8_IP3',  'Muon_fired_HLT_Mu8_IP5',    'Muon_fired_HLT_Mu8_IP6',
                    'Muon_fired_HLT_Mu9_IP4',   'Muon_fired_HLT_Mu9_IP5',   'Muon_fired_HLT_Mu9_IP6',
                    'PV_npvs', 'Pileup_nTrueInt', 'sf']
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
    # PU_SF addition
    if 'Data' in process:
        df['PU_SF']=1
    else:
        PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0
    df.to_parquet('/scratch/' +process+"_%s.parquet"%fileNumber )
    print("Saving in " + '/scratch/' +process+"_%s.parquet"%fileNumber )


if __name__ == "__main__":
    fileName    = sys.argv[1]
    maxEntries  = int(sys.argv[2])
    maxJet      = int(sys.argv[3])
    isMC        = int(sys.argv[4] )
    process     = sys.argv[5] 
    
    main(fileName, maxEntries, maxJet, isMC, process)