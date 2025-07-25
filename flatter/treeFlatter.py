import sys, re
import pandas as pd
import numpy as np
import ROOT
import uproot
from functions import load_mapping_dict
import awkward as ak
from correctionlib import _core
import gzip
from getFlatFeatureNames import getFlatFeatureNames
from jetsSelector import jetsSelector
from getJetSysJEC import getJetSysJEC
import time
import math

syst2 = []

def treeFlatten(fileName, maxEntries, maxJet, pN, processName, method, isJEC):
    start_time = time.time()
    maxEntries=int(maxEntries)
    maxJet=int(maxJet)
    isJEC = int(isJEC)
    if isJEC==1:
        jec_name = '_'.join(processName.split('_')[-2:])
        print("isJEC=1. JEC name is", jec_name )
    print("fileName", fileName)
    print("maxEntries", maxEntries)
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
    #if (pN==2) | (pN==20) | (pN==21) | (pN==22) | (pN==23) | (pN==36):
    #    GenPart_pdgId = branches["GenPart_pdgId"]
    #    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"]
    #    maskBB = ak.sum((abs(GenPart_pdgId)==5) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2
    #    myrange = np.arange(tree.num_entries)[~maskBB]
#
    #elif (pN==45) | (pN==46) | (pN==47) | (pN==48) | (pN==49) | (pN==50):
    #    GenPart_pdgId = branches["GenPart_pdgId"]
    #    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"]
    #    maskBB = ak.sum((abs(GenPart_pdgId)==5) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2
    #    myrange = np.arange(tree.num_entries)[maskBB]
    #else:
    #   myrange = range(maxEntries)
    myrange = range(maxEntries)
    

    # Open the WorkingPoint correction lib
    fname = "/t3home/gcelotto/ggHbb/systematics/wpDeepJet/btv-json-sf/data/UL2018/btagging.json.gz"
    if fname.endswith(".json.gz"):
        with gzip.open(fname,'rt') as file:
            #data = json.load(file)
            data = file.read().strip()
            cset = _core.CorrectionSet.from_string(data)
    else:
        cset = _core.CorrectionSet.from_file(fname)
    corrDeepJet_FixedWP_comb = cset["deepJet_comb"]
    corrDeepJet_FixedWP_light = cset["deepJet_incl"]
    wp_converter = cset["deepJet_wp_values"]
    
    #corrDeepJet_shape           = cset["deepJet_shape"]
    #btag_systs = ['central','down','down_jes', 'down_pileup', 'down_statistic', 'down_type3', 'up', 'up_jes', 'up_pileup', 'up_statistic', 'up_type3', 'down_correlated', 'down_uncorrelated', 'up_correlated', 'up_uncorrelated']
    for ev in myrange:
        
        features_ = []
        if maxEntries>100:
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()


##############################
#
#           Branches
#
##############################

    # Reco Jets 
        nJet                        = branches["nJet"][ev]
        Jet_eta                     = branches["Jet_eta"][ev]
        Jet_pt                      = branches["Jet_pt"][ev]
        Jet_phi                     = branches["Jet_phi"][ev]
        Jet_mass                    = branches["Jet_mass"][ev]

        # ID
        Jet_jetId                   = branches["Jet_jetId"][ev]
        Jet_puId                    = branches["Jet_puId"][ev]

        # Taggers
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
        Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]
        #Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
        #Jet_PNetRegPtRawCorr        = branches["Jet_PNetRegPtRawCorr"][ev]
        #Jet_PNetRegPtRawCorrNeutrino= branches["Jet_PNetRegPtRawCorrNeutrino"][ev]
        #Jet_PNetRegPtRawRes         = branches["Jet_PNetRegPtRawRes"][ev]
        
        # Vtx
        Jet_vtx3dL                  = branches["Jet_vtx3dL"][ev]
        Jet_vtx3deL                 = branches["Jet_vtx3deL"][ev]
        Jet_vtxPt                   = branches["Jet_vtxPt"][ev]
        Jet_vtxMass                 = branches["Jet_vtxMass"][ev]
        Jet_vtxNtrk                 = branches["Jet_vtxNtrk"][ev]

        # Others
        Jet_area                    = branches["Jet_area"][ev]
        Jet_rawFactor               = branches["Jet_rawFactor"][ev]
        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nConstituents           = branches["Jet_nConstituents"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        # Regression
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
    # Electrons Tracks
        nElectron                   = branches["nElectron"][ev]
        Electron_pfRelIso           = branches["Electron_pfRelIso"][ev]

    # Triggers
        Muon_fired_HLT_Mu12_IP6 =       branches["Muon_fired_HLT_Mu12_IP6"][ev]
        Muon_fired_HLT_Mu7_IP4 =        branches["Muon_fired_HLT_Mu7_IP4"][ev]
        Muon_fired_HLT_Mu8_IP3 =        branches["Muon_fired_HLT_Mu8_IP3"][ev]
        Muon_fired_HLT_Mu8_IP5 =        branches["Muon_fired_HLT_Mu8_IP5"][ev]
        Muon_fired_HLT_Mu8_IP6 =        branches["Muon_fired_HLT_Mu8_IP6"][ev]
        Muon_fired_HLT_Mu10p5_IP3p5 =        branches["Muon_fired_HLT_Mu10p5_IP3p5"][ev]
        Muon_fired_HLT_Mu8p5_IP3p5 =        branches["Muon_fired_HLT_Mu8p5_IP3p5"][ev]
        Muon_fired_HLT_Mu9_IP4 =        branches["Muon_fired_HLT_Mu9_IP4"][ev]
        Muon_fired_HLT_Mu9_IP5 =        branches["Muon_fired_HLT_Mu9_IP5"][ev]
        Muon_fired_HLT_Mu9_IP6 =        branches["Muon_fired_HLT_Mu9_IP6"][ev]
        nSV                    =        branches["nSV"][ev]
        PV_npvs                =        branches["PV_npvs"][ev]

    # Data MC dependent
        if processName[:4]=="Data":
            Pileup_nTrueInt = 0
        else:
            Jet_hadronFlavour           = branches["Jet_hadronFlavour"][ev]
            Pileup_nTrueInt         = branches["Pileup_nTrueInt"][ev]
            Jet_genJetIdx           = branches["Jet_genJetIdx"][ev]
            GenJet_hadronFlavour    = branches["GenJet_hadronFlavour"][ev]
            genWeight    = branches["genWeight"][ev]
            if isJEC:
                JEC_branch = branches["Jet_sys_%s"%(jec_name)][ev]
                Jet_pt = Jet_pt * (1 + JEC_branch)



##############################
#
#           Filling the rows
#
##############################

        maskJets = (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4)) & (Jet_pt>20) & (abs(Jet_eta)<2.5)
        jetsToCheck = np.min([maxJet, nJet])                                
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        

        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=method, Jet_pt=Jet_pt, maskJets=maskJets)

        if selected1==999:
            continue
        if selected2==999:
            assert False

        #This is wrong Jet_breg2018 has to be applied on mass as well : https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/BJetRegression
        # Correct the dataframes later
            
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1]    )
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2]    )
        dijet = jet1 + jet2

        features_.append(jet1.Pt())                         
        features_.append(Jet_eta[selected1])                
        features_.append(Jet_phi[selected1])                
        features_.append(jet1.M())                          
        features_.append(Jet_nMuons[selected1])
        features_.append(Jet_nConstituents[selected1])             
        # add jet_nmuons tight
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected1])**2 + (Muon_phi[muIdx]-Jet_phi[selected1])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_.append(counterMuTight)                 
        features_.append(Jet_nElectrons[selected1])         
        features_.append(Jet_btagDeepFlavB[selected1])
        features_.append(int(Jet_btagDeepFlavB[selected1]>0.71))
        #features_.append(Jet_btagPNetB[selected1])
        features_.append(selected1)
        features_.append(Jet_rawFactor[selected1])
        features_.append(Jet_bReg2018[selected1])
        #features_.append(Jet_PNetRegPtRawCorr[selected1])
        #features_.append(Jet_PNetRegPtRawCorrNeutrino[selected1])
        
        features_.append(Jet_jetId[selected1])
        features_.append(Jet_puId[selected1])
        features_.append(Jet_vtxPt[selected1])
        features_.append(Jet_vtxMass[selected1])
        features_.append(Jet_vtxNtrk[selected1])
        jet1_sv_3dSig = Jet_vtx3dL[selected1]/Jet_vtx3deL[selected1] if Jet_vtx3dL[selected1]!=0 else 0
        features_.append(jet1_sv_3dSig)
        


        


        features_.append(jet2.Pt())
        features_.append(Jet_eta[selected2])
        features_.append(Jet_phi[selected2])
        features_.append(jet2.M())
        features_.append(Jet_nMuons[selected2])
        features_.append(Jet_nConstituents[selected2])
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected2])**2 + (Muon_phi[muIdx]-Jet_phi[selected2])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_.append(counterMuTight)   
        features_.append(Jet_nElectrons[selected2])
        features_.append(Jet_btagDeepFlavB[selected2])
        features_.append(int(Jet_btagDeepFlavB[selected2]>0.71))
        #features_.append(Jet_btagPNetB[selected2])
        features_.append(selected2)
        features_.append(Jet_rawFactor[selected2])
        features_.append(Jet_bReg2018[selected2])
        #features_.append(Jet_PNetRegPtRawCorr[selected2])
        #features_.append(Jet_PNetRegPtRawCorrNeutrino[selected2])
        features_.append(Jet_jetId[selected2])
        features_.append(Jet_puId[selected2])
        features_.append(Jet_vtxPt[selected2])
        features_.append(Jet_vtxMass[selected2])
        features_.append(Jet_vtxNtrk[selected2])
        jet2_sv_3dSig = Jet_vtx3dL[selected2]/Jet_vtx3deL[selected2] if Jet_vtx3dL[selected2]!=0 else 0
        features_.append(jet2_sv_3dSig)


        if np.sum(maskJets)>2:
            for i in np.arange(nJet)[maskJets]:
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    selected3 = i
                    jet3.SetPtEtaPhiM(Jet_pt[selected3],Jet_eta[selected3],Jet_phi[selected3],Jet_mass[selected3])
                    features_.append(np.float32(jet3.Pt()))
                    features_.append(np.float32(Jet_eta[selected3]))
                    features_.append(np.float32(Jet_phi[selected3]))
                    features_.append(np.float32(Jet_mass[selected3]))
                    counterMuTight=0
                    for muIdx in range(len(Muon_pt)):
                        if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected3])**2 + (Muon_phi[muIdx]-Jet_phi[selected3])**2)<0.4) & (Muon_tightId[muIdx]):
                            counterMuTight=counterMuTight+1
                    features_.append(int(counterMuTight))   
                    #features_.append(np.float32(Jet_btagPNetB[selected3]))
                    features_.append(np.float32(Jet_btagDeepFlavB[selected3]))
                    features_.append(jet3.DeltaR(dijet))
                    
                    break
        else:
            selected3 = None
            features_.append(np.float32(0)) #pt
            features_.append(np.float32(0))
            features_.append(np.float32(0))
            features_.append(np.float32(0)) #mass
            features_.append(int(0))                
            #features_.append(np.float32(0))         # pnet
            features_.append(np.float32(0))         # deepjet
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

        boost_vector = - dijet.BoostVector()  # Boost to the bb system's rest frame

        jet1_rest = ROOT.TLorentzVector(jet1)  # Make a copy to boost
        jet1_rest.Boost(boost_vector)     # Boost jet1 into the rest frame

        # Step 3: Compute the cosine of the helicity angle
        # The helicity angle is the angle between jet1's momentum in the rest frame
        # and the boost direction of the bb system in the lab frame.
        cos_theta_star = jet1_rest.Vect().Dot(dijet.Vect()) / (jet1_rest.Vect().Mag() * dijet.Vect().Mag())
        features_.append(cos_theta_star)

        dijet_pTAsymmetry = (jet1.Pt() - jet2.Pt())/(jet1.Pt() + jet2.Pt())
        features_.append(dijet_pTAsymmetry)

        centrality = abs(jet1.Eta() + jet2.Eta())/2
        features_.append(centrality)


# Event Jet variables
        Jet_px_Masked = Jet_pt[maskJets] * np.cos(Jet_phi[maskJets])
        Jet_py_Masked = Jet_pt[maskJets] * np.sin(Jet_phi[maskJets])
        Jet_pz_Masked = Jet_pt[maskJets] * np.sinh(Jet_eta[maskJets])


        ptot_squared = np.sum(Jet_px_Masked**2 + Jet_py_Masked**2 + Jet_pz_Masked**2)
        S_xx = np.sum(Jet_px_Masked * Jet_px_Masked) / ptot_squared
        S_xy = np.sum(Jet_px_Masked * Jet_py_Masked) / ptot_squared
        S_xz = np.sum(Jet_px_Masked * Jet_pz_Masked) / ptot_squared
        S_yy = np.sum(Jet_py_Masked * Jet_py_Masked) / ptot_squared
        S_yz = np.sum(Jet_py_Masked * Jet_pz_Masked) / ptot_squared
        S_zz = np.sum(Jet_pz_Masked * Jet_pz_Masked) / ptot_squared

        # Step 4: Construct the symmetric S_matrix
        S_matrix = np.array([
            [S_xx, S_xy, S_xz],
            [S_xy, S_yy, S_yz],
            [S_xz, S_yz, S_zz]
        ])

        lambda1, lambda2, lambda3 = sorted(np.linalg.eigvals(S_matrix), reverse=True)

        # spherical approx lambda1 = lambda2 = lambda 3
        sphericity = 3/2*(lambda2+lambda3)
        features_.append(sphericity)

        features_.append(lambda1)
        features_.append(lambda2)
        features_.append(lambda3)

        # thrust axis
        Jet_phiT = np.linspace(0, 3.14, 500)
        Jet_pt_extended = Jet_pt[maskJets, np.newaxis]  # Shape (nJet, 1)
        Jet_phi_extended = Jet_phi[maskJets, np.newaxis]  # Shape (nJet, 1)
        T_values = np.sum(Jet_pt_extended * np.abs(np.cos(Jet_phi_extended - Jet_phiT)), axis=0)


        # Find the maximum value
        T_max = np.max(T_values)
        phiT_max = Jet_phiT[np.argmax(T_values)]
        features_.append(T_max)
        features_.append(phiT_max)




        
    # uncorrected quantities
        features_.append(Jet_pt[selected1])
        features_.append(Jet_pt[selected2])
# Event variables
# nJets
        features_.append(int(np.sum(maskJets)))
        features_.append(int(np.sum(Jet_pt[maskJets]>20)))
        features_.append(int(np.sum(Jet_pt[maskJets]>30)))
        features_.append(int(np.sum(Jet_pt[maskJets]>50)))
        
        # Loop over indices idx of masked Jets
        ht = 0
        for idx in np.arange(nJet)[maskJets]:
            ht = ht+Jet_pt[idx]
        features_.append((np.float32(ht)))
        features_.append(nMuon)
        features_.append(np.sum(Muon_pfRelIso04_all<0.15))
        features_.append(nElectron)
        #features_.append(nProbeTracks)
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.0490)))
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.2783)))
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.7100)))


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
            features_.append(bool(False))
            features_.append(np.float32(-999))
            features_.append(np.float32(-999))
            features_.append(int(-999))
            features_.append(int(-999))
            features_.append(np.float32(-999))
# Trigger
        features_.append(int(bool(Muon_fired_HLT_Mu12_IP6[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu10p5_IP3p5[muonIdx1])))
        features_.append(int(bool(Muon_fired_HLT_Mu8p5_IP3p5[muonIdx1])))
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
        if processName[:4]=="Data":
            features_.append(1)
            features_.append(1)
        else:
# Gen Info
            features_.append(Jet_hadronFlavour[selected1])
            features_.append(Jet_hadronFlavour[selected2])
            hadronFlavour3 = Jet_hadronFlavour[selected3] if selected3 is not None else -1
            features_.append(hadronFlavour3)


# BTag SF and Variations
            for syst in ["central", "up", "down"]:
                btagSF = 1
                for j in np.arange(nJet)[maskJets]:
                    # Define WP for the current jets
                    if wp_converter.evaluate("L") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("M"):
                        wp = "L"
                    elif wp_converter.evaluate("M") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("T"):
                        wp = "M"
                    elif wp_converter.evaluate("T") <= Jet_btagDeepFlavB[j]:
                        wp = "T"
                    else:
                        wp = None  
                    # wp stores the WP of the jet j
        #    # add btag systematics
                
                    if wp==None:
                        currentJet_btagSF=1
                    elif (abs(Jet_hadronFlavour[j])==4) | abs(Jet_hadronFlavour[j])==5:
                        currentJet_btagSF = corrDeepJet_FixedWP_comb.evaluate(syst, wp, abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                    elif (abs(Jet_hadronFlavour[j])==0) :
                        currentJet_btagSF = corrDeepJet_FixedWP_light.evaluate(syst, wp, abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                    else:
                        assert False
                    assert not math.isnan(currentJet_btagSF), "Nan Found in %s"%syst
                    btagSF = btagSF * currentJet_btagSF
                features_.append(btagSF)
            # End btag syst

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
            features_.append(genWeight)
        assert Muon_isTriggering[muonIdx1]
        file_.append(features_)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    return file_


def main(fileName, maxEntries, maxJet, pN, process, method, isJEC):
    print("FileName", fileName)
    print("Process", process, flush=True)
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, pN=pN, processName=process, method=method, isJEC=isJEC)
    df=pd.DataFrame(fileData)
    
    if process[:4]=="Data":
        mcLabel=False
    else:
        mcLabel = True
    print("mc label is ", mcLabel)
    smeared = False if pN <=43 else True
    featureNames = getFlatFeatureNames(mc=mcLabel, smeared=smeared)

    df.columns = featureNames
    print("Start try")
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
    print("FileNumber ", fileNumber)
    if process[:4]=='Data':
        df['PU_SF']=1
    else:

        PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0

    print('/scratch/' +process+"_%s.parquet"%fileNumber)
    df.to_parquet('/scratch/' +process+"_%s.parquet"%fileNumber )
    print("Here4")
    print("FileNumber ", fileNumber)
    print("Saving in " + '/scratch/' +process+"_%s.parquet"%fileNumber )

if __name__ == "__main__":
    fileName    = sys.argv[1]
    maxEntries  = int(sys.argv[2])
    maxJet      = int(sys.argv[3])
    pN        = int(sys.argv[4] )
    process     = sys.argv[5] 
    method = int(sys.argv[6])
    isJEC = int(sys.argv[7])
    
    main(fileName, maxEntries, maxJet, pN, process, method, isJEC)