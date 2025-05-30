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
import json

def get_efficiency(jet_pt, jet_eta, flav, eff_map_data):
    '''
    jet_pt = scalar value of pt of the jet considered
    jet_eta = scalar value of eta of the jet considered
    flav = scalar value of flav of the jet considered (0, 4, 5)
    wp = scalar value of wp of the jet considered (L, M, T)
    
    '''
    pt_bins = np.array(eff_map_data['pt_bins'])
    eta_bins = np.array(eff_map_data['eta_bins'])
    eff_map = {
        'b': np.array(eff_map_data['eff_map']['b']),
        'c': np.array(eff_map_data['eff_map']['c']),
        'light': np.array(eff_map_data['eff_map']['light'])
    }

    # From the map extract the efficiency

    # Absolute eta
    abs_eta = np.abs(jet_eta)
    # Digitize bins: returns bin indices
    pt_bin_idx = np.digitize(jet_pt, pt_bins) - 1
    eta_bin_idx = np.digitize(abs_eta, eta_bins) - 1
    
    # Clip indices to valid range (handle overflow by clipping to last bin)
    pt_bin_idx = np.clip(pt_bin_idx, 0, len(pt_bins) - 2)
    eta_bin_idx = np.clip(eta_bin_idx, 0, len(eta_bins) - 2)

    # Select flavour key for efficiency
    if flav == 5:
        flav_key = 'b'
    elif flav == 4:
        flav_key = 'c'
    else:
        flav_key = 'light'

    eff_array = eff_map[flav_key]
    eff = eff_array[pt_bin_idx, eta_bin_idx]

    return eff, flav_key


def treeFlatten(fileName, maxEntries, maxJet, pN, processName, method, isJEC, verbose):
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
    print("verbose", verbose)

    '''Require one muon in the dijet. Choose dijets based on their bscore. save all the features of the event append them in a list'''
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries), flush=True)
    file_ =[]
    

    #open the PU SF
    #df_PU = pd.read_csv("/t3home/gcelotto/ggHbb/PU_reweighting/output/pu_sfs.csv")
    # open the file for the SF
    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    triggerScaleFactor_rootFile = ROOT.TFile(histPath, "READ")
    if not triggerScaleFactor_rootFile or triggerScaleFactor_rootFile.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {histPath}")

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
    #myrange = range(7010, 7013)
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
    eff_maps_cache = {}
    for wp_ in ["L", "M", "T"]:
        with open(f"/t3home/gcelotto/ggHbb/flatter/efficiency_btag_map/json_maps/btag_efficiency_map_{processName}_{wp_}.json", 'r') as f:
            eff_maps_cache[wp_] = json.load(f)
    for ev in myrange:
        verbose and print("Event", ev)
        features_ = {}
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

        maskJets = (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)) & (Jet_pt>=20) & (abs(Jet_eta)<2.5)
        jetsToCheck = np.min([maxJet, nJet])                                
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=method, Jet_pt=Jet_pt, maskJets=maskJets)
        verbose and print("Ev : %d | %d %d %d %d"%(ev, selected1, selected2, muonIdx1, muonIdx2))

        if selected1==999:
            continue
        if selected2==999:
            assert False

        #This is wrong Jet_breg2018 has to be applied on mass as well : https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/BJetRegression
        # Correct the dataframes later
            
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], Jet_mass[selected1]    )
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], Jet_mass[selected2]    )
        dijet = jet1 + jet2

        features_['jet1_pt']=jet1.Pt()
        features_['jet1_eta']=Jet_eta[selected1]
        features_['jet1_phi']=Jet_phi[selected1]
        features_['jet1_mass']=jet1.M()
        features_['jet1_nMuons']=Jet_nMuons[selected1]
        features_['jet1_nConstituents']=Jet_nConstituents[selected1]
        # add jet_nmuons tight
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected1])**2 + (Muon_phi[muIdx]-Jet_phi[selected1])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_['jet1_nTightMuons'] = counterMuTight       
        features_['jet1_nElectrons'] = Jet_nElectrons[selected1]     
        features_['jet1_btagDeepFlavB'] = Jet_btagDeepFlavB[selected1]
        features_['jet1_btagTight'] = int(Jet_btagDeepFlavB[selected1]>=0.71)
        features_['jet1_idx'] = selected1
        features_['jet1_rawFactor'] = Jet_rawFactor[selected1]
        features_['jet1_bReg2018'] = Jet_bReg2018[selected1]
        
        features_['jet1_id'] = Jet_jetId[selected1]
        features_['jet1_puId'] = Jet_puId[selected1]
        features_['jet1_sv_pt'] = Jet_vtxPt[selected1]
        features_['jet1_sv_mass'] = Jet_vtxMass[selected1]
        features_['jet1_sv_Ntrk'] = Jet_vtxNtrk[selected1]
        jet1_sv_3dSig = Jet_vtx3dL[selected1]/Jet_vtx3deL[selected1] if Jet_vtx3dL[selected1]!=0 else 0
        features_['jet1_sv_3dSig'] = jet1_sv_3dSig
        


        


        features_['jet2_pt'] = jet2.Pt()
        features_['jet2_eta'] = Jet_eta[selected2]
        features_['jet2_phi'] = Jet_phi[selected2]
        features_['jet2_mass'] = jet2.M()
        features_['jet2_nMuons'] = Jet_nMuons[selected2]
        features_['jet2_nConstituents'] = Jet_nConstituents[selected2]
        counterMuTight=0
        for muIdx in range(len(Muon_pt)):
            if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected2])**2 + (Muon_phi[muIdx]-Jet_phi[selected2])**2)<0.4) & (Muon_tightId[muIdx]):
                counterMuTight=counterMuTight+1
        features_['jet2_nTightMuons'] = counterMuTight
        features_['jet2_nElectrons'] = Jet_nElectrons[selected2]
        features_['jet2_btagDeepFlavB'] = Jet_btagDeepFlavB[selected2]
        features_['jet2_btagTight'] = int(Jet_btagDeepFlavB[selected2]>=0.71)
        features_['jet2_idx'] = selected2
        features_['jet2_rawFactor'] = Jet_rawFactor[selected2]
        features_['jet2_bReg2018'] = Jet_bReg2018[selected2]
        features_['jet2_id'] = Jet_jetId[selected2]
        features_['jet2_puId'] = Jet_puId[selected2]
        features_['jet2_sv_pt'] = Jet_vtxPt[selected2]
        features_['jet2_sv_mass'] = Jet_vtxMass[selected2]
        features_['jet2_sv_Ntrk'] = Jet_vtxNtrk[selected2]
        jet2_sv_3dSig = Jet_vtx3dL[selected2]/Jet_vtx3deL[selected2] if Jet_vtx3dL[selected2]!=0 else 0
        features_['jet2_sv_3dSig'] = jet2_sv_3dSig


        if np.sum(maskJets)>2:
            for i in np.arange(nJet)[maskJets]:
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    selected3 = i
                    jet3.SetPtEtaPhiM(Jet_pt[selected3],Jet_eta[selected3],Jet_phi[selected3],Jet_mass[selected3])
                    features_['jet3_pt']= np.float32(jet3.Pt())
                    features_['jet3_eta']= np.float32(Jet_eta[selected3])
                    features_['jet3_phi']= np.float32(Jet_phi[selected3])
                    features_['jet3_mass']= np.float32(Jet_mass[selected3])
                    counterMuTight=0
                    for muIdx in range(len(Muon_pt)):
                        if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected3])**2 + (Muon_phi[muIdx]-Jet_phi[selected3])**2)<0.4) & (Muon_tightId[muIdx]):
                            counterMuTight=counterMuTight+1
                    features_['jet3_nTightMuons']= int(counterMuTight)
                    features_['jet3_btagDeepFlavB']= np.float32(Jet_btagDeepFlavB[selected3])
                    if Jet_btagDeepFlavB[selected3] < 0.049:
                        jet3_btagWP = 0
                    elif Jet_btagDeepFlavB[selected3] < 0.2783:
                        jet3_btagWP = 1
                    elif Jet_btagDeepFlavB[selected3] < 0.71:
                        jet3_btagWP = 2
                    else:
                        jet3_btagWP = 3
                    features_['jet3_btagWP'] = int(jet3_btagWP)
                    features_['dR_jet3_dijet']= jet3.DeltaR(dijet)
                    
                    break
        else:
            selected3 = None
            features_['jet3_pt'] = np.float32(0)
            features_['jet3_eta'] = np.float32(0)
            features_['jet3_phi'] = np.float32(0)
            features_['jet3_mass'] = np.float32(0)
            features_['jet3_nTightMuons'] = int(0)
            features_['jet3_btagDeepFlavB'] = np.float32(0)
            features_['jet3_btagWP'] = int(0)
            features_['dR_jet3_dijet'] = np.float32(0)


# Dijet
        if dijet.Pt()<1e-5:
            assert False
        features_['dijet_pt'] = np.float32(dijet.Pt())
        features_['dijet_eta'] = np.float32(dijet.Eta())
        features_['dijet_phi'] = np.float32(dijet.Phi())
        features_['dijet_mass'] = np.float32(dijet.M())
        features_['dijet_dR'] = np.float32(jet1.DeltaR(jet2))
        features_['dijet_dEta'] = np.float32(abs(jet1.Eta() - jet2.Eta()))
        deltaPhi = jet1.Phi()-jet2.Phi()
        deltaPhi = deltaPhi - 2*np.pi*(deltaPhi >= np.pi) + 2*np.pi*(deltaPhi< -np.pi)
        features_['dijet_dPhi'] = np.float32(abs(deltaPhi))
        
        tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
        features_['dijet_twist'] = np.float32(tau)

        cs_angle = 2*((jet1.Pz()*jet2.E() - jet2.Pz()*jet1.E())/(dijet.M()*np.sqrt(dijet.M()**2+dijet.Pt()**2)))
        features_['dijet_cs'] = np.float32(cs_angle)
        features_['normalized_dijet_pt'] = dijet.Pt()/(jet1.Pt()+jet2.Pt())

        boost_vector = - dijet.BoostVector()  # Boost to the bb system's rest frame

        jet1_rest = ROOT.TLorentzVector(jet1)  # Make a copy to boost
        jet1_rest.Boost(boost_vector)     # Boost jet1 into the rest frame

        # Step 3: Compute the cosine of the helicity angle
        # The helicity angle is the angle between jet1's momentum in the rest frame
        # and the boost direction of the bb system in the lab frame.
        cos_theta_star = jet1_rest.Vect().Dot(dijet.Vect()) / (jet1_rest.Vect().Mag() * dijet.Vect().Mag())
        features_['cos_theta_star'] = cos_theta_star

        dijet_pTAsymmetry = (jet1.Pt() - jet2.Pt())/(jet1.Pt() + jet2.Pt())
        features_['dijet_pTAsymmetry'] = dijet_pTAsymmetry

        centrality = abs(jet1.Eta() + jet2.Eta())/2
        features_['centrality'] = centrality


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
        features_['sphericity'] = sphericity

        features_['lambda1'] = lambda1
        features_['lambda2'] = lambda2
        features_['lambda3'] = lambda3

        # thrust axis
        Jet_phiT = np.linspace(0, 3.14, 500)
        Jet_pt_extended = Jet_pt[maskJets, np.newaxis]  # Shape (nJet, 1)
        Jet_phi_extended = Jet_phi[maskJets, np.newaxis]  # Shape (nJet, 1)
        T_values = np.sum(Jet_pt_extended * np.abs(np.cos(Jet_phi_extended - Jet_phiT)), axis=0)


        # Find the maximum value
        T_max = np.max(T_values)
        phiT_max = Jet_phiT[np.argmax(T_values)]
        features_['T_max'] = T_max
        features_['phiT_max'] = phiT_max




        
    # uncorrected quantities
        features_['jet1_pt_uncor'] = Jet_pt[selected1]
        features_['jet2_pt_uncor'] = Jet_pt[selected2]
# Event variables
# nJets
        features_['nJets'] = int(np.sum(maskJets))
        features_['nJets_20'] = int(np.sum(Jet_pt[maskJets]>=20))
        features_['nJets_30'] = int(np.sum(Jet_pt[maskJets]>=30))
        features_['nJets_50'] = int(np.sum(Jet_pt[maskJets]>=50))
        
        # Loop over indices idx of masked Jets
        ht = 0
        for idx in np.arange(nJet)[maskJets]:
            ht = ht+Jet_pt[idx]
        features_['ht'] = (np.float32(ht))
        features_['nMuons'] = nMuon
        features_['nIsoMuons'] = np.sum(Muon_pfRelIso04_all<0.15)
        features_['nElectrons'] = nElectron
        #features_.append(nProbeTracks)
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.0490)))
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.2783)))
        #features_.append(np.sum((Jet_pt[maskJets]>20) & (Jet_btagDeepFlavB[maskJets]>0.7100)))


# SV
        features_['nSV'] = int(nSV)

# Trig Muon
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx1], Muon_eta[muonIdx1], Muon_phi[muonIdx1], Muon_mass[muonIdx1])
        features_['muon_pt'] = np.float32(muon.Pt())
        features_['muon_eta'] = np.float32(muon.Eta())
        features_['muon_ptRel'] = np.float32(muon.Perp(jet1.Vect()))
        features_['muon_dxySig'] = np.float32(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1])
        features_['muon_dzSig'] = np.float32(Muon_dz[muonIdx1]/Muon_dzErr[muonIdx1])
        features_['muon_IP3d'] = np.float32(Muon_ip3d[muonIdx1])
        features_['muon_sIP3d'] = np.float32(Muon_sip3d[muonIdx1])
        features_['muon_tightId'] = bool(Muon_tightId[muonIdx1])
        features_['muon_pfRelIso03_all'] = np.float32(Muon_pfRelIso03_all[muonIdx1])
        features_['muon_pfRelIso04_all'] = np.float32(Muon_pfRelIso04_all[muonIdx1])
        features_['muon_tkIsoId'] = int(Muon_tkIsoId[muonIdx1])
        features_['muon_charge'] = int(Muon_charge[muonIdx1])

        leptonClass = 3
        # R1
        if muonIdx2 != 999:
            leptonClass = 1
            muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
            muon2.SetPtEtaPhiM(Muon_pt[muonIdx2], Muon_eta[muonIdx2], Muon_phi[muonIdx2], Muon_mass[muonIdx2])
            features_['leptonClass'] = int(leptonClass) 
            features_['muon2_pt'] = np.float32(muon2.Pt())
            features_['muon2_eta'] = np.float32(muon2.Eta())
            features_['muon2_dxySig'] = np.float32(Muon_dxy[muonIdx2]/Muon_dxyErr[muonIdx2])
            features_['muon2_dzSig'] = np.float32(Muon_dz[muonIdx2]/Muon_dzErr[muonIdx2])
            features_['muon2_IP3d'] = np.float32(Muon_ip3d[muonIdx2])
            features_['muon2_sIP3d'] = np.float32(Muon_sip3d[muonIdx2])
            features_['muon2_tightId'] = bool(Muon_tightId[muonIdx2])
            features_['muon2_pfRelIso03_all'] = np.float32(Muon_pfRelIso03_all[muonIdx2])
            features_['muon2_pfRelIso04_all'] = np.float32(Muon_pfRelIso04_all[muonIdx2])
            features_['muon2_tkIsoId'] = int(Muon_tkIsoId[muonIdx2])
            features_['muon2_charge'] = int(Muon_charge[muonIdx2])
            features_['dimuon_mass'] = np.float32((muon+muon2).M())
        else:
            # R2 or R3
            # find leptonic charge in the second jet
            for mu in range(nMuon):
                if mu==muonIdx1:
                    # dont want the muon in the first jet
                    continue
                if (mu != Jet_muonIdx1[selected2]) & (mu != Jet_muonIdx2[selected2]):
                    # avoid all muons not inside jet2
                    continue
                else:
                    leptonClass = 2
                    muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
                    muon2.SetPtEtaPhiM(Muon_pt[mu], Muon_eta[mu], Muon_phi[mu], Muon_mass[mu])
                    features_['leptonClass'] = int(leptonClass)
                    features_['muon2_pt'] = np.float32(muon2.Pt())
                    features_['muon2_eta'] = np.float32(muon2.Eta())
                    features_['muon2_dxySig'] = np.float32(Muon_dxy[mu]/Muon_dxyErr[mu])
                    features_['muon2_dzSig'] = np.float32(Muon_dz[mu]/Muon_dzErr[mu])
                    features_['muon2_IP3d'] = np.float32(Muon_ip3d[mu])
                    features_['muon2_sIP3d'] = np.float32(Muon_sip3d[mu])
                    features_['muon2_tightId'] = bool(Muon_tightId[mu])
                    features_['muon2_pfRelIso03_all'] = np.float32(Muon_pfRelIso03_all[mu])
                    features_['muon2_pfRelIso04_all'] = np.float32(Muon_pfRelIso04_all[mu])
                    features_['muon2_tkIsoId'] = int(int(Muon_tkIsoId[mu]))
                    features_['muon2_charge'] = int(Muon_charge[mu])
                    features_['dimuon_mass'] = np.float32((muon+muon2).M())
                    break
        # R3
        if leptonClass == 3:
            features_['leptonClass'] = int(leptonClass)
            features_['muon2_pt'] = np.float32(-999)
            features_['muon2_eta'] = np.float32(-999)
            features_['muon2_dxySig'] = np.float32(-999)
            features_['muon2_dzSig'] = np.float32(-999)
            features_['muon2_IP3d'] = np.float32(-999)
            features_['muon2_sIP3d'] = np.float32(-999)
            features_['muon2_tightId'] = bool(False)
            features_['muon2_pfRelIso03_all'] = np.float32(-999)
            features_['muon2_pfRelIso04_all'] = np.float32(-999)
            features_['muon2_tkIsoId'] = int(-999)
            features_['muon2_charge'] = int(-999)
            features_['dimuon_mass'] = np.float32(-999)
# Trigger
        features_['Muon_fired_HLT_Mu12_IP6'] = int(bool(Muon_fired_HLT_Mu12_IP6[muonIdx1]))
        features_['Muon_fired_HLT_Mu10p5_IP3p5'] = int(bool(Muon_fired_HLT_Mu10p5_IP3p5[muonIdx1]))
        features_['Muon_fired_HLT_Mu8p5_IP3p5'] = int(bool(Muon_fired_HLT_Mu8p5_IP3p5[muonIdx1]))
        features_['Muon_fired_HLT_Mu7_IP4'] = int(bool(Muon_fired_HLT_Mu7_IP4[muonIdx1]))
        features_['Muon_fired_HLT_Mu8_IP3'] = int(bool(Muon_fired_HLT_Mu8_IP3[muonIdx1]))
        features_['Muon_fired_HLT_Mu8_IP5'] = int(bool(Muon_fired_HLT_Mu8_IP5[muonIdx1]))
        features_['Muon_fired_HLT_Mu8_IP6'] = int(bool(Muon_fired_HLT_Mu8_IP6[muonIdx1]))
        features_['Muon_fired_HLT_Mu9_IP4'] = int(bool(Muon_fired_HLT_Mu9_IP4[muonIdx1]))
        features_['Muon_fired_HLT_Mu9_IP5'] = int(bool(Muon_fired_HLT_Mu9_IP5[muonIdx1]))
        features_['Muon_fired_HLT_Mu9_IP6'] = int(bool(Muon_fired_HLT_Mu9_IP6[muonIdx1]))
# PV
        features_['PV_npvs'] = int(PV_npvs)
        features_['Pileup_nTrueInt'] = np.float32(Pileup_nTrueInt)
# SF
        if processName[:4]=="Data":
            features_['PV_npvs'] = 1
            features_['Pileup_nTrueInt'] = 1
        else:
# Gen Info
            features_['jet1_genHadronFlavour'] = Jet_hadronFlavour[selected1]
            features_['jet2_genHadronFlavour'] = Jet_hadronFlavour[selected2]
            hadronFlavour3 = Jet_hadronFlavour[selected3] if selected3 is not None else -1
            features_['jet3_genHadronFlavour'] = hadronFlavour3


            if (('GluGluH' in processName) | ('VBF' in processName)) & (not isJEC):
                h = ROOT.TLorentzVector(0.,0.,0.,0.)
                b_gen = ROOT.TLorentzVector(0.,0.,0.,0.)
                antib_gen = ROOT.TLorentzVector(0.,0.,0.,0.)
                GenPart_pt = branches["GenPart_pt"][ev]
                GenPart_pdgId = branches["GenPart_pdgId"][ev]
                GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
                GenPart_eta = branches["GenPart_eta"][ev]
                GenPart_phi = branches["GenPart_phi"][ev]
                GenPart_mass = branches["GenPart_mass"][ev]
                GenPart_statusFlags = branches["GenPart_statusFlags"][ev]
                mH = (GenPart_pdgId==25) & (GenPart_statusFlags>8192)
                mb_gen = (GenPart_pdgId==5) & ((GenPart_genPartIdxMother >= 0)) & (GenPart_pdgId[GenPart_genPartIdxMother]==25)
                mantib_gen = (GenPart_pdgId==-5) & ((GenPart_genPartIdxMother >= 0)) & (GenPart_pdgId[GenPart_genPartIdxMother]==25)
                b_gen.SetPtEtaPhiM(GenPart_pt[mb_gen][0], GenPart_eta[mb_gen][0], GenPart_phi[mb_gen][0], GenPart_mass[mb_gen][0])
                antib_gen.SetPtEtaPhiM(GenPart_pt[mantib_gen][0], GenPart_eta[mantib_gen][0], GenPart_phi[mantib_gen][0], GenPart_mass[mantib_gen][0])
                h.SetPtEtaPhiM(GenPart_pt[mH][0], GenPart_eta[mH][0], GenPart_phi[mH][0], GenPart_mass[mH][0])
                features_['b_gen_pt'] = b_gen.Pt()
                features_['b_gen_eta'] = b_gen.Eta()
                features_['b_gen_phi'] = b_gen.Phi()
                features_['b_gen_mass'] = b_gen.M()
                features_['antib_gen_pt'] = antib_gen.Pt()
                features_['antib_gen_eta'] = antib_gen.Eta()
                features_['antib_gen_phi'] = antib_gen.Phi()
                features_['antib_gen_mass'] = antib_gen.M()
                features_['higgs_gen_pt'] = h.Pt()
                features_['higgs_gen_eta'] = h.Eta()
                features_['higgs_gen_phi'] = h.Phi()
                features_['higgs_gen_mass'] = h.M()

                #dR and deltaPt/pt
                # Compute dR for all pairings
                dR_b_jet1 = b_gen.DeltaR(jet1)
                dR_b_jet2 = b_gen.DeltaR(jet2)
                dR_antib_jet1 = antib_gen.DeltaR(jet1)
                dR_antib_jet2 = antib_gen.DeltaR(jet2)
                
                score_1 = dR_b_jet1 + dR_antib_jet2
                score_2 = dR_b_jet2 + dR_antib_jet1

                if score_1 < score_2:


                    features_['dR_jet1_genQuark'] = b_gen.DeltaR(jet1)
                    features_['dR_jet2_genQuark'] = antib_gen.DeltaR(jet2)
                    features_['dpT_jet1_genQuark'] = (b_gen.Pt() - jet1.Pt())/b_gen.Pt() 
                    features_['dpT_jet2_genQuark'] =  (antib_gen.Pt() - jet2.Pt())/antib_gen.Pt() 

                else:
                    features_['dR_jet1_genQuark'] = antib_gen.DeltaR(jet1)
                    features_['dR_jet2_genQuark'] = b_gen.DeltaR(jet2)
                    features_['dpT_jet1_genQuark'] = (antib_gen.Pt() - jet1.Pt())/antib_gen.Pt() 
                    features_['dpT_jet2_genQuark'] = (b_gen.Pt() - jet2.Pt())/b_gen.Pt()  
                    
                    


# BTag SF and Variations
            for syst in ["central", "up", "down"]:
                # Load btag Map


                

                btagSF_event = 1.0
                for j in np.arange(nJet)[maskJets]:


                    # Extract the Scale factor
                    
                    # Extract the WP of the current jet
                    if wp_converter.evaluate("L") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("M"):
                        wp = "L"
                    elif wp_converter.evaluate("M") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("T"):
                        wp = "M"
                    elif wp_converter.evaluate("T") <= Jet_btagDeepFlavB[j]:
                        wp = "T"
                    else:
                        wp = None  
                    

                    if (abs(Jet_hadronFlavour[j])==4) | abs(Jet_hadronFlavour[j])==5:
                        currentJet_btagSF_T = corrDeepJet_FixedWP_comb.evaluate(syst, "T", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                        currentJet_btagSF_M = corrDeepJet_FixedWP_comb.evaluate(syst, "M", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                        currentJet_btagSF_L = corrDeepJet_FixedWP_comb.evaluate(syst, "L", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))

                    elif (abs(Jet_hadronFlavour[j])==0) :
                        currentJet_btagSF_T = corrDeepJet_FixedWP_light.evaluate(syst, "T", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                        currentJet_btagSF_M = corrDeepJet_FixedWP_light.evaluate(syst, "M", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))
                        currentJet_btagSF_L = corrDeepJet_FixedWP_light.evaluate(syst, "L", abs(Jet_hadronFlavour[j]), abs(jet1.Eta()), float(Jet_pt[j]))

                    eff_T, flav_key = get_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["T"])
                    eff_M, _        = get_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["M"])
                    eff_L, _        = get_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["L"])



                    if wp=="T":
                        # Jet number j is Tight. Apply simply SF
                        weight_factor = currentJet_btagSF_T
                    elif wp=="M":
                        weight_factor = (currentJet_btagSF_M * eff_M  - currentJet_btagSF_T * eff_T) / (eff_M - eff_T)
                    elif wp=="L":
                        weight_factor = (currentJet_btagSF_L * eff_L  - currentJet_btagSF_M * eff_M) / (eff_L - eff_M)
                    elif wp==None:
                        weight_factor = (1  - currentJet_btagSF_L * eff_L) / (1 - eff_L)
                        # Jet is M not T
                    btagSF_event *= weight_factor
            
                features_[f'btag_{syst}'] = btagSF_event
        
        

            # End btag syst
            hist_trigger = triggerScaleFactor_rootFile.Get("hist_scale_factor")
            xbin = hist_trigger.GetXaxis().FindBin(Muon_pt[muonIdx1])
            ybin = hist_trigger.GetYaxis().FindBin(abs(Muon_dxy[muonIdx1]/Muon_dxyErr[muonIdx1]))
            # overflow gets the same triggerSF as the last bin
            if xbin == hist_trigger.GetNbinsX()+1:
                xbin=xbin-1
            if ybin == hist_trigger.GetNbinsY()+1:
                ybin=ybin-1
            # if underflow gets the same triggerSF as the first bin
            if xbin == 0:
                xbin=1
            if ybin == 0:
                ybin=1
            features_['sf'] = np.float32(hist_trigger.GetBinContent(xbin,ybin))
            features_['genWeight'] = genWeight
        assert Muon_isTriggering[muonIdx1]
        verbose and print("Features appended")
        file_.append(features_)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    return file_


def main(fileName, maxEntries, maxJet, pN, process, method, isJEC, verbose):
    print("FileName", fileName)
    print("Process", process, flush=True)
    
    
    # Event by event operations:
    
    
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, pN=pN, processName=process, method=method, isJEC=isJEC, verbose=verbose)
    df=pd.DataFrame(fileData)
    
    
    
    
    
    
    
    
    if process[:4]=="Data":
        mcLabel=False
    else:
        mcLabel = True
    print("mc label is ", mcLabel)
    smeared = False if pN <=43 else True
    #featureNames = getFlatFxeatureNames(mc=mcLabel, smeared=smeared)

    #df.columns = featureNames
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
        df['btag_central']=1
        df['btag_up']=1
        df['btag_down']=1
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
    pN        = int(sys.argv[4])
    process     = sys.argv[5] 
    method = int(sys.argv[6])
    isJEC = int(sys.argv[7])
    verbose = int(sys.argv[8])
    print("calling main", flush=True)
    main(fileName, maxEntries, maxJet, pN, process, method, isJEC, verbose)