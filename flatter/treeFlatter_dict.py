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
import os
from getZ_KFactor import getZ_KFactor


def getMuonID_SF(json_data, wp_name, eta, pt):
    """
    Extract scale factor (and uncertainties) for given eta, pt.
    Handles underflow/overflow by taking the first/last bin.
    """
    wp_dict = json_data[wp_name]["abseta_pt"]
    binning = wp_dict["binning"]

    abseta_bins = binning[0]["binning"]
    pt_bins = binning[1]["binning"]

    abseta = abs(eta)

    # --- find abseta bin (with underflow/overflow handling) ---
    if abseta < abseta_bins[0]:
        lo, hi = abseta_bins[0], abseta_bins[1]
    elif abseta >= abseta_bins[-1]:
        lo, hi = abseta_bins[-2], abseta_bins[-1]
    else:
        for i in range(len(abseta_bins) - 1):
            if abseta_bins[i] <= abseta < abseta_bins[i + 1]:
                lo, hi = abseta_bins[i], abseta_bins[i + 1]
                break
    ab_key = f"abseta:[{lo},{hi}]"

    # --- find pt bin (with underflow/overflow handling) ---
    if pt < pt_bins[0]:
        plo, phi = pt_bins[0], pt_bins[1]
    elif pt >= pt_bins[-1]:
        plo, phi = pt_bins[-2], pt_bins[-1]
    else:
        for i in range(len(pt_bins) - 1):
            if pt_bins[i] <= pt < pt_bins[i + 1]:
                plo, phi = pt_bins[i], pt_bins[i + 1]
                break
    pt_key = f"pt:[{plo},{phi}]"

    # --- extract the info ---
    sf_info = wp_dict[ab_key][pt_key]

    return sf_info




def get_muon_recoSF(data, eta):
    """
    Returns (value, stat, syst) for a muon of given eta, using the medium-pT bin for all pT.
    If extrap_frac is set (e.g. 0.005 for 0.5%), it is added in quadrature to 'syst'.
    """
    sf_dict = data["NUM_TrackerMuons_DEN_genTracks"]["abseta_pt"]
    abseta_bins = sf_dict["binning"][0]["binning"]
    abseta = abs(eta)

    # find abseta bin (inclusive on lower edge, exclusive on upper)
    ab_key = None
    for lo, hi in zip(abseta_bins[:-1], abseta_bins[1:]):
        if lo <= abseta < hi:
            ab_key = f"abseta:[{lo},{hi}]"
            break
    if ab_key is None:
        # Out of range: clamp to nearest bin and (optionally) inflate syst
        if abseta < abseta_bins[0]:
            lo, hi = abseta_bins[0], abseta_bins[1]
        else:
            lo, hi = abseta_bins[-2], abseta_bins[-1]
        ab_key = f"abseta:[{lo},{hi}]"

    # always use the medium-pT bin key
    pt_key = "pt:[40,60]"

    info = sf_dict[ab_key][pt_key]
    val, stat, syst = info["value"], info["stat"], info["syst"]

    return {"value": val, "stat": stat, "syst": syst}


def get_btag_map_efficiency(jet_pt, jet_eta, flav, eff_map_data):
    '''
    jet_pt = scalar value of pt of the jet considered
    jet_eta = scalar value of eta of the jet considered
    flav = scalar value of flav of the jet considered (0, 4, 5)
    eff_map_data = scalar value of wp of the jet considered (L, M, T)
    
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


def treeFlatten(fileName, maxEntries, maxJet, pN, processName, method, isJEC, verbose, JECname, isMC):
    start_time = time.time()
    maxEntries=int(maxEntries)
    maxJet=int(maxJet)
    isJEC = int(isJEC)
    print("fileName", fileName)
    print("maxEntries", maxEntries)
    print("maxJet", maxJet)
    print("verbose", verbose)
    # Legend of Names. Examples
    # process       = complete name (GluGluHToBB_JECAbsoluteMPFBias_Down)
    # processName   = only physics name (GluGluHToBB)
    # JECname       = (JECAbsoluteMPFBias_Down)


    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries if maxEntries==-1 else maxEntries
    print("Entries : %d"%(maxEntries), flush=True)
    file_ =[]
    

    # open the file for the SF
    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    triggerScaleFactor_rootFile = ROOT.TFile(histPath, "READ")
    if not triggerScaleFactor_rootFile or triggerScaleFactor_rootFile.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {histPath}")
    
    # Open the WorkingPoint correction lib
    fname = "/t3home/gcelotto/ggHbb/systematics/wpDeepJet/btv-json-sf/data/UL2018/btagging.json.gz"
    if fname.endswith(".json.gz"):
        import gzip
        with gzip.open(fname,'rt') as file:
            #data = json.load(file)
            data = file.read().strip()
            cset = _core.CorrectionSet.from_string(data)
    else:
        cset = _core.CorrectionSet.from_file(fname)
    corrDeepJet_FixedWP_comb = cset["deepJet_comb"]
    corrDeepJet_FixedWP_light = cset["deepJet_incl"]
    wp_converter = cset["deepJet_wp_values"]

    # Open the Jet_puId Scale Factor Evaluator
    fname = "/t3home/gcelotto/ggHbb/puID_SF/jmar.json.gz"
    if fname.endswith(".json.gz"):
        
        with gzip.open(fname,'rt') as file:
            data = file.read().strip()
            puId_SF_evaluator = _core.CorrectionSet.from_string(data)
    else:
        puId_SF_evaluator = _core.CorrectionSet.from_file(fname)

    # Open the map of efficiency for btag SF
    btagMapsExist=False
    processNameForBtag = "GluGluHToBBMINLO" if processName=="GluGluHToBBMINLO_private" else processName
    if os.path.exists(f"/t3home/gcelotto/ggHbb/flatter/efficiency_btag_map/json_maps/btag_efficiency_map_{processNameForBtag}_T.json"):
        btagMapsExist=True
        eff_maps_cache = {}
        for wp_ in ["L", "M", "T"]:
            print(f"Opening the map {processName}_{wp_}.json ...")
            with open(f"/t3home/gcelotto/ggHbb/flatter/efficiency_btag_map/json_maps/btag_efficiency_map_{processNameForBtag}_{wp_}.json", 'r') as f:
                eff_maps_cache[wp_] = json.load(f)

    with open("/t3home/gcelotto/ggHbb/LeptonSF/RecoEfficiencies_MediumPtMuons.json") as f:
        muon_RECO_map = json.load(f)

    with open("/t3home/gcelotto/ggHbb/LeptonSF/IDEfficiencies_MediumPtMuons.json") as f:
        muon_ID_map = json.load(f)

    with open("/t3home/gcelotto/ggHbb/LeptonSF/ISOEfficiencies_MediumPtMuons.json") as f:
        muon_ISO_map = json.load(f)

    electrons_SF_map = _core.CorrectionSet.from_file('/t3home/gcelotto/ggHbb/LeptonSF/electron.json.gz')


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
    
        
        # Open the Pileup ID MAP
        #with open(f"/t3home/gcelotto/ggHbb/flatter/efficiency_puid_map/json_maps/puID_efficiency_map_{processName}_{wp_}.json", 'r') as f:
        #    eff_puId_map = json.lead(f)
    for ev in myrange:
        verbose and print("Event", ev)
        features_ = {}
        if maxEntries>100:
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
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
        Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]

        # ID
        Jet_jetId                   = branches["Jet_jetId"][ev]
        Jet_puId                    = branches["Jet_puId"][ev]

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
        Jet_leadTrackPt             = branches["Jet_leadTrackPt"][ev]
        Jet_nElectrons              = branches["Jet_nElectrons"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        # Regression
        Jet_bReg2018                 = branches["Jet_bReg2018"][ev]
        
        # Taggers
        #Jet_btagDeepFlavC           = branches["Jet_btagDeepFlavC"][ev]
        #Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
        #Jet_PNetRegPtRawCorr        = branches["Jet_PNetRegPtRawCorr"][ev]
        #Jet_PNetRegPtRawCorrNeutrino= branches["Jet_PNetRegPtRawCorrNeutrino"][ev]
        #Jet_PNetRegPtRawRes         = branches["Jet_PNetRegPtRawRes"][ev]
        

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
        Muon_mediumId               = branches["Muon_mediumId"][ev]
        Muon_tkIsoId                = branches["Muon_tkIsoId"][ev]
        Muon_pfRelIso04_all         = branches["Muon_pfRelIso04_all"][ev]

    # Electrons
        nElectron                   = branches["nElectron"][ev]
        Electron_pt                 = branches["Electron_pt"][ev]
        Electron_eta                = branches["Electron_eta"][ev]
        Electron_phi                = branches["Electron_phi"][ev]
        if not "private" in processName:
            Muon_genPartIdx = branches["Muon_genPartIdx"][ev]
            Electron_isPF               = branches["Electron_isPFcand"][ev]
            Electron_charge             = branches["Electron_charge"][ev]
            Electron_pfRelIso03_all     = branches["Electron_pfRelIso03_all"][ev]
            Electron_r9                 = branches["Electron_r9"][ev]
            Electron_cutBased        = branches["Electron_cutBased"][ev]
            Electron_dxy                    = branches["Electron_dxy"][ev]
            Electron_dxyErr                 = branches["Electron_dxyErr"][ev]
            Electron_mvaIso                 = branches["Electron_mvaIso"][ev]
            Electron_mvaIso_WP80            = branches["Electron_mvaIso_WP80"][ev]
            Electron_mvaIso_WP90            = branches["Electron_mvaIso_WP90"][ev]
            Electron_mvaIso_WPL             = branches["Electron_mvaIso_WPL"][ev]
            Electron_mvaNoIso_WP80          = branches["Electron_mvaNoIso_WP80"][ev]
            Electron_mvaNoIso_WP90          = branches["Electron_mvaNoIso_WP90"][ev]
            Electron_mvaNoIso_WPL           = branches["Electron_mvaNoIso_WPL"][ev]
            Electron_convVeto               = branches["Electron_convVeto"][ev]



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
        if isMC==0:
            Pileup_nTrueInt = 0
        else:
            #Gen Information
            Jet_hadronFlavour        = branches["Jet_hadronFlavour"][ev]
            Pileup_nTrueInt          = branches["Pileup_nTrueInt"][ev]
            Jet_genJetIdx            = branches["Jet_genJetIdx"][ev]
            GenJetNu_pt              = branches["GenJetNu_pt"][ev]
            GenJetNu_eta             = branches["GenJetNu_eta"][ev]
            GenJetNu_phi             = branches["GenJetNu_phi"][ev]
            GenJetNu_mass            = branches["GenJetNu_mass"][ev]
            GenJet_pt                = branches["GenJet_pt"][ev]
            GenJet_eta               = branches["GenJet_eta"][ev]
            GenJet_phi               = branches["GenJet_phi"][ev]
            GenJet_mass              = branches["GenJet_mass"][ev]
            GenJet_hadronFlavour     = branches["GenJet_hadronFlavour"][ev]
            GenJet_partonMotherPdgId = branches["GenJet_partonMotherPdgId"][ev]
            GenJet_partonMotherIdx   = branches["GenJet_partonMotherIdx"][ev]
            genWeight    = branches["genWeight"][ev]
            if (('ZJetsToQQ' in processName) | ('EWKZJets' in processName)):
                # Add NLO K-factor
                LHEPart_pt       = branches["LHEPart_pt"][ev]
                LHEPart_pdgId    = branches["LHEPart_pdgId"][ev]
            if isJEC:
                JEC_branch = branches["Jet_sys_%s"%(JECname)][ev]
                Jet_pt = Jet_pt * (1 + JEC_branch)



##############################
#
#           Filling the rows
#
##############################

        maskJets = (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>4)) & (Jet_pt>=20) & (abs(Jet_eta)<2.5)
        # puId==0 means 000: fail all PU ID;
        # puId==1 means 001: pass loose ID, fail medium, fail tight;
        # puId==3 means 011: pass loose and medium ID, fail tight;
        # puId==7 means 111: pass loose, medium, tight ID.

        jetsToCheck = np.min([maxJet, nJet])                                
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet4  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        
        # to be checked
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=method, Jet_pt=Jet_pt, maskJets=maskJets)
        verbose and print("Ev : %d | %d %d %d %d"%(ev, selected1, selected2, muonIdx1, muonIdx2))

        if selected1==999:
            continue
        if selected2==999:
            assert False
        #This is wrong Jet_breg2018 has to be applied on mass as well : https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/BJetRegression
        # Correct the dataframes later
        energy1 = np.sqrt(Jet_pt[selected1]**2 + (Jet_pt[selected1]*np.sinh(Jet_eta[selected1]))**2 +  Jet_mass[selected1]**2)
        energy2 = np.sqrt(Jet_pt[selected2]**2 + (Jet_pt[selected2]*np.sinh(Jet_eta[selected2]))**2 +  Jet_mass[selected2]**2)
        jet1.SetPtEtaPhiE(Jet_pt[selected1]*Jet_bReg2018[selected1], Jet_eta[selected1], Jet_phi[selected1], energy1*Jet_bReg2018[selected1]    )
        jet2.SetPtEtaPhiE(Jet_pt[selected2]*Jet_bReg2018[selected2], Jet_eta[selected2], Jet_phi[selected2], energy2*Jet_bReg2018[selected2]    )
        dijet = jet1 + jet2

        features_['jet1_pt']=jet1.Pt()
        features_['jet1_eta']=Jet_eta[selected1]
        features_['jet1_phi']=Jet_phi[selected1]
        features_['jet1_mass']=jet1.M()
        features_['jet1_nMuons']=Jet_nMuons[selected1]
        features_['jet1_nConstituents']=Jet_nConstituents[selected1]
        features_['jet1_leadTrackPt']=Jet_leadTrackPt[selected1]

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
        features_['jet2_leadTrackPt']=Jet_leadTrackPt[selected2]
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

        # Case of 3+ good Jets
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
                    features_['jet3_leadTrackPt']=Jet_leadTrackPt[selected3]
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
                    features_['dPhi_jet3_dijet']= jet3.DeltaPhi(dijet)
                    
                    break
        else:
            selected3 = None
            features_['jet3_pt'] = np.float32(0)
            features_['jet3_eta'] = np.float32(0)
            features_['jet3_phi'] = np.float32(0)
            features_['jet3_mass'] = np.float32(0)
            features_['jet3_leadTrackPt']=np.float32(0)
            features_['jet3_nTightMuons'] = int(0)
            features_['jet3_btagDeepFlavB'] = np.float32(0)
            features_['jet3_btagWP'] = int(0)
            features_['dR_jet3_dijet'] = np.float32(0)
            features_["dPhi_jet3_dijet"] = np.float32(0)
        
# in case of 4 jets
        if np.sum(maskJets)>3:
            for i in np.arange(nJet)[maskJets]:
                if ((i ==selected1) | (i==selected2) | (i==selected3)):
                    continue
                else:
                    selected4 = i
                    jet4.SetPtEtaPhiM(Jet_pt[selected4],Jet_eta[selected4],Jet_phi[selected4],Jet_mass[selected4])
                    features_['jet4_pt']= np.float32(jet4.Pt())
                    features_['jet4_eta']= np.float32(Jet_eta[selected4])
                    features_['jet4_phi']= np.float32(Jet_phi[selected4])
                    features_['jet4_mass']= np.float32(Jet_mass[selected4])
                    features_['jet4_leadTrackPt']=Jet_leadTrackPt[selected4]
                    counterMuTight=0
                    for muIdx in range(len(Muon_pt)):
                        if (np.sqrt((Muon_eta[muIdx]-Jet_eta[selected4])**2 + (Muon_phi[muIdx]-Jet_phi[selected4])**2)<0.4) & (Muon_tightId[muIdx]):
                            counterMuTight=counterMuTight+1
                    features_['jet4_nTightMuons']= int(counterMuTight)
                    features_['jet4_btagDeepFlavB']= np.float32(Jet_btagDeepFlavB[selected4])
                    if Jet_btagDeepFlavB[selected4] < 0.049:
                        jet4_btagWP = 0
                    elif Jet_btagDeepFlavB[selected4] < 0.2783:
                        jet4_btagWP = 1
                    elif Jet_btagDeepFlavB[selected4] < 0.71:
                        jet4_btagWP = 2
                    else:
                        jet4_btagWP = 3
                    features_['jet4_btagWP'] = int(jet4_btagWP)
                    features_['dR_jet4_dijet']= jet4.DeltaR(dijet)
                    features_['dPhi_jet4_dijet']= jet4.DeltaPhi(dijet)
                    
                    break
        else:
            selected4 = None
            features_['jet4_pt'] = np.float32(0)
            features_['jet4_eta'] = np.float32(0)
            features_['jet4_phi'] = np.float32(0)
            features_['jet4_mass'] = np.float32(0)
            features_['jet4_leadTrackPt']=np.float32(0)
            features_['jet4_nTightMuons'] = int(0)
            features_['jet4_btagDeepFlavB'] = np.float32(0)
            features_['jet4_btagWP'] = int(0)
            features_['dR_jet4_dijet'] = np.float32(0)
            features_['dPhi_jet4_dijet'] = np.float32(0)


# Dijet
        if dijet.Pt()<1e-5:
            assert False
        features_['dijet_pt'] = np.float32(dijet.Pt())
        #features_['dijet_eta_nano'] = branches["dijet_eta"][ev]
        #features_['dijet_pt_nano'] = branches["dijet_pt"][ev]
        #features_['dijet_phi_nano'] = branches["dijet_phi"][ev]
        features_['dijet_eta'] = np.float32(dijet.Eta())
        features_['dijet_phi'] = np.float32(dijet.Phi())
        features_['dijet_mass'] = np.float32(dijet.M())
        features_['dijet_dR'] = np.float32(jet1.DeltaR(jet2))
        features_['dijet_dEta'] = np.float32(abs(jet1.Eta() - jet2.Eta()))
        # Here replace
        deltaPhi = jet1.Phi()-jet2.Phi()
        deltaPhi = deltaPhi - 2*np.pi*(deltaPhi >= np.pi) + 2*np.pi*(deltaPhi< -np.pi)
        features_['dijet_dPhi'] = np.float32(abs(deltaPhi))
        # to be replaced with
        #features_["dijet_dPhi"] = np.float32(jet1.DeltaPhi(jet2))
        
        tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))
        features_['dijet_twist'] = np.float32(tau)

        cs_angle = 2*((jet1.Pz()*jet2.E() - jet2.Pz()*jet1.E())/(dijet.M()*np.sqrt(dijet.M()**2+dijet.Pt()**2)))
        features_['dijet_cs'] = np.float32(cs_angle)
        features_['normalized_dijet_pt'] = dijet.Pt()/(jet1.Pt()+jet2.Pt())

        
        # This was checked.
        # using the same vector for boosting and trasnforming you find E=m, px=0, py=0, pz=0
        boost_vector = -dijet.BoostVector()  # Boost to the bb system's rest frame

        jet1_rest = ROOT.TLorentzVector(jet1)  # Make a copy to boost
        jet1_rest.Boost(boost_vector)     # Boost jet1 into the rest frame
        jet2_rest = ROOT.TLorentzVector(jet2)  # Make a copy to boost
        jet2_rest.Boost(boost_vector)     # Boost jet1 into the rest frame
        features_['jet1_rest_pt'] = jet1_rest.Pt()
        features_['jet1_rest_eta'] = jet1_rest.Eta()
        features_['jet1_rest_phi'] = jet1_rest.Phi()
        #features_['jet1_rest_mass'] = jet1_rest.Mass() does not change under boost. It's invariant

        features_['jet2_rest_pt'] = jet2_rest.Pt()
        features_['jet2_rest_eta'] = jet2_rest.Eta()
        features_['jet2_rest_phi'] = jet2_rest.Phi()
        #features_['jet2_rest_mass'] = jet2_rest.Mass() does not change under boost. It's invariant

        # Compute the cosine of the helicity angle
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
        if lambda1>1:
            print(S_matrix)
        # Reverse so that lambda1 is always the biggest

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
## ttbar CR
    #Muon as Leading
        if not "private" in processName:

            # Select muons
            muon_mask = (Muon_pt > 25) & (np.abs(Muon_eta) < 2.4) & (Muon_pfIsoId>=4) & (Muon_tightId==1)
            electron_mask = (Electron_pt > 25) & (np.abs(Electron_eta) < 2.4)  & (Electron_cutBased>=3) #& (Electron_pfRelIso03_all<0.1)

            # At least one muon & one electron passing the cuts
            selected_muon_idx = None
            selected_ele_idx = None
            if np.any(muon_mask) and np.any(electron_mask):
                # Loop over muons & electrons to find the first opposite-charge pair
                for mu_idx in np.where(muon_mask)[0]:
                    for el_idx in np.where(electron_mask)[0]:
                        if Muon_charge[mu_idx] * Electron_charge[el_idx] < 0:
                            selected_muon_idx = mu_idx
                            selected_ele_idx = el_idx
                            break  # stop at the first matching muon
                    if selected_muon_idx is not None:
                        break

                if selected_muon_idx is not None:
                    features_["Muon_tt_pt"]         = np.float32(Muon_pt[selected_muon_idx])
                    features_["Muon_tt_eta"]        = np.float32(Muon_eta[selected_muon_idx])
                    features_["Muon_tt_phi"]        = np.float32(Muon_phi[selected_muon_idx])
                    
                    features_["Muon_tt_dxy"]            = np.float32(Muon_dxy[selected_muon_idx])
                    features_["Muon_tt_dz"]             = np.float32(Muon_dz[selected_muon_idx])
                    features_["Muon_tt_genMatched"]     = int(Muon_genPartIdx[selected_muon_idx])

                    features_["Muon_tt_charge"]     = int(Muon_charge[selected_muon_idx])
                    features_["Muon_tt_mediumId"]   = int(Muon_mediumId[selected_muon_idx])
                    features_["Muon_tt_pfIsoId"]    = int(Muon_pfIsoId[selected_muon_idx])
                    muon_tt_RECO_SF = get_muon_recoSF(muon_RECO_map, eta=Muon_eta[selected_muon_idx])
                    features_["Muon_tt_RECO_SF"] = muon_tt_RECO_SF["value"]
                    features_["Muon_tt_RECO_stat"] = muon_tt_RECO_SF["stat"]
                    features_["Muon_tt_RECO_syst"] = muon_tt_RECO_SF["syst"]
                    muon_tt_ID_SF = getMuonID_SF(muon_ID_map, "NUM_TightID_DEN_TrackerMuons", Muon_eta[selected_muon_idx], Muon_pt[selected_muon_idx])
                    features_["Muon_tt_ID_SF"] = muon_tt_ID_SF["value"]
                    features_["Muon_tt_ID_stat"] = muon_tt_ID_SF["stat"]
                    features_["Muon_tt_ID_syst"] = muon_tt_ID_SF["syst"]

                    muon_ISO_SF = getMuonID_SF(muon_ISO_map, "NUM_TightRelIso_DEN_TightIDandIPCut", Muon_eta[selected_muon_idx],  Muon_pt[selected_muon_idx])
                    features_["Muon_tt_ISO_SF"] = muon_ISO_SF["value"]
                    features_["Muon_tt_ISO_stat"] = muon_ISO_SF["stat"]
                    features_["Muon_tt_ISO_syst"] = muon_ISO_SF["syst"]



                    features_["Electron_tt_pt"]         = np.float32(Electron_pt[selected_ele_idx])
                    features_["Electron_tt_eta"]        = np.float32(Electron_eta[selected_ele_idx])
                    features_["Electron_tt_phi"]        = np.float32(Electron_phi[selected_ele_idx])
                    features_["Electron_tt_charge"]     = int(Electron_charge[selected_ele_idx])
                    features_["Electron_tt_isPF"]       = int(Electron_isPF[selected_ele_idx])
                    features_["Electron_tt_pfRelIso03_all"]   = np.float32(Electron_pfRelIso03_all[selected_ele_idx])

                    features_["Electron_tt_r9"]            = np.float32(Electron_r9[selected_ele_idx])
                    features_["Electron_tt_cutBased"]       = np.float32(Electron_cutBased[selected_ele_idx])
                    features_["Electron_tt_dxy"]            =np.float32(Electron_dxy[selected_ele_idx])
                    features_["Electron_tt_dxyErr"]            =np.float32(Electron_dxyErr[selected_ele_idx])


                    features_["Electron_tt_mvaIso"] = np.float32(Electron_mvaIso[selected_ele_idx])
                    features_["Electron_tt_mvaIso_WP80"] = np.float32(Electron_mvaIso_WP80[selected_ele_idx])
                    features_["Electron_tt_mvaIso_WP90"] = np.float32(Electron_mvaIso_WP90[selected_ele_idx])
                    features_["Electron_tt_mvaIso_WPL"] = np.float32(Electron_mvaIso_WPL[selected_ele_idx])
                    features_["Electron_tt_mvaNoIso_WP80"] = np.float32(Electron_mvaNoIso_WP80[selected_ele_idx])
                    features_["Electron_tt_mvaNoIso_WP90"] = np.float32(Electron_mvaNoIso_WP90[selected_ele_idx])
                    features_["Electron_tt_mvaNoIso_WPL"] = np.float32(Electron_mvaNoIso_WPL[selected_ele_idx])
                    features_["Electron_tt_convVeto"] = np.float32(Electron_convVeto[selected_ele_idx])

                    if Electron_pt[selected_ele_idx] >= 20 :
                        features_["Electron_tt_SF"] = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018","sf","RecoAbove20",float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))
                        features_["Electron_tt_SF_up"]  = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfup",  "RecoAbove20", float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))
                        features_["Electron_tt_SF_down"]= electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfdown","RecoAbove20", float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))
                    else:
                        features_["Electron_tt_SF"]         = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018","sf","RecoBelow20",float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))
                        features_["Electron_tt_SF_up"]      = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfup",  "RecoBelow20", float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))
                        features_["Electron_tt_SF_down"]    = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfdown","RecoBelow20", float(Electron_eta[selected_ele_idx]), float(Electron_pt[selected_ele_idx]))



                    mu = ROOT.TLorentzVector(0.,0.,0.,0.)
                    el = ROOT.TLorentzVector(0.,0.,0.,0.)
                    mu.SetPtEtaPhiM(Muon_pt[selected_muon_idx], Muon_eta[selected_muon_idx], Muon_phi[selected_muon_idx], 0.106)
                    el.SetPtEtaPhiM(Electron_pt[selected_ele_idx], Electron_eta[selected_ele_idx], Electron_phi[selected_ele_idx], 0.000511)
                    features_["dilepton_tt_mass"]   = np.float32((mu+el).M())
            if selected_muon_idx is  None:
                    features_["Muon_tt_pt"]         = np.float32(-999.)
                    features_["Muon_tt_eta"]        = np.float32(-999.)
                    features_["Muon_tt_phi"]        = np.float32(-999.)
                    features_["Muon_tt_charge"]     = int(-999)
                    features_["Muon_tt_mediumId"]   = int(-1)
                    features_["Muon_tt_pfIsoId"]    = int(-1)
                    features_["Muon_tt_dxy"]            = np.float32(-999.)
                    features_["Muon_tt_dz"]             = np.float32(-999.)
                    features_["Muon_tt_genMatched"]     = int(-1)
                    
                    features_["Muon_tt_RECO_SF"]    = np.float32(-1)
                    features_["Muon_tt_RECO_stat"]  = np.float32(-1)
                    features_["Muon_tt_RECO_syst"]  = np.float32(-1)

                    features_["Muon_tt_ID_SF"]      = np.float32(-1)
                    features_["Muon_tt_ID_stat"]    = np.float32(-1)
                    features_["Muon_tt_ID_syst"]    = np.float32(-1)

                    features_["Muon_tt_ISO_SF"]     = np.float32(-1)
                    features_["Muon_tt_ISO_stat"]   = np.float32(-1)
                    features_["Muon_tt_ISO_syst"]   = np.float32(-1)

                    features_["Electron_tt_pt"]         = np.float32(-999.)
                    features_["Electron_tt_eta"]        = np.float32(-999.)
                    features_["Electron_tt_phi"]        = np.float32(-999.)
                    features_["Electron_tt_charge"]     = int(-999)
                    features_["Electron_tt_isPF"]       = int(-1)
                    features_["Electron_tt_pfRelIso03_all"]   = np.float32(-999.)
                    features_["Electron_tt_r9"]            = np.float32(0.)
                    features_["Electron_tt_cutBased"]       = int(-1)           #  (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                    features_["Electron_tt_dxy"]            =np.float32(-999.)
                    features_["Electron_tt_dxyErr"]            =np.float32(-999.)

                    features_["Electron_tt_mvaIso"] = np.float32(-999.)
                    features_["Electron_tt_mvaIso_WP80"] = np.float32(-999.)
                    features_["Electron_tt_mvaIso_WP90"] = np.float32(-999.)
                    features_["Electron_tt_mvaIso_WPL"] = np.float32(-999.)
                    features_["Electron_tt_mvaNoIso_WP80"] = np.float32(-999.)
                    features_["Electron_tt_mvaNoIso_WP90"] = np.float32(-999.)
                    features_["Electron_tt_mvaNoIso_WPL"] = np.float32(-999.)
                    features_["Electron_tt_convVeto"] = np.float32(-999.)
                    features_["Electron_tt_SF"] = float(-1)

                    features_["Electron_tt_SF"] = float(-1)
                    features_["Electron_tt_SF_up"] = float(-1)
                    features_["Electron_tt_SF_down"] = float(-1)

                    features_["dilepton_tt_mass"]       = np.float32(0.)


#Muon1 and Muon2 for ZZ
        
        if nMuon>=2:
            muonZ1= ROOT.TLorentzVector(0., 0., 0., 0.)
            muonZ2= ROOT.TLorentzVector(0., 0., 0., 0.)
            muonZ1.SetPtEtaPhiM(Muon_pt[0], Muon_eta[0], Muon_phi[0], 0.106)
            muonZ2.SetPtEtaPhiM(Muon_pt[1], Muon_eta[1], Muon_phi[1], 0.106)
            dimuonZZ = muonZ1 + muonZ2
            features_['muonZ1_pt'] = muonZ1.Pt()
            features_['muonZ1_eta'] = muonZ1.Eta()
            features_['muonZ1_phi'] = muonZ1.Phi()
            features_['muonZ1_mass'] = 0.106
            features_['muonZ1_charge'] = Muon_charge[0]
            features_['muonZ2_pt'] = muonZ2.Pt()
            features_['muonZ2_eta'] = muonZ1.Eta()
            features_['muonZ2_phi'] = muonZ1.Phi()
            features_['muonZ2_mass'] = 0.106
            features_['muonZ2_charge'] = Muon_charge[1]

            features_['dimuonZZ_pt'] = dimuonZZ.Pt()
            features_['dimuonZZ_eta'] = dimuonZZ.Eta()
            features_['dimuonZZ_phi'] = dimuonZZ.Phi()
            features_['dimuonZZ_mass'] = dimuonZZ.M()
        else:
            features_['muonZ1_pt'] = -1
            features_['muonZ1_eta'] = -1
            features_['muonZ1_phi'] = -1
            features_['muonZ1_mass'] = -1
            features_['muonZ1_charge'] = 0
            features_['muonZ2_pt'] = -1
            features_['muonZ2_eta'] = -1
            features_['muonZ2_phi'] = -1
            features_['muonZ2_mass'] = -1
            features_['muonZ2_charge'] = 0

            features_['dimuonZZ_pt'] =      -1
            features_['dimuonZZ_eta'] =     -1
            features_['dimuonZZ_phi'] =     -1
            features_['dimuonZZ_mass'] =    -1
#
#   ***** ELECTRONS ******
# Now repeat the same for electrons
#            
#
        if nElectron>=2:
            eleZ1= ROOT.TLorentzVector(0., 0., 0., 0.)
            eleZ2= ROOT.TLorentzVector(0., 0., 0., 0.)
            eleZ1.SetPtEtaPhiM(Electron_pt[0], Electron_eta[0], Electron_phi[0], 0.000511)
            eleZ2.SetPtEtaPhiM(Electron_pt[1], Electron_eta[1], Electron_phi[1], 0.000511)
            diele = eleZ1 + eleZ2
            features_['eleZ1_pt'] = eleZ1.Pt()
            features_['eleZ1_eta'] = eleZ1.Eta()
            features_['eleZ1_phi'] = eleZ1.Phi()
            features_['eleZ1_mass'] = 0.000511
            features_['eleZ2_pt'] = eleZ2.Pt()
            features_['eleZ2_eta'] = eleZ1.Eta()
            features_['eleZ2_phi'] = eleZ1.Phi()
            features_['eleZ2_mass'] = 0.000511

            features_['dieleZZ_pt'] = diele.Pt()
            features_['dieleZZ_eta'] = diele.Eta()
            features_['dieleZZ_phi'] = diele.Phi()
            features_['dieleZZ_mass'] = diele.M()
        else:
            features_['eleZ1_pt'] = -1
            features_['eleZ1_eta'] = -1
            features_['eleZ1_phi'] = -1
            features_['eleZ1_mass'] = -1
            features_['eleZ2_pt'] = -1
            features_['eleZ2_eta'] = -1
            features_['eleZ2_phi'] = -1
            features_['eleZ2_mass'] = -1

            features_['dieleZZ_pt'] =      -1
            features_['dieleZZ_eta'] =     -1
            features_['dieleZZ_phi'] =     -1
            features_['dieleZZ_mass'] =    -1


# Trig Muon
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(Muon_pt[muonIdx1], Muon_eta[muonIdx1], Muon_phi[muonIdx1], Muon_mass[muonIdx1])
        muonTrig_RECO_ID = get_muon_recoSF(muon_RECO_map, eta=muon.Eta())
        features_["muonTrig_RECO_ID"] = np.float32(muonTrig_RECO_ID["value"])
        features_['muon_pt'] = np.float32(muon.Pt())
        features_['muon_eta'] = np.float32(muon.Eta())
        features_['muon_phi'] = np.float32(muon.Phi())
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
            muon2Trig_RECO_ID = get_muon_recoSF(muon_RECO_map, eta=Muon_eta[muonIdx2])
            features_["muon2Trig_RECO_ID"] = np.float32(muon2Trig_RECO_ID["value"])
            features_['leptonClass'] = int(leptonClass) 
            features_['muon2_pt'] = np.float32(muon2.Pt())
            features_['muon2_eta'] = np.float32(muon2.Eta())
            features_['muon2_phi'] = np.float32(muon2.Phi())
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
                    muon2Trig_RECO_ID = get_muon_recoSF(muon_RECO_map, eta=Muon_eta[mu])
                    features_["muon2Trig_RECO_ID"] = np.float32(muon2Trig_RECO_ID["value"])
                    features_['muon2_pt'] = np.float32(muon2.Pt())
                    features_['muon2_eta'] = np.float32(muon2.Eta())
                    features_['muon2_phi'] = np.float32(muon2.Phi())
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
            features_['leptonClass'] = int(leptonClass)
            features_["muon2Trig_RECO_ID"] = np.float32(0)
            features_['muon2_pt'] = np.float32(0)
            features_['muon2_eta'] = np.float32(0)
            features_['muon2_phi'] = np.float32(0)
            features_['muon2_dxySig'] = np.float32(0)
            features_['muon2_dzSig'] = np.float32(0)
            features_['muon2_IP3d'] = np.float32(0)
            features_['muon2_sIP3d'] = np.float32(0)
            features_['muon2_tightId'] = bool(False)
            features_['muon2_pfRelIso03_all'] = np.float32(0)
            features_['muon2_pfRelIso04_all'] = np.float32(0)
            features_['muon2_tkIsoId'] = int(0)
            features_['muon2_charge'] = int(0)
            features_['dimuon_mass'] = np.float32(0.106)

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
                    # b-jet1 antib-jet2


                    features_['dR_jet1_genQuark'] = b_gen.DeltaR(jet1)
                    features_['dR_jet2_genQuark'] = antib_gen.DeltaR(jet2)
                    features_['dpT_jet1_genQuark'] = (b_gen.Pt() - jet1.Pt())/b_gen.Pt() 
                    features_['dpT_jet2_genQuark'] =  (antib_gen.Pt() - jet2.Pt())/antib_gen.Pt() 
                    features_['jet1_genQuark_pt'] = b_gen.Pt()
                    features_['jet2_genQuark_pt'] = antib_gen.Pt()
                    features_['jet1_genQuarkFlavour'] =  5
                    features_['jet2_genQuarkFlavour'] =  -5

                else:
                    # b-jet2 antib-jet1
                    features_['dR_jet1_genQuark'] = antib_gen.DeltaR(jet1)
                    features_['dR_jet2_genQuark'] = b_gen.DeltaR(jet2)
                    features_['dpT_jet1_genQuark'] = (antib_gen.Pt() - jet1.Pt())/antib_gen.Pt() 
                    features_['dpT_jet2_genQuark'] = (b_gen.Pt() - jet2.Pt())/b_gen.Pt()  
                    features_['jet1_genQuark_pt'] = antib_gen.Pt()
                    features_['jet2_genQuark_pt'] = b_gen.Pt()
                    features_['jet1_genQuarkFlavour'] =  5
                    features_['jet2_genQuarkFlavour'] =  -5


                # Match with genJet
                if Jet_genJetIdx[selected1]!=-1:
                    #Matched with genjet
                    features_["GenJet1_pt"] =GenJet_pt[Jet_genJetIdx[selected1]]
                    features_["GenJet1_eta"] =GenJet_eta[Jet_genJetIdx[selected1]]
                    features_["GenJet1_phi"] =GenJet_phi[Jet_genJetIdx[selected1]]
                    features_["GenJet1_mass"] =GenJet_mass[Jet_genJetIdx[selected1]]                
                    features_["GenJet1_partonMotherPdgId"] =GenJet_partonMotherPdgId[Jet_genJetIdx[selected1]]                

                    
                    genJet = ROOT.TLorentzVector(0.,0.,0.,0.)
                    genJetNu = ROOT.TLorentzVector(0.,0.,0.,0.)
                    genJet.SetPtEtaPhiM(GenJet_pt[Jet_genJetIdx[selected1]], GenJet_eta[Jet_genJetIdx[selected1]], GenJet_phi[Jet_genJetIdx[selected1]], GenJet_mass[Jet_genJetIdx[selected1]])

                    min_dR, min_idx = 999, 999
                    for j_nu in range(len(GenJetNu_pt)):
                        genJetNu.SetPtEtaPhiM(GenJetNu_pt[j_nu], GenJetNu_eta[j_nu], GenJetNu_phi[j_nu], GenJetNu_mass[j_nu])
                        if genJetNu.DeltaR(genJet) < min_dR:
                            min_dR = genJetNu.DeltaR(genJet)
                            min_idx = j_nu
                    genJetNu.SetPtEtaPhiM(GenJetNu_pt[min_idx], GenJetNu_eta[min_idx], GenJetNu_phi[min_idx], GenJetNu_mass[min_idx])
                    features_["GenJetNu1_pt"] = GenJetNu_pt[min_idx]
                    features_["GenJetNu1_eta"] = GenJetNu_eta[min_idx]
                    features_["GenJetNu1_phi"] = GenJetNu_phi[min_idx]
                    features_["GenJetNu1_mass"] = GenJetNu_mass[min_idx]
                    features_["dR_genJet1_genJetNu1"] = genJet.DeltaR(genJetNu)
                else:
                    features_["GenJet1_pt"] =-99
                    features_["GenJet1_eta"] =-99
                    features_["GenJet1_phi"] =-99
                    features_["GenJet1_mass"] =-99
                    features_["GenJet1_partonMotherPdgId"] = -99

                    features_["GenJetNu1_pt"] = -99
                    features_["GenJetNu1_eta"] = -99
                    features_["GenJetNu1_phi"] = -99
                    features_["GenJetNu1_mass"] = -99
                    features_["dR_genJet1_genJetNu1"] = -99
                if Jet_genJetIdx[selected2]!=-1:
                    #Matched with genjet
                    features_["GenJet2_pt"] =GenJet_pt[Jet_genJetIdx[selected2]]
                    features_["GenJet2_eta"] =GenJet_eta[Jet_genJetIdx[selected2]]
                    features_["GenJet2_phi"] =GenJet_phi[Jet_genJetIdx[selected2]]
                    features_["GenJet2_mass"] =GenJet_mass[Jet_genJetIdx[selected2]]                
                    features_["GenJet2_partonMotherPdgId"] = GenJet_partonMotherPdgId[Jet_genJetIdx[selected2]]                

                    
                    genJet = ROOT.TLorentzVector(0.,0.,0.,0.)
                    genJetNu = ROOT.TLorentzVector(0.,0.,0.,0.)
                    genJet.SetPtEtaPhiM(GenJet_pt[Jet_genJetIdx[selected2]], GenJet_eta[Jet_genJetIdx[selected2]], GenJet_phi[Jet_genJetIdx[selected2]], GenJet_mass[Jet_genJetIdx[selected2]])

                    min_dR, min_idx = 999, 999
                    for j_nu in range(len(GenJetNu_pt)):
                        genJetNu.SetPtEtaPhiM(GenJetNu_pt[j_nu], GenJetNu_eta[j_nu], GenJetNu_phi[j_nu], GenJetNu_mass[j_nu])
                        if genJetNu.DeltaR(genJet) < min_dR:
                            min_dR = genJetNu.DeltaR(genJet)
                            min_idx = j_nu
                    genJetNu.SetPtEtaPhiM(GenJetNu_pt[min_idx], GenJetNu_eta[min_idx], GenJetNu_phi[min_idx], GenJetNu_mass[min_idx])
                    features_["GenJetNu2_pt"] = GenJetNu_pt[min_idx]
                    features_["GenJetNu2_eta"] = GenJetNu_eta[min_idx]
                    features_["GenJetNu2_phi"] = GenJetNu_phi[min_idx]
                    features_["GenJetNu2_mass"] = GenJetNu_mass[min_idx]
                    features_["dR_genJet2_genJetNu2"] = genJet.DeltaR(genJetNu)
                else:
                    features_["GenJet2_pt"] =-99
                    features_["GenJet2_eta"] =-99
                    features_["GenJet2_phi"] =-99
                    features_["GenJet2_mass"] =-99
                    features_["GenJet2_partonMotherPdgId"] =-99

                    features_["GenJetNu2_pt"] = -99
                    features_["GenJetNu2_eta"] = -99
                    features_["GenJetNu2_phi"] = -99
                    features_["GenJetNu2_mass"] = -99
                    features_["dR_genJet2_genJetNu2"] = -99
            if (('ZJetsToQQ' in processName) | ('EWKZJets' in processName)):
                LHEPart_pt = branches["LHEPart_pt"][ev]
                LHEPart_pdgId = branches["LHEPart_pdgId"][ev]
                k_factor = getZ_KFactor(pt=LHEPart_pt[LHEPart_pdgId==23])
                
                features_['NLO_kfactor'] = k_factor


            # PileupID SF computation
            jet_pileupId_SF  =1.
            for syst in ["nom", "up", "down"]:
                for j in np.arange(nJet)[maskJets]:
                    if (Jet_pt[j]<50) & (Jet_genJetIdx[j]>-1):
                        wp = "L"
                        current_pileupID_SF = puId_SF_evaluator["PUJetID_eff"].evaluate(float(Jet_eta[j]), float(Jet_pt[j]), syst, wp)
                        jet_pileupId_SF = jet_pileupId_SF*current_pileupID_SF
                    else:
                        jet_pileupId_SF = 1
                features_["jet_pileupId_SF_"+syst]  = jet_pileupId_SF

                    
                    

# PU ID SF
#            if ("GluGluH_M" in processName) | (processName[:4]=="Data"):
#                features_['puId_central'] = 1
#                features_['puId_up'] = 1
#                features_['puId_down'] = 1
#            else:
#                for syst in ["nom", "up", "down"]:
#                    # Load btag Map
#                    puID_SF_event = 1.0
#                    for j in np.arange(nJet)[maskJets]:
#
#
#                        # Extract the tagged WP of the current jet
#                        wp = "T" if Jet_puId[j]>=7 else "notT"
#
#                        # Extract the SF for each tagged selection
#                        map_name = "PUJetID_eff"
#                        currentSF_puID_T = puId_SF_evaluator[map_name].evaluate(Jet_eta[j], Jet_pt[j], syst, wp)
#                        #currentJet_btagSF_T = corrDeepJet_FixedWP_comb.evaluate(syst, "T", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))
#
#                        eff_puId_T = get_puID_efficiency(Jet_pt[j], Jet_eta[j], eff_puId_map["T"])
#
#
#
#
#                        if wp=="T":
#                            # Jet number j is Tight. Apply simply SF
#                            weight_factor = currentSF_puID_T
#                        elif wp=="notT":
#                            weight_factor = (1 - currentSF_puID_T * eff_puId_T ) / (1-eff_puId_T)
#                        jet_puId_SF_event *= weight_factor
#
#                    features_[f'jet_puId_SF_{syst}'] = jet_puId_SF_event
#







# BTag SF and Variations
            if not btagMapsExist:
                features_['btag_central'] = 1
                features_['btag_up'] = 1
                features_['btag_down'] = 1
                features_['sf'] = np.float32(1.)
                features_['genWeight'] = np.float32(1.)
            else:
                for syst in ["central", "up", "down"]:
                    # Load btag Map
                    btagSF_event = 1.0
                    for j in np.arange(nJet)[maskJets]:


                        # Extract the tagged WP of the current jet
                        if wp_converter.evaluate("L") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("M"):
                            wp = "L"
                        elif wp_converter.evaluate("M") <= Jet_btagDeepFlavB[j] < wp_converter.evaluate("T"):
                            wp = "M"
                        elif wp_converter.evaluate("T") <= Jet_btagDeepFlavB[j]:
                            wp = "T"
                        else:
                            wp = None  

                        # Extract the SF for each tagged selection
                        if (abs(Jet_hadronFlavour[j])==4) | abs(Jet_hadronFlavour[j])==5:
                            currentJet_btagSF_T = corrDeepJet_FixedWP_comb.evaluate(syst, "T", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))
                            currentJet_btagSF_M = corrDeepJet_FixedWP_comb.evaluate(syst, "M", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))
                            currentJet_btagSF_L = corrDeepJet_FixedWP_comb.evaluate(syst, "L", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))

                        elif (abs(Jet_hadronFlavour[j])==0) :
                            currentJet_btagSF_T = corrDeepJet_FixedWP_light.evaluate(syst, "T", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))
                            currentJet_btagSF_M = corrDeepJet_FixedWP_light.evaluate(syst, "M", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))
                            currentJet_btagSF_L = corrDeepJet_FixedWP_light.evaluate(syst, "L", abs(Jet_hadronFlavour[j]), float(abs(Jet_eta[j])), float(Jet_pt[j]))

                        eff_T, flav_key = get_btag_map_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["T"])
                        eff_M, _        = get_btag_map_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["M"])
                        eff_L, _        = get_btag_map_efficiency(Jet_pt[j], Jet_eta[j], Jet_hadronFlavour[j], eff_maps_cache["L"])



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


def main(fileName, maxEntries, maxJet, pN, fullProcessName, method, isJEC, verbose, isMC):
    print("FileName", fileName)
    print("Process", fullProcessName, flush=True)
    # If isJEC is True Process contains also the name of the JEC uncertainty e.g. GluGluHToBB_JECAbsoluteMPFBias_Down
    # fullProcessName = complete name (GluGluHToBB_JECAbsoluteMPFBias_Down)
    # processName = only physics name (GluGluHToBB)
    # JECname = (JECAbsoluteMPFBias_Down)
    if isJEC:
        JECname = '_'.join(fullProcessName.split('_')[-2:])
        processName = '_'.join(fullProcessName.split('_')[:-2])
        print("processName : ", processName)
        print("JEC : ", JECname)
    else:
        JECname = ''
        if '_smeared' in fullProcessName:
            # We are in JER smearing case
            processName = '_smeared'.join(fullProcessName.split('_')[:-1])
            JERname = '_smeared'.join(fullProcessName.split('_')[-1:])
            print("JER is : ", JERname)
            print("processName : ", processName)

        else:
            processName = fullProcessName
            print("No JER no JEC")
            print("processName : ", processName)
    
    
    # Event by event operations:
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, pN=pN, processName=processName, method=method, isJEC=isJEC, verbose=verbose, JECname=JECname, isMC=isMC)
    df=pd.DataFrame(fileData)
    


    #df.columns = featureNames
    print("Look for fileNumber in the file")
    try:
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
    except:
        fileNumber = 1999
    print("FileNumber ", fileNumber)
    # \D single non digit character
    # (\d{1,4}) any numbers of 1 to 4 digits
    # . a dot
    # \w+ any word characters(letter digits, underscore)
    # $ End of string
    # .group(1) First capturing group (the 4 digits number). 
    # .group(0) Return the entire matching _129.root
    # .group(1) Returns the group of interest which is the part in the brackets

    # PU_SF. To be applied only on MC
    if fullProcessName[:4]=='Data':
        df['PU_SF']=1
    else:
        PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0

    print('/scratch/' +fullProcessName+"_%s.parquet"%fileNumber)
    df.to_parquet('/scratch/' +fullProcessName+"_%s.parquet"%fileNumber )
    print("Saving in " + '/scratch/' +fullProcessName+"_%s.parquet"%fileNumber )
    print("File exists : ", os.path.exists('/scratch/' +fullProcessName+"_%s.parquet"%fileNumber ))

if __name__ == "__main__":
    fileName    = sys.argv[1]
    maxEntries  = int(sys.argv[2])
    maxJet      = int(sys.argv[3])
    pN        = int(sys.argv[4])
    fullProcessName     = sys.argv[5] 
    method = int(sys.argv[6])
    isJEC = int(sys.argv[7])
    verbose = int(sys.argv[8])
    isMC = int(sys.argv[9])
    print("calling main", flush=True)
    main(fileName, maxEntries, maxJet, pN, fullProcessName, method, isJEC, verbose, isMC)