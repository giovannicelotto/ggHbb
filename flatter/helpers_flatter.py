import numpy as np
import ROOT
from treeFlatter_dict_getSFs import getMuonID_SF, get_muon_recoSF, btag_wp
from getZ_KFactor import getZ_KFactor
def get_event_branches(branches, ev, isMC, run=2):
    if run==2:
        return {
        "event"          : branches["event"][ev],
        "run"          : branches["run"][ev],
        "luminosityBlock"          : branches["luminosityBlock"][ev],

    # Reco Jets 
        "nJet"                        : branches["nJet"][ev],
        "Jet_eta"                     : branches["Jet_eta"][ev],
        "Jet_pt"                      : branches["Jet_pt"][ev],
        "Jet_phi"                     : branches["Jet_phi"][ev],
        "Jet_mass"                    : branches["Jet_mass"][ev],
        "Jet_btagDeepFlavB"           : branches["Jet_btagDeepFlavB"][ev],
        
        #"Jet_pf1_charge"           : branches["Jet_pf1_charge"][ev],
        #"Jet_pf1_pt"           : branches["Jet_pf1_pt"][ev],
        #"Jet_pf1_eta"           : branches["Jet_pf1_eta"][ev],
        #"Jet_pf1_phi"           : branches["Jet_pf1_phi"][ev],
        #"Jet_pf1_mass"           : branches["Jet_pf1_mass"][ev],
#
        #"Jet_pf2_charge"           : branches["Jet_pf2_charge"][ev],
        #"Jet_pf2_pt"           : branches["Jet_pf2_pt"][ev],
        #"Jet_pf2_eta"           : branches["Jet_pf2_eta"][ev],
        #"Jet_pf2_phi"           : branches["Jet_pf2_phi"][ev],
        #"Jet_pf2_mass"           : branches["Jet_pf2_mass"][ev],
#
#
        #"Jet_pf3_charge"           : branches["Jet_pf3_charge"][ev],
        #"Jet_pf3_pt"           : branches["Jet_pf3_pt"][ev],
        #"Jet_pf3_eta"           : branches["Jet_pf3_eta"][ev],
        #"Jet_pf3_phi"           : branches["Jet_pf3_phi"][ev],
        #"Jet_pf3_mass"           : branches["Jet_pf3_mass"][ev],
#
#
        #"Jet_pf4_charge"           : branches["Jet_pf4_charge"][ev],
        #"Jet_pf4_pt"           : branches["Jet_pf4_pt"][ev],
        #"Jet_pf4_eta"           : branches["Jet_pf4_eta"][ev],
        #"Jet_pf4_phi"           : branches["Jet_pf4_phi"][ev],
        #"Jet_pf4_mass"           : branches["Jet_pf4_mass"][ev],
#
#
        #"Jet_pf5_charge"           : branches["Jet_pf5_charge"][ev],
        #"Jet_pf5_pt"           : branches["Jet_pf5_pt"][ev],
        #"Jet_pf5_eta"           : branches["Jet_pf5_eta"][ev],
        #"Jet_pf5_phi"           : branches["Jet_pf5_phi"][ev],
        #"Jet_pf5_mass"           : branches["Jet_pf5_mass"][ev],

        
        

        # ID
        "Jet_jetId"                   : branches["Jet_jetId"][ev],
        "Jet_puId"                    : branches["Jet_puId"][ev],

        # Vtx
        "Jet_vtx3dL"                  : branches["Jet_vtx3dL"][ev],
        "Jet_vtx3deL"                 : branches["Jet_vtx3deL"][ev],
        "Jet_vtxPt"                   : branches["Jet_vtxPt"][ev],
        "Jet_vtxMass"                 : branches["Jet_vtxMass"][ev],
        "Jet_vtxNtrk"                 : branches["Jet_vtxNtrk"][ev],

        # Others
        "Jet_area"                    : branches["Jet_area"][ev],
        "Jet_rawFactor"               : branches["Jet_rawFactor"][ev],
        "Jet_qgl"                     : branches["Jet_qgl"][ev],
        "Jet_nMuons"                  : branches["Jet_nMuons"][ev],
        "Jet_nConstituents"           : branches["Jet_nConstituents"][ev],
        "Jet_leadTrackPt"             : branches["Jet_leadTrackPt"][ev],
        "Jet_nElectrons"              : branches["Jet_nElectrons"][ev],
        "Jet_muonIdx1"                : branches["Jet_muonIdx1"][ev],
        "Jet_muonIdx2"                : branches["Jet_muonIdx2"][ev],
        # Regression
        "Jet_bReg2018"                 : branches["Jet_bReg2018"][ev],


        "dijet_eta"                     : branches["dijet_eta"][ev],
        "dijet_phi"                     : branches["dijet_phi"][ev],
        
        # Taggers
        #"#Jet_btagDeepFlavC"           : branches["Jet_btagDeepFlavC"][ev],
        #"#Jet_btagPNetB"               : branches["Jet_btagPNetB"][ev],
        #"#Jet_PNetRegPtRawCorr"        : branches["Jet_PNetRegPtRawCorr"][ev],
        #"#Jet_PNetRegPtRawCorrNeutrino": branches["Jet_PNetRegPtRawCorrNeutrino"][ev],
        #"#Jet_PNetRegPtRawRes"         : branches["Jet_PNetRegPtRawRes"][ev],
        

    # Muons
        "nMuon"                       : branches["nMuon"][ev],
        "Muon_pt"                     : branches["Muon_pt"][ev],
        "Muon_eta"                    : branches["Muon_eta"][ev],
        "Muon_phi"                    : branches["Muon_phi"][ev],
        "Muon_mass"                   : branches["Muon_mass"][ev],
        "Muon_isTriggering"           : branches["Muon_isTriggering"][ev],
        "Muon_dxy"                    : branches["Muon_dxy"][ev],
        "Muon_dxyErr"                 : branches["Muon_dxyErr"][ev],
        "Muon_dz"                     : branches["Muon_dz"][ev],
        "Muon_dzErr"                  : branches["Muon_dzErr"][ev],
        "Muon_pfIsoId"                : branches["Muon_pfIsoId"][ev],  # 1:PFIsoVeryLoose, 2:PFIsoLoose, 3:PFIsoMedium, 4:PFIsoTight, 5:PFIsoVeryTight, 6:PFIsoVeryVeryTight),
        "Muon_pfRelIso03_all"         : branches["Muon_pfRelIso03_all"][ev],
        "Muon_ip3d"                   : branches["Muon_ip3d"][ev],
        "Muon_sip3d"                  : branches["Muon_sip3d"][ev],
        "Muon_charge"                 : branches["Muon_charge"][ev],
        "Muon_tightId"                : branches["Muon_tightId"][ev],
        "Muon_mediumId"               : branches["Muon_mediumId"][ev],
        "Muon_tkIsoId"                : branches["Muon_tkIsoId"][ev],
        "Muon_pfRelIso04_all"         : branches["Muon_pfRelIso04_all"][ev],

    # Electrons
        "Jet_electronIdx1"            : branches["Jet_electronIdx1"][ev],
        "nElectron"                   : branches["nElectron"][ev],
        "Electron_pt"                 : branches["Electron_pt"][ev],
        "Electron_eta"                : branches["Electron_eta"][ev],
        "Electron_phi"                : branches["Electron_phi"][ev],
        "Electron_dz"                 : branches["Electron_dz"][ev],
        "Electron_dzErr"              : branches["Electron_dzErr"][ev],
        "Electron_ip3d"               : branches["Electron_ip3d"][ev],
        "Electron_sip3d"              : branches["Electron_sip3d"][ev],
        
        "Electron_isPF"               : branches["Electron_isPFcand"][ev],
        "Electron_charge"             : branches["Electron_charge"][ev],
        "Electron_pfRelIso03_all"     : branches["Electron_pfRelIso03_all"][ev],
        "Electron_r9"                 : branches["Electron_r9"][ev],
        "Electron_cutBased"        : branches["Electron_cutBased"][ev], # (0:fail, 1:veto, 2:loose, 3:medium, 4:tight) from CMSSW
        "Electron_dxy"                    : branches["Electron_dxy"][ev],
        "Electron_dxyErr"                 : branches["Electron_dxyErr"][ev],
        "Electron_mvaIso"                 : branches["Electron_mvaIso"][ev],
        "Electron_mvaIso_WP80"            : branches["Electron_mvaIso_WP80"][ev],
        "Electron_mvaIso_WP90"            : branches["Electron_mvaIso_WP90"][ev],
        "Electron_mvaIso_WPL"             : branches["Electron_mvaIso_WPL"][ev],
        "Electron_mvaNoIso_WP80"          : branches["Electron_mvaNoIso_WP80"][ev],
        "Electron_mvaNoIso_WP90"          : branches["Electron_mvaNoIso_WP90"][ev],
        "Electron_mvaNoIso_WPL"           : branches["Electron_mvaNoIso_WPL"][ev],
        "Electron_convVeto"               : branches["Electron_convVeto"][ev],



    # Triggers
        "Muon_fired_HLT_Mu12_IP6" :       branches["Muon_fired_HLT_Mu12_IP6"][ev],
        "Muon_fired_HLT_Mu7_IP4" :        branches["Muon_fired_HLT_Mu7_IP4"][ev],
        "Muon_fired_HLT_Mu8_IP3" :        branches["Muon_fired_HLT_Mu8_IP3"][ev],
        "Muon_fired_HLT_Mu8_IP5" :        branches["Muon_fired_HLT_Mu8_IP5"][ev],
        "Muon_fired_HLT_Mu8_IP6" :        branches["Muon_fired_HLT_Mu8_IP6"][ev],
        "Muon_fired_HLT_Mu10p5_IP3p5" :        branches["Muon_fired_HLT_Mu10p5_IP3p5"][ev],
        "Muon_fired_HLT_Mu8p5_IP3p5" :        branches["Muon_fired_HLT_Mu8p5_IP3p5"][ev],
        "Muon_fired_HLT_Mu9_IP4" :        branches["Muon_fired_HLT_Mu9_IP4"][ev],
        "Muon_fired_HLT_Mu9_IP5" :        branches["Muon_fired_HLT_Mu9_IP5"][ev],
        "Muon_fired_HLT_Mu9_IP6" :        branches["Muon_fired_HLT_Mu9_IP6"][ev],
        "nSV"                    :        branches["nSV"][ev],
        "PV_npvs"                :        branches["PV_npvs"][ev],
        
    # Data MC dependent
    #Gen Information
        "Muon_genPartIdx"          : branches["Muon_genPartIdx"][ev] if isMC==1 else None,
        "Jet_hadronFlavour"        : branches["Jet_hadronFlavour"][ev] if isMC==1 else None,
        "Pileup_nTrueInt"          : branches["Pileup_nTrueInt"][ev] if isMC==1 else None,
        "Jet_genJetIdx"            : branches["Jet_genJetIdx"][ev] if isMC==1 else None,
        "GenJetNu_pt"              : branches["GenJetNu_pt"][ev] if isMC==1 else None,
        "GenJetNu_eta"             : branches["GenJetNu_eta"][ev] if isMC==1 else None,
        "GenJetNu_phi"             : branches["GenJetNu_phi"][ev] if isMC==1 else None,
        "GenJetNu_mass"            : branches["GenJetNu_mass"][ev] if isMC==1 else None,
        "GenJet_pt"                : branches["GenJet_pt"][ev] if isMC==1 else None,
        "GenJet_eta"               : branches["GenJet_eta"][ev] if isMC==1 else None,
        "GenJet_phi"               : branches["GenJet_phi"][ev] if isMC==1 else None,
        "GenJet_mass"              : branches["GenJet_mass"][ev] if isMC==1 else None,
        "GenJet_hadronFlavour"     : branches["GenJet_hadronFlavour"][ev] if isMC==1 else None,
        "GenJet_partonMotherPdgId" : branches["GenJet_partonMotherPdgId"][ev] if isMC==1 else None,
        "GenJet_partonMotherIdx"   : branches["GenJet_partonMotherIdx"][ev] if isMC==1 else None,
        "genWeight"    : branches["genWeight"][ev] if isMC==1 else 1,




    }
    else:
        return {
        "event"          : branches["event"][ev],
        "run"          : branches["run"][ev],
        "luminosityBlock"          : branches["luminosityBlock"][ev],

    # Reco Jets 
        "nJet"                        : branches["nJet"][ev],
        "Jet_eta"                     : branches["Jet_eta"][ev],
        "Jet_pt"                      : branches["Jet_pt"][ev],
        "Jet_phi"                     : branches["Jet_phi"][ev],
        "Jet_mass"                    : branches["Jet_mass"][ev],
        "Jet_btagDeepFlavB"           : branches["Jet_btagDeepFlavB"][ev],
        "HLT_Mu10_Barrel_L1HP11_IP6"  : branches["HLT_Mu10_Barrel_L1HP11_IP6"][ev],
        # ID
        #"Jet_jetId"                   : branches["Jet_jetId"][ev],
        #"Jet_puId"                    : branches["Jet_puId"][ev],

        # Vtx
        #"Jet_vtx3dL"                  : branches["Jet_vtx3dL"][ev],
        #"Jet_vtx3deL"                 : branches["Jet_vtx3deL"][ev],
        #"Jet_vtxPt"                   : branches["Jet_vtxPt"][ev],
        #"Jet_vtxMass"                 : branches["Jet_vtxMass"][ev],
        #"Jet_vtxNtrk"                 : branches["Jet_vtxNtrk"][ev],

        # Others
        "Jet_area"                    : branches["Jet_area"][ev],
        "Jet_rawFactor"               : branches["Jet_rawFactor"][ev],
        #"Jet_qgl"                     : branches["Jet_qgl"][ev],
        "Jet_nMuons"                  : branches["Jet_nMuons"][ev],
        "Jet_nConstituents"           : branches["Jet_nConstituents"][ev],
        #"Jet_leadTrackPt"             : branches["Jet_leadTrackPt"][ev],
        "Jet_nElectrons"              : branches["Jet_nElectrons"][ev],
        "Jet_muonIdx1"                : branches["Jet_muonIdx1"][ev],
        "Jet_muonIdx2"                : branches["Jet_muonIdx2"][ev],
        # Regression
        #"Jet_bReg2018"                 : branches["Jet_bReg2018"][ev],


        #"dijet_eta"                     : branches["dijet_eta"][ev],
        #"dijet_phi"                     : branches["dijet_phi"][ev],
        
        # Taggers
        #"#Jet_btagDeepFlavC"           : branches["Jet_btagDeepFlavC"][ev],
        #"#Jet_btagPNetB"               : branches["Jet_btagPNetB"][ev],
        #"#Jet_PNetRegPtRawCorr"        : branches["Jet_PNetRegPtRawCorr"][ev],
        #"#Jet_PNetRegPtRawCorrNeutrino": branches["Jet_PNetRegPtRawCorrNeutrino"][ev],
        #"#Jet_PNetRegPtRawRes"         : branches["Jet_PNetRegPtRawRes"][ev],
        

    # Muons
        "nMuon"                       : branches["nMuon"][ev],
        "Muon_pt"                     : branches["Muon_pt"][ev],
        "Muon_eta"                    : branches["Muon_eta"][ev],
        "Muon_phi"                    : branches["Muon_phi"][ev],
        "Muon_mass"                   : branches["Muon_mass"][ev],
        #"Muon_isTriggering"           : branches["Muon_isTriggering"][ev],
        "Muon_dxy"                    : branches["Muon_dxy"][ev],
        "Muon_dxyErr"                 : branches["Muon_dxyErr"][ev],
        "Muon_dz"                     : branches["Muon_dz"][ev],
        "Muon_dzErr"                  : branches["Muon_dzErr"][ev],
        "Muon_pfIsoId"                : branches["Muon_pfIsoId"][ev],  # 1:PFIsoVeryLoose, 2:PFIsoLoose, 3:PFIsoMedium, 4:PFIsoTight, 5:PFIsoVeryTight, 6:PFIsoVeryVeryTight),
        "Muon_pfRelIso03_all"         : branches["Muon_pfRelIso03_all"][ev],
        "Muon_ip3d"                   : branches["Muon_ip3d"][ev],
        "Muon_sip3d"                  : branches["Muon_sip3d"][ev],
        "Muon_charge"                 : branches["Muon_charge"][ev],
        "Muon_tightId"                : branches["Muon_tightId"][ev],
        "Muon_mediumId"               : branches["Muon_mediumId"][ev],
        #"Muon_tkIsoId"                : branches["Muon_tkIsoId"][ev],
        "Muon_pfRelIso04_all"         : branches["Muon_pfRelIso04_all"][ev],

    # Electrons
        "Jet_electronIdx1"            : branches["Jet_electronIdx1"][ev],
        "nElectron"                   : branches["nElectron"][ev],



    # Triggers
        "nSV"                    :        branches["nSV"][ev],
        "PV_npvs"                :        branches["PV_npvs"][ev],
        
    # Data MC dependent
    #Gen Information
        "Muon_genPartIdx"          : branches["Muon_genPartIdx"][ev] if isMC==1 else None,
        "Jet_hadronFlavour"        : branches["Jet_hadronFlavour"][ev] if isMC==1 else None,
        "Pileup_nTrueInt"          : branches["Pileup_nTrueInt"][ev] if isMC==1 else None,
        "genWeight"    : branches["genWeight"][ev] if isMC==1 else 1,




    }


def get_event_genBranches(branches, ev, processName):
    dict = {
    "GenPart_pt" : branches["GenPart_pt"][ev],
    "GenPart_pdgId" : branches["GenPart_pdgId"][ev],
    "GenPart_genPartIdxMother" : branches["GenPart_genPartIdxMother"][ev],
    "GenPart_eta" : branches["GenPart_eta"][ev],
    "GenPart_phi" : branches["GenPart_phi"][ev],
    "GenPart_mass" : branches["GenPart_mass"][ev],
    "GenPart_statusFlags" : branches["GenPart_statusFlags"][ev],


    }
    dict["PSWeight"] = branches["PSWeight"][ev]
    if "LHEScaleWeight" in branches.fields:
        dict["LHEScaleWeight"] = branches["LHEScaleWeight"][ev]
    else:
        dict["LHEScaleWeight"] = np.ones(9)
    if "LHEPdfWeight" in branches.fields:
        dict["LHEPdfWeight"] = branches["LHEPdfWeight"][ev]
    else:
        dict["LHEPdfWeight"] = np.ones(103)
    if (('ZJetsToQQ' in processName) | ('EWKZJets' in processName)):
        dict["LHEPart_pt"] = branches["LHEPart_pt"][ev],
        dict["LHEPart_pdgId"] = branches["LHEPart_pdgId"][ev],
    return dict
def fill_jet_features(prefix, idx, evt, jet_vec, dijet=None, jetIsPresent=None):
    features = {}

    features[f"{prefix}_pt"]   = jet_vec.Pt() if jet_vec is not None else 0.
    features[f"{prefix}_eta"]  = evt["Jet_eta"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_phi"]  = evt["Jet_phi"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_mass"] = jet_vec.M() if jet_vec is not None else 0.

    features[f"{prefix}_btagDeepFlavB"] = evt["Jet_btagDeepFlavB"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_rawFactor"]     = evt["Jet_rawFactor"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_nMuons"]        = evt["Jet_nMuons"][idx] if jet_vec is not None else 0
    features[f"{prefix}_nConstituents"] = evt["Jet_nConstituents"][idx] if jet_vec is not None else 0
    features[f"{prefix}_leadTrackPt"]   = evt["Jet_leadTrackPt"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_puId"]          = evt["Jet_puId"][idx] if jet_vec is not None else 0
    features[f"{prefix}_jetId"]         = evt["Jet_jetId"][idx] if jet_vec is not None else 0
    features[f"{prefix}_nElectrons"]   = evt["Jet_nElectrons"][idx]      if jet_vec is not None else 0
    features[f"{prefix}_btagTight"]    = int(evt["Jet_btagDeepFlavB"][idx]>=0.71) if jet_vec is not None else 0
    
    #features[f"{prefix}_pf1_pt"]    = float(evt["Jet_pf1_pt"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf1_eta"]    = float(evt["Jet_pf1_eta"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf1_phi"]    = float(evt["Jet_pf1_phi"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf1_mass"]    = float(evt["Jet_pf1_mass"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf1_charge"]    = int(evt["Jet_pf1_charge"][idx]) if jet_vec is not None else 0
#
    #features[f"{prefix}_pf2_pt"]    = float(evt["Jet_pf2_pt"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf2_eta"]    = float(evt["Jet_pf2_eta"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf2_phi"]    = float(evt["Jet_pf2_phi"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf2_mass"]    = float(evt["Jet_pf2_mass"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf2_charge"]    = int(evt["Jet_pf2_charge"][idx]) if jet_vec is not None else 0
#
#
#
    #features[f"{prefix}_pf3_pt"]    = float(evt["Jet_pf3_pt"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf3_eta"]    = float(evt["Jet_pf3_eta"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf3_phi"]    = float(evt["Jet_pf3_phi"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf3_mass"]    = float(evt["Jet_pf3_mass"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf3_charge"]    = int(evt["Jet_pf3_charge"][idx]) if jet_vec is not None else 0
#
#
#
    #features[f"{prefix}_pf4_pt"]    = float(evt["Jet_pf4_pt"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf4_eta"]    = float(evt["Jet_pf4_eta"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf4_phi"]    = float(evt["Jet_pf4_phi"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf4_mass"]    = float(evt["Jet_pf4_mass"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf4_charge"]    = int(evt["Jet_pf4_charge"][idx]) if jet_vec is not None else 0
#
#
#
    #features[f"{prefix}_pf5_pt"]    = float(evt["Jet_pf5_pt"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf5_eta"]    = float(evt["Jet_pf5_eta"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf5_phi"]    = float(evt["Jet_pf5_phi"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf5_mass"]    = float(evt["Jet_pf5_mass"][idx]) if jet_vec is not None else 0
    #features[f"{prefix}_pf5_charge"]    = int(evt["Jet_pf5_charge"][idx]) if jet_vec is not None else 0


    counterMuTight=0
    if jet_vec is not None:
        for muIdx in range(len(evt["Muon_pt"])):
            if (np.sqrt((evt["Muon_eta"][muIdx]-evt["Jet_eta"][idx])**2 + (evt["Muon_phi"][muIdx]-evt["Jet_phi"][idx])**2)<0.4) & (evt["Muon_tightId"][muIdx]):
                counterMuTight=counterMuTight+1
    features[f"{prefix}_nTightMuons"] = counterMuTight        if jet_vec is not None else 0

    jet_btagWP = btag_wp(evt["Jet_btagDeepFlavB"][idx], evt["Jet_pt"][idx]) if jet_vec is not None else -1
    features[f"{prefix}_btagWP"] = int(jet_btagWP) if jet_vec is not None else 0
    features[f"{prefix}_idx"] = idx if jet_vec is not None else -1
    features[f"{prefix}_rawFactor"] = evt["Jet_rawFactor"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_bReg2018"] = evt["Jet_bReg2018"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_sv_pt"] = evt["Jet_vtxPt"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_sv_mass"] = evt["Jet_vtxMass"][idx] if jet_vec is not None else 0.
    features[f"{prefix}_sv_Ntrk"] = evt["Jet_vtxNtrk"][idx] if jet_vec is not None else 0
    if jet_vec is not None:
        jet_sv_3dSig = (evt["Jet_vtx3dL"][idx])/(evt["Jet_vtx3deL"][idx]) if (evt["Jet_vtx3dL"][idx]!=0) else 0
    else:
        jet_sv_3dSig = 0.
    features[f"{prefix}_sv_3dSig"] = jet_sv_3dSig if jet_vec is not None else 0.

    if dijet is not None:
        features[f"dR_{prefix}_dijet"]   = jet_vec.DeltaR(dijet) if jet_vec is not None else 0.
        features[f"dPhi_{prefix}_dijet"] = jet_vec.DeltaPhi(dijet) if jet_vec is not None else 0.
        features[f"dEta_{prefix}_dijet"] = jet_vec.DeltaPhi(dijet) if jet_vec is not None else 0.
    else:
        features[f"dR_{prefix}_dijet"]   = 0.
        features[f"dPhi_{prefix}_dijet"] = 0.
        features[f"dEta_{prefix}_dijet"] = 0.
        
    if jetIsPresent:
        features[f"has_{prefix}"] = 1
    elif jetIsPresent==False:
        features[f"has_{prefix}"] = 0
    else:
        # if jet is present is None don't do anything
        pass


    return features


def fill_ttbar_CR_features(evt, isMC, muonIdx1, muonIdx2, processName, muon_RECO_map, muon_ID_map, muon_ISO_map, electrons_SF_map):
    features = {}
    weight_factor = 1.0

        # Select muons
    muon_mask = (evt["Muon_pt"] > 25) & (np.abs(evt["Muon_eta"]) < 2.4) & (evt["Muon_pfIsoId"]>=4) & (evt["Muon_tightId"]==1)
    electron_mask = (evt["Electron_pt"] > 25) & (np.abs(evt["Electron_eta"]) < 2.5)  & (evt["Electron_cutBased"]>=3) #& (Electron_pfRelIso03_all<0.1)

    # At least one muon & one electron passing the cuts
    selected_muon_idx = None
    selected_ele_idx = None
    if np.any(muon_mask) and np.any(electron_mask):
        # Loop over muons & electrons to find the first opposite-charge pair
        for mu_idx in np.where(muon_mask)[0]:
            if (mu_idx==muonIdx1) | (mu_idx==muonIdx2):

                continue 
            for el_idx in np.where(electron_mask)[0]:
                if evt["Muon_charge"][mu_idx] * evt["Electron_charge"][el_idx] < 0:
                    selected_muon_idx = mu_idx
                    selected_ele_idx = el_idx
                    break  # stop at the first matching muon
            if selected_muon_idx is not None:
                break

        if selected_muon_idx is not None:
            features["is_ttbar_CR"] = 1
        else:
            features["is_ttbar_CR"] = 0
    else:
        features["is_ttbar_CR"] = 0
    

    features["Muon_tt_pt"]         = np.float32(evt["Muon_pt"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Muon_tt_eta"]        = np.float32(evt["Muon_eta"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Muon_tt_phi"]        = np.float32(evt["Muon_phi"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    
    features["Muon_tt_dxy"]            = np.float32(evt["Muon_dxy"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Muon_tt_dz"]             = np.float32(evt["Muon_dz"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    if isMC==1:
        features["Muon_tt_genMatched"]     = int(evt["Muon_genPartIdx"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.

    features["Muon_tt_charge"]     = int(evt["Muon_charge"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Muon_tt_mediumId"]   = int(evt["Muon_mediumId"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Muon_tt_pfIsoId"]    = int(evt["Muon_pfIsoId"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else -999.

    muon_tt_RECO_SF = get_muon_recoSF(muon_RECO_map, eta=evt["Muon_eta"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_RECO_SF"] = muon_tt_RECO_SF["value"] if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_RECO_stat"] = muon_tt_RECO_SF["stat"] if features["is_ttbar_CR"]==1 else 0.
    features["Muon_tt_RECO_syst"] = muon_tt_RECO_SF["syst"] if features["is_ttbar_CR"]==1 else 0.
    muon_tt_ID_SF = getMuonID_SF(muon_ID_map, "NUM_TightID_DEN_TrackerMuons", evt["Muon_eta"][selected_muon_idx], evt["Muon_pt"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_ID_SF"] = muon_tt_ID_SF["value"] if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_ID_stat"] = muon_tt_ID_SF["stat"]  if features["is_ttbar_CR"]==1 else 0.
    features["Muon_tt_ID_syst"] = muon_tt_ID_SF["syst"]  if features["is_ttbar_CR"]==1 else 0.

    muon_ISO_SF = getMuonID_SF(muon_ISO_map, "NUM_TightRelIso_DEN_TightIDandIPCut", evt["Muon_eta"][selected_muon_idx],  evt["Muon_pt"][selected_muon_idx]) if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_ISO_SF"] = muon_ISO_SF["value"] if features["is_ttbar_CR"]==1 else 1.
    features["Muon_tt_ISO_stat"] = muon_ISO_SF["stat"] if features["is_ttbar_CR"]==1 else 0.
    features["Muon_tt_ISO_syst"] = muon_ISO_SF["syst"] if features["is_ttbar_CR"]==1 else 0.



    features["Electron_tt_pt"]         = np.float32(evt["Electron_pt"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_eta"]        = np.float32(evt["Electron_eta"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_phi"]        = np.float32(evt["Electron_phi"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_charge"]     = int(evt["Electron_charge"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_isPF"]       = int(evt["Electron_isPF"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_pfRelIso03_all"]   = np.float32(evt["Electron_pfRelIso03_all"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.

    features["Electron_tt_r9"]            = np.float32(evt["Electron_r9"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_cutBased"]       = np.float32(evt["Electron_cutBased"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_dxy"]            =np.float32(evt["Electron_dxy"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_dxyErr"]            =np.float32(evt["Electron_dxyErr"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.


    features["Electron_tt_mvaIso"] = np.float32(evt["Electron_mvaIso"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaIso_WP80"] = np.float32(evt["Electron_mvaIso_WP80"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaIso_WP90"] = np.float32(evt["Electron_mvaIso_WP90"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaIso_WPL"] = np.float32(evt["Electron_mvaIso_WPL"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaNoIso_WP80"] = np.float32(evt["Electron_mvaNoIso_WP80"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaNoIso_WP90"] = np.float32(evt["Electron_mvaNoIso_WP90"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_mvaNoIso_WPL"] = np.float32(evt["Electron_mvaNoIso_WPL"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.
    features["Electron_tt_convVeto"] = np.float32(evt["Electron_convVeto"][selected_ele_idx]) if features["is_ttbar_CR"]==1 else -999.

    #if Electron_pt[selected_ele_idx] >= 20 :
    features["Electron_tt_RECO_SF"] = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018","sf","RecoAbove20",float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.
    features["Electron_tt_RECO_SF_up"]  = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfup",  "RecoAbove20", float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.
    features["Electron_tt_RECO_SF_down"]= electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfdown","RecoAbove20", float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.

    features["Electron_tt_ID_SF"] = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018","sf","Medium",float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.
    features["Electron_tt_ID_SF_up"]  = electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfup",  "Medium", float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.
    features["Electron_tt_ID_SF_down"]= electrons_SF_map["UL-Electron-ID-SF"].evaluate("2018", "sfdown","Medium", float(evt["Electron_eta"][selected_ele_idx]), float(evt["Electron_pt"][selected_ele_idx])) if features["is_ttbar_CR"]==1 else 1.

    mu = ROOT.TLorentzVector(0.,0.,0.,0.)
    el = ROOT.TLorentzVector(0.,0.,0.,0.)
    mu.SetPtEtaPhiM(evt["Muon_pt"][selected_muon_idx], evt["Muon_eta"][selected_muon_idx], evt["Muon_phi"][selected_muon_idx], 0.106) if features["is_ttbar_CR"]==1 else None
    el.SetPtEtaPhiM(evt["Electron_pt"][selected_ele_idx], evt["Electron_eta"][selected_ele_idx], evt["Electron_phi"][selected_ele_idx], 0.000511) if features["is_ttbar_CR"]==1 else None
    features["dilepton_tt_mass"]   = np.float32((mu+el).M()) if features["is_ttbar_CR"]==1 else 0.


    weight_factor *=  features["Muon_tt_RECO_SF"] * features["Muon_tt_ID_SF"] * features["Muon_tt_ISO_SF"] * features["Electron_tt_ID_SF"] * features["Electron_tt_RECO_SF"]
    

    return features, weight_factor

def fill_dijet_features(dijet_vec, jet1_vec, jet2_vec, evt, run=2):
    features = {}
    if dijet_vec.Pt()<1e-5:
        assert False
    features['dijet_pt'] = np.float32(dijet_vec.Pt())
    #features['dijet_eta_nano'] = evt["dijet_eta"]
    #features['dijet_pt_nano'] = evt["dijet_pt"]
    #if np.abs(evt["dijet_pt"]-dijet_vec.Pt())>1:
    #    print("%.3f vs %.3f"%(evt["dijet_pt"], dijet_vec.Pt()))
    #    print("%.3f vs %.3f"%(evt["dijet_jet1_pt"], jet1_vec.Pt()))
    #    print("%.3f vs %.3f"%(evt["dijet_jet2_pt"], jet2_vec.Pt()))
    if run==2:
        features['dijet_eta_nano'] = evt["dijet_eta"]
        features['dijet_phi_nano'] = evt["dijet_phi"]
    features['dijet_eta'] = np.float32(dijet_vec.Eta())
    features['dijet_phi'] = np.float32(dijet_vec.Phi())
    features['dijet_mass'] = np.float32(dijet_vec.M())
    features['dijet_dR'] = np.float32(jet1_vec.DeltaR(jet2_vec))
    features['dijet_dEta'] = np.float32(abs(jet1_vec.Eta() - jet2_vec.Eta()))
    # Here replace
    deltaPhi = jet1_vec.Phi()-jet2_vec.Phi()
    deltaPhi = deltaPhi - 2*np.pi*(deltaPhi >= np.pi) + 2*np.pi*(deltaPhi< -np.pi)
    features['dijet_dPhi'] = np.float32(abs(deltaPhi))
    
    tau = np.arctan(abs(deltaPhi)/abs(jet1_vec.Eta() - jet2_vec.Eta() + 0.0000001))
    features['dijet_twist'] = np.float32(tau)

    cs_angle = 2*((jet1_vec.Pz()*jet2_vec.E() - jet2_vec.Pz()*jet1_vec.E())/(dijet_vec.M()*np.sqrt(dijet_vec.M()**2+dijet_vec.Pt()**2)))
    features['dijet_cs'] = np.float32(cs_angle)
    features['normalized_dijet_pt'] = dijet_vec.Pt()/(jet1_vec.Pt()+jet2_vec.Pt())
    dijet_pTAsymmetry = (jet1_vec.Pt() - jet2_vec.Pt())/(jet1_vec.Pt() + jet2_vec.Pt())
    features['dijet_pT_asymmetry'] = dijet_pTAsymmetry

    return features

def fill_trig_muon_features(muon,muonIdx, jet_vec, jetIdx, muon_RECO_map, evt, has_muon=None):
    features={}
    muonTrig_RECO_ID = get_muon_recoSF(muon_RECO_map, eta=muon.Eta()) if muon is not None else 1.
    features[f"jet{jetIdx}_muon_RECO_SF"] = np.float32(muonTrig_RECO_ID["value"]) if muon is not None else 1.
    features[f"jet{jetIdx}_muon_RECO_Stat"] = np.float32(muonTrig_RECO_ID["stat"]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_RECO_Syst"] = np.float32(muonTrig_RECO_ID["syst"]) if muon is not None else 0.
    muon_weight = np.float32(muonTrig_RECO_ID["value"]) if muon is not None else 1.

    features[f"jet{jetIdx}_muon_pt"] = np.float32(muon.Pt()) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_eta"] = np.float32(muon.Eta()) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_phi"] = np.float32(muon.Phi()) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_ptRel"] = np.float32(muon.Perp(jet_vec.Vect())) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_dxySig"] = np.float32(evt["Muon_dxy"][muonIdx]/evt["Muon_dxyErr"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_dxy"] = np.float32(evt["Muon_dxy"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_dzSig"] = np.float32(evt["Muon_dz"][muonIdx]/evt["Muon_dzErr"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_IP3d"] = np.float32(evt["Muon_ip3d"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_sIP3d"] = np.float32(evt["Muon_sip3d"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_tightId"] = bool(evt["Muon_tightId"][muonIdx]) if muon is not None else False
    features[f"jet{jetIdx}_muon_pfRelIso03_all"] = np.float32(evt["Muon_pfRelIso03_all"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_pfRelIso04_all"] = np.float32(evt["Muon_pfRelIso04_all"][muonIdx]) if muon is not None else 0.
    features[f"jet{jetIdx}_muon_charge"] = int(evt["Muon_charge"][muonIdx]) if muon is not None else 0

    if has_muon:
        features[f"jet{jetIdx}_has_muon"]=1
    elif has_muon==False:
        features[f"jet{jetIdx}_has_muon"]=0
    else:
        pass
    return features, muon_weight


def fill_gen_info(evt, evt_gen, jet1_vec, jet2_vec, selected1, selected2, selected3, isMC, processName):
    """
    Fill generator-level information for MC events.
    Returns a dict of features.
    """
    features, weight_genOnly = {}, 1
    features['PV_npvs'] = int(evt["PV_npvs"]) if isMC==1 else -1
    features['genWeight'] = evt["genWeight"]
    weight_genOnly *= evt["genWeight"]
    features['Pileup_nTrueInt'] = np.float32(evt["Pileup_nTrueInt"]) if isMC==1 else -1
    features['jet1_genHadronFlavour'] = evt["Jet_hadronFlavour"][selected1] if isMC==1 else -1
    features['jet2_genHadronFlavour'] = evt["Jet_hadronFlavour"][selected2] if isMC==1 else -1
    hadronFlavour3 = evt["Jet_hadronFlavour"][selected3] if (selected3 is not None) & (isMC==1) else -1
    features['jet3_genHadronFlavour'] = hadronFlavour3 
    if (('GluGluH' in processName) | ('VBF' in processName)):
        h = ROOT.TLorentzVector(0.,0.,0.,0.)
        b_gen = ROOT.TLorentzVector(0.,0.,0.,0.)
        antib_gen = ROOT.TLorentzVector(0.,0.,0.,0.)
        
        
        
        mH = (evt_gen["GenPart_pdgId"]==25) & (evt_gen["GenPart_statusFlags"]>8192)
        mb_gen = (evt_gen["GenPart_pdgId"]==5) & ((evt_gen["GenPart_genPartIdxMother"] >= 0)) & (evt_gen["GenPart_pdgId"][evt_gen["GenPart_genPartIdxMother"]]==25)
        mantib_gen = (evt_gen["GenPart_pdgId"]==-5) & ((evt_gen["GenPart_genPartIdxMother"] >= 0)) & (evt_gen["GenPart_pdgId"][evt_gen["GenPart_genPartIdxMother"]]==25)
        b_gen.SetPtEtaPhiM(evt_gen["GenPart_pt"][mb_gen][0], evt_gen["GenPart_eta"][mb_gen][0], evt_gen["GenPart_phi"][mb_gen][0], evt_gen["GenPart_mass"][mb_gen][0])
        antib_gen.SetPtEtaPhiM(evt_gen["GenPart_pt"][mantib_gen][0], evt_gen["GenPart_eta"][mantib_gen][0], evt_gen["GenPart_phi"][mantib_gen][0], evt_gen["GenPart_mass"][mantib_gen][0])
        h.SetPtEtaPhiM(evt_gen["GenPart_pt"][mH][0], evt_gen["GenPart_eta"][mH][0], evt_gen["GenPart_phi"][mH][0], evt_gen["GenPart_mass"][mH][0])
        features['b_gen_pt'] = b_gen.Pt()
        features['b_gen_eta'] = b_gen.Eta()
        features['b_gen_phi'] = b_gen.Phi()
        features['b_gen_mass'] = b_gen.M()
        features['antib_gen_pt'] = antib_gen.Pt()
        features['antib_gen_eta'] = antib_gen.Eta()
        features['antib_gen_phi'] = antib_gen.Phi()
        features['antib_gen_mass'] = antib_gen.M()
        features['higgs_gen_pt'] = h.Pt()
        features['higgs_gen_eta'] = h.Eta()
        features['higgs_gen_phi'] = h.Phi()
        features['higgs_gen_mass'] = h.M()

        #dR and deltaPt/pt
        # Compute dR for all pairings
        dR_b_jet1 = b_gen.DeltaR(jet1_vec)
        dR_b_jet2 = b_gen.DeltaR(jet2_vec)
        dR_antib_jet1 = antib_gen.DeltaR(jet1_vec)
        dR_antib_jet2 = antib_gen.DeltaR(jet2_vec)
        
        score_1 = dR_b_jet1 + dR_antib_jet2
        score_2 = dR_b_jet2 + dR_antib_jet1

        if score_1 < score_2:
            # b-jet1 antib-jet2

            features['dR_jet1_genQuark'] = b_gen.DeltaR(jet1_vec)
            features['dR_jet2_genQuark'] = antib_gen.DeltaR(jet2_vec)
            features['dpT_jet1_genQuark'] = (b_gen.Pt() - jet1_vec.Pt())/b_gen.Pt() 
            features['dpT_jet2_genQuark'] =  (antib_gen.Pt() - jet2_vec.Pt())/antib_gen.Pt() 
            features['jet1_genQuark_pt'] = b_gen.Pt()
            features['jet2_genQuark_pt'] = antib_gen.Pt()
            features['jet1_genQuarkFlavour'] =  5
            features['jet2_genQuarkFlavour'] =  -5

        else:
            # b-jet2 antib-jet1
            features['dR_jet1_genQuark'] = antib_gen.DeltaR(jet1_vec)
            features['dR_jet2_genQuark'] = b_gen.DeltaR(jet2_vec)
            features['dpT_jet1_genQuark'] = (antib_gen.Pt() - jet1_vec.Pt())/antib_gen.Pt() 
            features['dpT_jet2_genQuark'] = (b_gen.Pt() - jet2_vec.Pt())/b_gen.Pt()  
            features['jet1_genQuark_pt'] = antib_gen.Pt()
            features['jet2_genQuark_pt'] = b_gen.Pt()
            features['jet1_genQuarkFlavour'] =  -5
            features['jet2_genQuarkFlavour'] =  5

        # Match with genJet
        if evt["Jet_genJetIdx"][selected1]!=-1:
            pass
            #Matched with genJet
            #features["GenJet1_pt"] =evt["GenJet_pt"][evt["Jet_genJetIdx"][selected1]]
            #features["GenJet1_eta"] =evt["GenJet_eta"][evt["Jet_genJetIdx"][selected1]]
            #features["GenJet1_phi"] =evt["GenJet_phi"][evt["Jet_genJetIdx"][selected1]]
            #features["GenJet1_mass"] =evt["GenJet_mass"][evt["Jet_genJetIdx"][selected1]]                
            #features["GenJet1_partonMotherPdgId"] =evt["GenJet_partonMotherPdgId"][evt["Jet_genJetIdx"][selected1]]                

            
            #genJet = ROOT.TLorentzVector(0.,0.,0.,0.)
            #genJetNu = ROOT.TLorentzVector(0.,0.,0.,0.)
            #genJet.SetPtEtaPhiM(evt["GenJet_pt"][evt["Jet_genJetIdx"][selected1]], evt["GenJet_eta"][evt["Jet_genJetIdx"][selected1]], evt["GenJet_phi"][evt["Jet_genJetIdx"][selected1]], evt["GenJet_mass"][evt["Jet_genJetIdx"][selected1]])

            #min_dR, min_idx = 999, 999
            #for j_nu in range(len(evt["GenJetNu_pt"])):
            #    genJetNu.SetPtEtaPhiM(evt["GenJetNu_pt"][j_nu], evt["GenJetNu_eta"][j_nu], evt["GenJetNu_phi"][j_nu], evt["GenJetNu_mass"][j_nu])
            #    if genJetNu.DeltaR(genJet) < min_dR:
            #        min_dR = genJetNu.DeltaR(genJet)
            #        min_idx = j_nu
            #genJetNu.SetPtEtaPhiM(evt["GenJetNu_pt"][min_idx], evt["GenJetNu_eta"][min_idx], evt["GenJetNu_phi"][min_idx], evt["GenJetNu_mass"][min_idx])
            #features["GenJetNu1_pt"] = evt["GenJetNu_pt"][min_idx]
            #features["GenJetNu1_eta"] = evt["GenJetNu_eta"][min_idx]
            #features["GenJetNu1_phi"] = evt["GenJetNu_phi"][min_idx]
            #features["GenJetNu1_mass"] = evt["GenJetNu_mass"][min_idx]
            #features["dR_genJet1_genJetNu1"] = genJet.DeltaR(genJetNu)
        else:
            pass
            #features["GenJet1_pt"] =-99
            #features["GenJet1_eta"] =-99
            #features["GenJet1_phi"] =-99
            #features["GenJet1_mass"] =-99
            #features["GenJet1_partonMotherPdgId"] = -99

            #features["GenJetNu1_pt"] = -99
            #features["GenJetNu1_eta"] = -99
            #features["GenJetNu1_phi"] = -99
            #features["GenJetNu1_mass"] = -99
            #features["dR_genJet1_genJetNu1"] = -99
        if evt["Jet_genJetIdx"][selected2]!=-1:
            #Matched with genJet
            #features["GenJet2_pt"] =evt["GenJet_pt"][evt["Jet_genJetIdx"][selected2]]
            #features["GenJet2_eta"] =evt["GenJet_eta"][evt["Jet_genJetIdx"][selected2]]
            #features["GenJet2_phi"] =evt["GenJet_phi"][evt["Jet_genJetIdx"][selected2]]
            #features["GenJet2_mass"] =evt["GenJet_mass"][evt["Jet_genJetIdx"][selected2]]                
            #features["GenJet2_partonMotherPdgId"] = evt["GenJet_partonMotherPdgId"][evt["Jet_genJetIdx"][selected2]]                
            pass

            
            #genJet = ROOT.TLorentzVector(0.,0.,0.,0.)
            #genJetNu = ROOT.TLorentzVector(0.,0.,0.,0.)
            #genJet.SetPtEtaPhiM(evt["GenJet_pt"][evt["Jet_genJetIdx"][selected2]], evt["GenJet_eta"][evt["Jet_genJetIdx"][selected2]], evt["GenJet_phi"][evt["Jet_genJetIdx"][selected2]], evt["GenJet_mass"][evt["Jet_genJetIdx"][selected2]])

            #min_dR, min_idx = 999, 999
            #for j_nu in range(len(evt["GenJetNu_pt"])):
            #    genJetNu.SetPtEtaPhiM(evt["GenJetNu_pt"][j_nu], evt["GenJetNu_eta"][j_nu], evt["GenJetNu_phi"][j_nu], evt["GenJetNu_mass"][j_nu])
            #    if genJetNu.DeltaR(genJet) < min_dR:
            #        min_dR = genJetNu.DeltaR(genJet)
            #        min_idx = j_nu
            #genJetNu.SetPtEtaPhiM(evt["GenJetNu_pt"][min_idx], evt["GenJetNu_eta"][min_idx], evt["GenJetNu_phi"][min_idx], evt["GenJetNu_mass"][min_idx])
            #features["GenJetNu2_pt"] = evt["GenJetNu_pt"][min_idx]
            #features["GenJetNu2_eta"] = evt["GenJetNu_eta"][min_idx]
            #features["GenJetNu2_phi"] = evt["GenJetNu_phi"][min_idx]
            #features["GenJetNu2_mass"] = evt["GenJetNu_mass"][min_idx]
            #features["dR_genJet2_genJetNu2"] = genJet.DeltaR(genJetNu)
        else:
            pass
            #features["GenJet2_pt"] =-99
            #features["GenJet2_eta"] =-99
            #features["GenJet2_phi"] =-99
            #features["GenJet2_mass"] =-99
            #features["GenJet2_partonMotherPdgId"] =-99

            #features["GenJetNu2_pt"] = -99
            #features["GenJetNu2_eta"] = -99
            #features["GenJetNu2_phi"] = -99
            #features["GenJetNu2_mass"] = -99
            #features["dR_genJet2_genJetNu2"] = -99
    if (('ZJetsToQQ' in processName) | ('EWKZJets' in processName)):
        pt = np.array(evt_gen["LHEPart_pt"])
        mask = np.array(evt_gen["LHEPart_pdgId"])==23
        pt_Z = (pt[mask])
        k_factor = getZ_KFactor(pt=pt_Z)

    else:
        k_factor = 1.
    features["Zqq_NLO_kfactor"] = k_factor
    weight_genOnly *= k_factor
    return features, weight_genOnly


def fill_event_variables(evt, maskJets):
    features = {}
    

    # Event Jet variables
    Jet_px_Masked = evt["Jet_pt"][maskJets] * np.cos(evt["Jet_phi"][maskJets])
    Jet_py_Masked = evt["Jet_pt"][maskJets] * np.sin(evt["Jet_phi"][maskJets])
    Jet_pz_Masked = evt["Jet_pt"][maskJets] * np.sinh(evt["Jet_eta"][maskJets])


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
    features['sphericity'] = sphericity

    features['lambda1'] = lambda1
    features['lambda2'] = lambda2
    features['lambda3'] = lambda3
# evt["nJet"]s
    features['nJets'] = int(np.sum(maskJets))
    features['nJets_20'] = int(np.sum(evt["Jet_pt"][maskJets]>=20))
    features['nJets_30'] = int(np.sum(evt["Jet_pt"][maskJets]>=30))
    features['nJets_50'] = int(np.sum(evt["Jet_pt"][maskJets]>=50))
    
    # Loop over indices idx of masked Jets
    ht = 0
    for idx in np.arange(evt["nJet"])[maskJets]:
        ht = ht+evt["Jet_pt"][idx]
    features['ht'] = (np.float32(ht))
    features['nMuons'] = evt["nMuon"]
    features['nIsoMuons'] = np.sum(evt["Muon_pfRelIso04_all"]<0.15)
    features['nElectrons'] = evt["nElectron"]
    features['nSV'] = int(evt["nSV"])
    #evt["Jet_phi"]T = np.linspace(0, 3.14, 500)
    #evt["Jet_pt"]_extended = evt["Jet_pt"][maskJets, np.newaxis]  # Shape (evt["nJet"], 1)
    #evt["Jet_phi"]_extended = evt["Jet_phi"][maskJets, np.newaxis]  # Shape (evt["nJet"], 1)
    #T_values = np.sum(evt["Jet_pt"]_extended * np.abs(np.cos(evt["Jet_phi"]_extended - evt["Jet_phi"]T)), axis=0)


    # Find the maximum value
    #T_max = np.max(T_values)
    #phiT_max = evt["Jet_phi"]T[np.argmax(T_values)]
    #features_['T_max'] = T_max
    #features_['phiT_max'] = phiT_max

    return features

def fill_rest_features(dijet_vec, jet1_rest_vec, jet2_rest_vec):
    # This was checked.
    # using the same vector for boosting and trasnforming you find E=m, px=0, py=0, pz=0
    features={}
    features['jet1_rest_pt'] = jet1_rest_vec.Pt()
    features['jet1_rest_eta'] = jet1_rest_vec.Eta()
    features['jet1_rest_phi'] = jet1_rest_vec.Phi()
    features['jet2_rest_pt'] = jet2_rest_vec.Pt()
    features['jet2_rest_eta'] = jet2_rest_vec.Eta()
    features['jet2_rest_phi'] = jet2_rest_vec.Phi()
    # Mass does not change under boost. It's invariant

    # Compute the cosine of the helicity angle
    # The helicity angle is the angle between jet1's momentum in the rest frame
    # and the boost direction of the bb system in the lab frame.
    cos_theta_star = jet1_rest_vec.Vect().Dot(dijet_vec.Vect()) / (jet1_rest_vec.Vect().Mag() * dijet_vec.Vect().Mag())
    features['cos_theta_star'] = cos_theta_star
    return features


def fill_uncorrected_features(evt, selected1, selected2, jet1_uncor, jet2_uncor):
    features={}
    features['jet1_pt_uncor'] = evt["Jet_pt"][selected1]
    features['jet1_mass_uncor'] = evt["Jet_mass"][selected1]
    features['jet2_pt_uncor'] = evt["Jet_pt"][selected2]
    features['jet2_mass_uncor'] = evt["Jet_mass"][selected2]
    features['dijet_pt_uncor'] = (jet1_uncor + jet2_uncor).Pt()
    features['dijet_mass_uncor'] = (jet1_uncor + jet2_uncor).M()
    return features