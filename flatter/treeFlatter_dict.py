import sys, re, yaml
import pandas as pd
import numpy as np
import ROOT
import uproot
from functions import load_mapping_dict
import awkward as ak
from correctionlib import _core
import gzip
from jetsSelector import jetsSelector_new
#from getJetSysJEC import getJetSysJEC
import time
import math
import json
from addFeatures import addFeatureToFile
import os

sys.path.append("/t3home/gcelotto/ggHbb/PNN/slurm/mjj/")
from predictTorch import predict
from functions import getDfProcesses_v2
from helpers_flatter import get_event_branches, fill_ttbar_CR_features, fill_jet_features, fill_dijet_features, fill_trig_muon_features, get_event_genBranches, fill_gen_info, fill_event_variables, fill_rest_features, fill_uncorrected_features
from treeFlatter_dict_getSFs import get_btag_map_efficiency, get_muon_recoSF, get_trig_SF, log_bad_btag_sf, get_btag_SF

# Feature naming conventions:
# - jetX_*      : reco jet features
# - jetX_muon_* : muon inside jet
# - gen*        : generator-level
# - *_sf        : multiplicative scale factors
# 
def has_non_finite(features):
    for k, v in features.items():
        try:
            if not np.isfinite(v):
                return True, k, v
        except TypeError:
            # non-numeric (should not happen, but be safe)
            continue
    return False, None, None
def log_bad_event(logfile, process, fileNumber, event, key, value):
    with open(logfile, "a+") as f:
        f.write(
            f"{process} {fileNumber} ev={event} bad_feature={key} value={value}\n"
        )

def treeFlatten(fileName, maxEntries, maxJet, pN, processName, method, isJEC, verbose, JECname, isMC, folder_cfg, fileNumber):
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
    if isMC:
        effMC_path = folder_cfg["trig_SF_folder"]+"/trgMu_effMC_UL.root"
        effData_path = folder_cfg["trig_SF_folder"]+"/trgMu_effData_UL.root"
        effMC_rootFile = ROOT.TFile(effMC_path, "READ")
        effData_rootFile = ROOT.TFile(effData_path, "READ")

        if not effMC_rootFile or effMC_rootFile.IsZombie():
            raise RuntimeError(f"Failed to open ROOT file: {effMC_path}")
        if not effData_rootFile or effData_rootFile.IsZombie():
            raise RuntimeError(f"Failed to open ROOT file: {effData_path}")
    else:
        pass

    
    # Open the WorkingPoint correction lib
    # /cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2018-UL-NanoAODv9/latest/btagging.json.gz
    fname_jec = folder_cfg["JEC_unc_folder"]+"/jet_jerc_2.json.gz"
    ceval = _core.CorrectionSet.from_file(fname_jec)
    JECunc = ceval["Summer19UL18_V5_MC_Regrouped_Total_AK4PFchs"]


    fname = folder_cfg["btag_SF_folder"]+"/btagging.json.gz"
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

    # Open the evt["Jet_puId"] Scale Factor Evaluator
    fname = folder_cfg["jet_pu_id_sf_folder"]+"/jmar.json.gz"
    if fname.endswith(".json.gz"):
        
        with gzip.open(fname,'rt') as file:
            data = file.read().strip()
            puId_SF_evaluator = _core.CorrectionSet.from_string(data)
    else:
        puId_SF_evaluator = _core.CorrectionSet.from_file(fname)

    # Open the map of efficiency for btag SF
    btagMapsExist=False
    processNameForBtag = "GluGluHToBBMINLO" if "GluGluHToBBMINLO_" in processName else processName #tr and private get the same maps
    if os.path.exists(folder_cfg["BTagEfficiencyMap_folder"]+f"/btag_efficiency_map_{processNameForBtag}_T.json"):
        btagMapsExist=True
        eff_maps_cache_btag = {}
        for wp_ in ["L", "M", "T"]:
            print(f"Opening the map {processName}_{wp_}.json ...")
            with open(folder_cfg["BTagEfficiencyMap_folder"]+f"/btag_efficiency_map_{processNameForBtag}_{wp_}.json", 'r') as f:
                eff_maps_cache_btag[wp_] = json.load(f)

    with open(folder_cfg["LeptonSF_folder"]+"/RecoEfficiencies_MediumPtMuons.json") as f:
        muon_RECO_map = json.load(f)

    with open(folder_cfg["LeptonSF_folder"]+"/IDEfficiencies_MediumPtMuons.json") as f:
        muon_ID_map = json.load(f)

    with open(folder_cfg["LeptonSF_folder"]+"/ISOEfficiencies_MediumPtMuons.json") as f:
        muon_ISO_map = json.load(f)

    electrons_SF_map = _core.CorrectionSet.from_file(folder_cfg["LeptonSF_folder"]+'/electron.json.gz')


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
        evt = get_event_branches(branches, ev,isMC)


##############################
#
#           Filling the rows
#
##############################
        event_weight  = 1.0
        maskJets = (evt["Jet_jetId"]==6) & ((evt["Jet_pt"]>=50) | (evt["Jet_puId"]>=4)) & (evt["Jet_pt"]>=20) & (abs(evt["Jet_eta"])<2.5)
        # puId==0 means 000: fail all PU ID;
        # puId==1 means 001: pass loose ID, fail medium, fail tight;
        # puId==3 means 011: pass loose and medium ID, fail tight;
        # puId==7 means 111: pass loose, medium, tight ID.
        

        jetsToCheck = np.min([maxJet, evt["nJet"]])                                
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet4  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet1_uncor  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2_uncor  = ROOT.TLorentzVector(0.,0.,0.,0.)
        
        features_["HEM_issue"] = int(np.sum((evt["Jet_pt"]>20) & (evt["Jet_jetId"]==6) & ((evt["Jet_pt"]>50) | (evt["Jet_puId"]>=4)) & (evt["Jet_eta"]>-2.5) & (evt["Jet_eta"]<-1.3) & (evt["Jet_phi"]>-1.57) & (evt["Jet_phi"]<-0.87))>=1)
        # to be checked
        selected1, selected2, muonIdx1, muonIdx2, LeadingMuons_inJets = jetsSelector_new(    nJet = evt["nJet"], Jet_eta = evt["Jet_eta"], Jet_muonIdx1 = evt["Jet_muonIdx1"], Jet_muonIdx2 = evt["Jet_muonIdx2"],
                                                                    Muon_isTriggering = evt["Muon_isTriggering"], Muon_pt = evt["Muon_pt"], Muon_eta = evt["Muon_eta"], Muon_dxy = evt["Muon_dxy"], Muon_dxyErr=evt["Muon_dxyErr"],
                                                                    jetsToCheck = jetsToCheck, Jet_btagDeepFlavB = evt["Jet_btagDeepFlavB"], Jet_puId = evt["Jet_puId"], Jet_jetId = evt["Jet_jetId"],
                                                                    maskJets = maskJets, method = method, Jet_pt = evt["Jet_pt"])
        verbose and print("Ev : %d | %d %d %d %d"%(ev, selected1, selected2, muonIdx1, muonIdx2))
        # LeadingMuons_inJets represents the muons that need to be checked to calculate the trigger SF
        if selected1==999:
            continue
        if selected2==999:
            assert False
        #evt["Jet_breg2018"] has to be applied on energy as well : https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsWG/BJetRegression
        #Correct the dataframes later
        energy1 = np.sqrt(evt["Jet_pt"][selected1]**2 + (evt["Jet_pt"][selected1]*np.sinh(evt["Jet_eta"][selected1]))**2 +  evt["Jet_mass"][selected1]**2)
        energy2 = np.sqrt(evt["Jet_pt"][selected2]**2 + (evt["Jet_pt"][selected2]*np.sinh(evt["Jet_eta"][selected2]))**2 +  evt["Jet_mass"][selected2]**2)
        jet1.SetPtEtaPhiE(evt["Jet_pt"][selected1]*evt["Jet_bReg2018"][selected1], evt["Jet_eta"][selected1], evt["Jet_phi"][selected1], energy1*evt["Jet_bReg2018"][selected1]    )
        jet2.SetPtEtaPhiE(evt["Jet_pt"][selected2]*evt["Jet_bReg2018"][selected2], evt["Jet_eta"][selected2], evt["Jet_phi"][selected2], energy2*evt["Jet_bReg2018"][selected2]    )
        dijet = jet1 + jet2

        # Jet1
        features_.update(fill_jet_features("jet1", selected1, evt, jet1, dijet=dijet, isMC=isMC, JECunc=JECunc))
        # Jet2
        features_.update(fill_jet_features("jet2", selected2, evt, jet2, dijet=dijet, isMC=isMC, JECunc=JECunc))
        if muonIdx2!=999:
            features_["jet2_has_trigMuon"] = 1
        else:
            features_["jet2_has_trigMuon"] = 0

        # 3rd Jets
        if np.sum(maskJets)>2:
            for i in np.arange(evt["nJet"])[maskJets]:
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    selected3 = i
                    jet3.SetPtEtaPhiM(evt["Jet_pt"][selected3],evt["Jet_eta"][selected3],evt["Jet_phi"][selected3],evt["Jet_mass"][selected3])
                    features_.update(fill_jet_features("jet3", selected3, evt, jet3, dijet=dijet, jetIsPresent=True, isMC=isMC, JECunc=JECunc))
                    break
        else:
            selected3 = None
            features_.update(fill_jet_features("jet3", idx=None, evt=evt, jet_vec=None, dijet=None, jetIsPresent=False, isMC=isMC, JECunc=JECunc))
        
        # 4th Jets
        if np.sum(maskJets)>3:
            for i in np.arange(evt["nJet"])[maskJets]:
                if ((i ==selected1) | (i==selected2) | (i==selected3)):
                    continue
                else:
                    selected4 = i
                    jet4.SetPtEtaPhiM(evt["Jet_pt"][selected4],evt["Jet_eta"][selected4],evt["Jet_phi"][selected4],evt["Jet_mass"][selected4])
                    features_.update(fill_jet_features("jet4", idx=selected4, evt=evt, jet_vec=jet4, dijet=dijet, jetIsPresent=True))
                    break
        else:
            selected4 = None
            features_.update(fill_jet_features("jet4", idx=None, evt=evt, jet_vec=None, dijet=None, jetIsPresent=False))
            
        # Dijet
        dijet_features = fill_dijet_features(dijet, jet1, jet2, evt)
        features_.update(dijet_features)

        #boost_vector = -dijet.BoostVector()  # Boost to the bb system's rest frame
        #jet1_rest = ROOT.TLorentzVector(jet1)  # Make a copy to boost
        #jet1_rest.Boost(boost_vector)     # Boost jet1 into the rest frame
        #jet2_rest = ROOT.TLorentzVector(jet2)  # Make a copy to boost
        #jet2_rest.Boost(boost_vector)     # Boost jet1 into the rest frame
        # Rest Features
        #rest_features = fill_rest_features(dijet_vec=dijet, jet1_rest_vec=jet1_rest, jet2_rest_vec=jet2_rest)
        #features_.update(rest_features)
        # Event  Variables (Shape and global features)
        event_variables = fill_event_variables(evt, maskJets)
        features_.update(event_variables)

    # uncorrected quantities
        jet1_uncor.SetPtEtaPhiM(evt["Jet_pt"][selected1], evt["Jet_eta"][selected1], evt["Jet_phi"][selected1], evt["Jet_mass"][selected1])
        jet2_uncor.SetPtEtaPhiM(evt["Jet_pt"][selected2], evt["Jet_eta"][selected2], evt["Jet_phi"][selected2], evt["Jet_mass"][selected2])
        uncorrected_features = fill_uncorrected_features(evt, selected1, selected2, jet1_uncor, jet2_uncor)
        features_.update(uncorrected_features)
        


## ttbar CR

        tt_features, tt_weight = fill_ttbar_CR_features(evt, isMC, muonIdx1, muonIdx2,processName, muon_RECO_map, muon_ID_map, muon_ISO_map, electrons_SF_map)
        
        features_.update(tt_features)
        event_weight *= tt_weight

# Trig Muon
        muon = ROOT.TLorentzVector(0., 0., 0., 0.)
        muon.SetPtEtaPhiM(evt["Muon_pt"][muonIdx1], evt["Muon_eta"][muonIdx1], evt["Muon_phi"][muonIdx1], evt["Muon_mass"][muonIdx1])
        features_muon, muon_weight = fill_trig_muon_features(muon,muonIdx1, jet1, "1", muon_RECO_map, evt)
        features_.update(features_muon)
        event_weight *= muon_weight


        jet2_muon_isTriggering = False
        isMuon2 = False
        # R1
        if (muonIdx2 != 999):# & (evt["Muon_pt"][muonIdx2]>9) & (np.abs(evt["Muon_eta"][muonIdx2])<1.5) & (abs(evt["Muon_dxy"][muonIdx2]/evt["Muon_dxyErr"][muonIdx2])>6):
            # Second muon is triggering as Mu9IP6
            #assert evt["Muon_isTriggering"][muonIdx2] == True
            jet2_muon_isTriggering = int(evt["Muon_isTriggering"][muonIdx2])
            isMuon2 = True
            muon2 = ROOT.TLorentzVector(0., 0., 0., 0.)
            muon2.SetPtEtaPhiM(evt["Muon_pt"][muonIdx2], evt["Muon_eta"][muonIdx2], evt["Muon_phi"][muonIdx2], evt["Muon_mass"][muonIdx2])
            features_muon2, muon2_weight =fill_trig_muon_features(muon2,muonIdx2, jet2, "2", muon_RECO_map, evt, has_muon=isMuon2)
            features_.update(features_muon2)
            event_weight *= muon2_weight
            

            
        else:
            features_muon2, muon2_weight = fill_trig_muon_features(None,None, jet2, "2",  muon_RECO_map, evt, has_muon=isMuon2)
            features_.update(features_muon2)
            event_weight *= muon2_weight
        features_["jet2_muon_isTriggering"] = int(jet2_muon_isTriggering)
        features_['dimuon_mass'] = np.float32((muon+muon2).M()) if isMuon2 else 0.106
# Trigger
        features_['jet1_muon_fired_HLT_Mu12_IP6'] = int(bool(evt["Muon_fired_HLT_Mu12_IP6"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu10p5_IP3p5'] = int(bool(evt["Muon_fired_HLT_Mu10p5_IP3p5"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu8p5_IP3p5'] = int(bool(evt["Muon_fired_HLT_Mu8p5_IP3p5"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu7_IP4'] = int(bool(evt["Muon_fired_HLT_Mu7_IP4"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu8_IP3'] = int(bool(evt["Muon_fired_HLT_Mu8_IP3"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu8_IP5'] = int(bool(evt["Muon_fired_HLT_Mu8_IP5"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu8_IP6'] = int(bool(evt["Muon_fired_HLT_Mu8_IP6"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu9_IP4'] = int(bool(evt["Muon_fired_HLT_Mu9_IP4"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu9_IP5'] = int(bool(evt["Muon_fired_HLT_Mu9_IP5"][muonIdx1]))
        features_['jet1_muon_fired_HLT_Mu9_IP6'] = int(bool(evt["Muon_fired_HLT_Mu9_IP6"][muonIdx1]))

        if isMC:
            evt_gen = get_event_genBranches(branches, ev, processName=processName)
            features_gen, weight_gen  = fill_gen_info(evt, evt_gen, jet1, jet2, selected1, selected2, selected3, isMC, processName)
            features_.update(features_gen)
            event_weight *= weight_gen
            # https://indico.ihep.ac.cn/event/16093/contributions/43970/attachments/21148/24086/SF.pdf
            # PDF
            #if len(evt_gen["LHEPdfWeight"])==101:
            #    for k in range(1,101):
            #        features_[f"LHEPdfWeight_{k}"] = evt_gen["LHEPdfWeight"][k]
            if ("HToBB" in processName) | ("TTTo" in processName):
                for k in range(1,101):
                    features_[f"LHEPdfWeight_{k}"] = evt_gen["LHEPdfWeight"][k]
                # Alpha_S
                features_[f"LHEAlphasWeight_down"] = evt_gen["LHEPdfWeight"][101]
                features_[f"LHEAlphasWeight_up"] = evt_gen["LHEPdfWeight"][102]
                # MuR and MuF
                scale_order = [
                        (0.5, 0.5),(1.0, 0.5),(2.0, 0.5),
                        (0.5, 1.0),(1.0, 1.0),(2.0, 1.0),
                        (0.5, 2.0),(1.0, 2.0),(2.0, 2.0)
                    ]
                #if len(evt_gen["LHEScaleWeight"])==8:
                #    scale_order.remove((1.0,1.0))
                for idx, (muf, mur) in enumerate(scale_order):
                    muf_str = "down" if muf==0.5 else ("up" if muf==2.0 else "nom")
                    mur_str = "down" if mur==0.5 else ("up" if mur==2.0 else "nom")
                    features_[f"LHEScaleWeight_MuF{muf_str}_MuR{mur_str}"] = evt_gen["LHEScaleWeight"][idx]
                # ISR and FR
                ISR_FSR_order = [
                    (2.0,1.0), (1.0,2.0),
                    (0.5,1.0), (1.0,0.5)
                ]
                if len(evt_gen["PSWeight"])==1:
                    for idx, (isr, fsr) in enumerate(ISR_FSR_order):
                        isr_str = "down" if isr==0.5 else ("up" if isr==2.0 else "nom")
                        fsr_str = "down" if fsr==0.5 else ("up" if fsr==2.0 else "nom")
                        features_[f"PSWeight_ISR{isr_str}_FSR{fsr_str}"] = 1
                else:
                    for idx, (isr, fsr) in enumerate(ISR_FSR_order):
                        isr_str = "down" if isr==0.5 else ("up" if isr==2.0 else "nom")
                        fsr_str = "down" if fsr==0.5 else ("up" if fsr==2.0 else "nom")
                        features_[f"PSWeight_ISR{isr_str}_FSR{fsr_str}"] = evt_gen["PSWeight"][idx]
            
            #deltaPDF = [evt_gen["LHEPdfWeight"][k] - 1 for k in range(1, 101)]
            #deltaPDF = np.array(deltaPDF, dtype=np.float32)
            #deltaPDF = np.sqrt(np.sum(deltaPDF**2))
            #features_["deltaPDF"] = deltaPDF
# Gen Info
            # PileupID SF computation
            for syst in ["nom", "up", "down"]:
                jet_pileupId_SF  =1.
                for j in np.arange(evt["nJet"])[maskJets]:
                    if (evt["Jet_pt"][j]<50) & (evt["Jet_genJetIdx"][j]>-1):
                        wp = "L"
                        current_pileupID_SF = puId_SF_evaluator["PUJetID_eff"].evaluate(float(evt["Jet_eta"][j]), float(evt["Jet_pt"][j]), syst, wp)
                        jet_pileupId_SF *= current_pileupID_SF
                    else:
                        jet_pileupId_SF *= 1
                features_["jet_pileupId_SF_"+syst]  = jet_pileupId_SF
                if syst=="nom":
                    event_weight *= jet_pileupId_SF

# BTag SF and Variations
            btag_dictionary = get_btag_SF(btagMapsExist, evt, maskJets, corrDeepJet_FixedWP_comb, corrDeepJet_FixedWP_light, eff_maps_cache_btag, wp_converter, processName)
            features_["btag_sf_hf_central"] = btag_dictionary["btag_sf_hf_central"]
            features_["btag_sf_hf_up"] = btag_dictionary["btag_sf_hf_up"]
            features_["btag_sf_hf_down"] = btag_dictionary["btag_sf_hf_down"]
            features_["btag_sf_lightf_central"] = btag_dictionary["btag_sf_lightf_central"]
            features_["btag_sf_light_up"] = btag_dictionary["btag_sf_light_up"]
            features_["btag_sf_light_down"] = btag_dictionary["btag_sf_light_down"]

            event_weight *= btag_dictionary["btag_sf_hf_central"] * btag_dictionary["btag_sf_lightf_central"]

        

            trig_sf, err_trig_sf = get_trig_SF(LeadingMuons_inJets, 
                                  evt["Muon_isTriggering"],
                                  evt["Muon_pt"],
                                  evt["Muon_dxy"],
                                  evt["Muon_dxyErr"],
                                  #muon_pt1=features_["jet1_muon_pt"],
                                  #muon_sIP1=abs(features_["jet1_muon_dxySig"]),
                                  #muon_pt2=features_["jet2_muon_pt"],
                                  #muon_sIP2=abs(features_["jet2_muon_dxySig"]),
                                  muon_2_isTriggering=features_["jet2_muon_isTriggering"],
                                  effData_rootfile=effData_rootFile,
                                  effMC_rootfile=effMC_rootFile)
            features_['trig_sf'] = trig_sf
            features_['err_trig_sf'] = err_trig_sf
            event_weight  *= np.float32(trig_sf)
# LHEScaleWeight
        # Code here

        assert evt["Muon_isTriggering"][muonIdx1]
        features_["flat_weight"] = event_weight
        bad, key, val = has_non_finite(features_)
        if bad:
            log_bad_event(
                logfile=folder_cfg["log_bad_event"],
                process=processName,
                fileNumber=fileNumber,
                event=ev,
                key=key,
                value=val,
            )
            continue
        file_.append(features_)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    return file_


def main(fileName, maxEntries, maxJet, pN, fullProcessName, method, isJEC, verbose, isMC, fileNumber):
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
    with open("/t3home/gcelotto/ggHbb/flatter/treeFlatter_cfg.yaml", "r") as f:
        folder_cfg = yaml.safe_load(f)
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, pN=pN, processName=processName, method=method, isJEC=isJEC, verbose=verbose, JECname=JECname, isMC=isMC, folder_cfg=folder_cfg, fileNumber=fileNumber)
    df=pd.DataFrame(fileData)
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
        df['xsection']=1
    else:
        PU_map = load_mapping_dict(folder_cfg["PU_Ratio_folder"]+'/PU_PVtoPUSF.json')
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0
        df['flat_weight'] *=   df['PU_SF']

        dfProcessesMC = getDfProcesses_v2()[0]
        xsections = dfProcessesMC.iloc[pN].xsection
        df['xsection']=xsections


    df = addFeatureToFile(df)
    # NN prediction?
    if "QCD_topology" in df.columns:
        df_pred = df.drop(['QCD_topology'], axis=1)
        data_predictions1, data_predictions_qm = predict(df_pred, modelName="Jan21_3_50p0", boosted=3, quantile_matching=True, run=2)
    else:
        data_predictions1, data_predictions_qm = predict(df, modelName="Jan21_3_50p0", boosted=3, quantile_matching=True, run=2)
    
    df["NN"] = data_predictions1
    df["NN_qm"] = data_predictions_qm


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
    fileNumber = int(sys.argv[10])
    print("calling main", flush=True)
    main(fileName, maxEntries, maxJet, pN, fullProcessName, method, isJEC, verbose, isMC, fileNumber)