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

syst2 = []

def treeFlatten(fileName, maxEntries, maxJet, isMC, processName):
    maxEntries=int(maxEntries)
    maxJet=int(maxJet)
    #isMC=int(isMC)
    print("fileName", fileName)
    print("maxEntries", maxEntries)
    #print("isMC", isMC)
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
    #if (isMC==2) | (isMC==20) | (isMC==21) | (isMC==22) | (isMC==23) | (isMC==36):
    #    GenPart_pdgId = branches["GenPart_pdgId"]
    #    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"]
    #    maskBB = ak.sum((abs(GenPart_pdgId)==5) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2
    #    myrange = np.arange(tree.num_entries)[~maskBB]
#
    #elif (isMC==45) | (isMC==46) | (isMC==47) | (isMC==48) | (isMC==49) | (isMC==50):
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
    corrDeepJet_FixedWP_muJets = cset["deepJet_mujets"]
    #corrDeepJet_shape           = cset["deepJet_shape"]
    #btag_systs = ['central','down','down_jes', 'down_pileup', 'down_statistic', 'down_type3', 'up', 'up_jes', 'up_pileup', 'up_statistic', 'up_type3', 'down_correlated', 'down_uncorrelated', 'up_correlated', 'up_uncorrelated']
    #wp_converter = cset["deepJet_wp_values"]
    for ev in myrange:
        
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

        Jet_vtx3dL                  = branches["Jet_vtx3dL"][ev]
        Jet_vtx3deL                 = branches["Jet_vtx3deL"][ev]
        Jet_vtxPt                   = branches["Jet_vtxPt"][ev]
        Jet_vtxMass                 = branches["Jet_vtxMass"][ev]
        Jet_vtxNtrk                 = branches["Jet_vtxNtrk"][ev]



        Jet_qgl                     = branches["Jet_qgl"][ev]
        Jet_nMuons                  = branches["Jet_nMuons"][ev]
        Jet_nConstituents           = branches["Jet_nConstituents"][ev]
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
    # Electrons Tracks
        nElectron                   = branches["nElectron"][ev]
        Electron_pfRelIso           = branches["Electron_pfRelIso"][ev]
        #nProbeTracks                = branches["nProbeTracks"][ev]


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

        if 'Data' in processName:
            Pileup_nTrueInt = 0
        else:
            Pileup_nTrueInt         = branches["Pileup_nTrueInt"][ev]
            Jet_genJetIdx           = branches["Jet_genJetIdx"][ev]
            GenJet_hadronFlavour    = branches["GenJet_hadronFlavour"][ev]
            genWeight    = branches["genWeight"][ev]

        jetsToCheck = np.min([maxJet, nJet])                                 # !!! max 4 jets to check   !!!
        jet1  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet2  = ROOT.TLorentzVector(0.,0.,0.,0.)
        jet3  = ROOT.TLorentzVector(0.,0.,0.,0.)
        dijet = ROOT.TLorentzVector(0.,0.,0.,0.)
        jetsToCheck = np.min([maxJet, nJet])
        
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId)

        if selected1==999:
            #print("skipped")
            continue
        if selected2==999:
            assert False

        
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
        features_.append(Jet_btagPNetB[selected1])
        features_.append(selected1)
        features_.append(Jet_rawFactor[selected1])
        features_.append(Jet_bReg2018[selected1])
        features_.append(Jet_PNetRegPtRawCorr[selected1])
        features_.append(Jet_PNetRegPtRawCorrNeutrino[selected1])
        
        features_.append(Jet_jetId[selected1])
        features_.append(Jet_puId[selected1])
        features_.append(Jet_vtxPt[selected1])
        features_.append(Jet_vtxMass[selected1])
        features_.append(Jet_vtxNtrk[selected1])
        jet1_sv_3dSig = Jet_vtx3dL[selected1]/Jet_vtx3deL[selected1] if Jet_vtx3dL[selected1]!=0 else 0
        features_.append(jet1_sv_3dSig)
        
        if wp_converter.evaluate("L") <= Jet_btagDeepFlavB[selected1] < wp_converter.evaluate("M"):
            wp = "L"
        elif wp_converter.evaluate("M") <= Jet_btagDeepFlavB[selected1] < wp_converter.evaluate("T"):
            wp = "M"
        elif wp_converter.evaluate("T") <= Jet_btagDeepFlavB[selected1]:
            wp = "T"
        else:
            wp = None  # Optional: handle case where score is below "L"
        #if (isMC!=0) & (isMC!=39) & (isMC!=57):
        #    for syst in ["central", "up", "down"]:
        #        if (Jet_genJetIdx[selected1]!= -1) & (wp is not None): 
        #            if abs(GenJet_hadronFlavour[Jet_genJetIdx[selected1]])!=0:
        #                jet1_btag_sf = corrDeepJet_FixedWP_muJets.evaluate(syst, wp, abs(GenJet_hadronFlavour[Jet_genJetIdx[selected1]]), abs(jet1.Eta()), jet1.Pt())
        #            else:
        #                jet1_btag_sf = -1 # -1 for usdg
        #        else:
        #            jet1_btag_sf = 0    # 0 for non matched jets or jets with bscore below L
        #        features_.append(jet1_btag_sf)

        


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
        features_.append(Jet_btagPNetB[selected2])
        features_.append(selected2)
        features_.append(Jet_rawFactor[selected2])
        features_.append(Jet_bReg2018[selected2])
        features_.append(Jet_PNetRegPtRawCorr[selected2])
        features_.append(Jet_PNetRegPtRawCorrNeutrino[selected2])
        features_.append(Jet_jetId[selected2])
        features_.append(Jet_puId[selected2])
        features_.append(Jet_vtxPt[selected2])
        features_.append(Jet_vtxMass[selected2])
        features_.append(Jet_vtxNtrk[selected2])
        jet2_sv_3dSig = Jet_vtx3dL[selected2]/Jet_vtx3deL[selected2] if Jet_vtx3dL[selected2]!=0 else 0
        features_.append(jet2_sv_3dSig)
        
        #if (isMC!=0) & (isMC!=39) & (isMC!=57):
        #    for syst in syst2:
        #        if (Jet_genJetIdx[selected1]!= -1) & (wp is not None): 
        #            if abs(GenJet_hadronFlavour[Jet_genJetIdx[selected1]])!=0:
        #                jet2_btag_sf = corrDeepJet_shape.evaluate(syst, abs(GenJet_hadronFlavour[Jet_genJetIdx[selected2]]), abs(jet2.Eta()), jet2.Pt(), float(Jet_btagDeepFlavB[selected2]))
        #            else:
        #                jet2_btag_sf = -1 # -1 for usdg
        #        else:
        #            jet2_btag_sf = 0    
        #        features_.append(jet2_btag_sf)


        if len(Jet_pt)>2:
            for i in range(len(Jet_pt)):
                if ((i ==selected1) | (i==selected2)):
                    continue
                else:
                    jet3.SetPtEtaPhiM(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i])
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
                    features_.append(np.float32(Jet_btagDeepFlavB[i]))
                    features_.append(jet3.DeltaR(dijet))
                    break
        else:
            features_.append(np.float32(0)) #pt
            features_.append(np.float32(0))
            features_.append(np.float32(0))
            features_.append(np.float32(0)) #mass
            features_.append(int(0))                
            features_.append(np.float32(0))         # pnet
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

        boost_vector = -dijet.BoostVector()  # Boost to the bb system's rest frame

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


        Jet_px = Jet_pt * np.cos(Jet_phi)
        Jet_py = Jet_pt * np.sin(Jet_phi)
        Jet_pz = Jet_pt * np.sinh(Jet_eta)

        #jet1_components = [jet1.Px(), jet1.Py(), jet1.Pz()]
        #jet2_components = [jet2.Px(), jet2.Py(), jet2.Pz()]

        # Step 1: Extract the 3-momentum components for all jets
        ptot_squared = np.sum(Jet_px**2 + Jet_py**2 + Jet_pz**2)
        S_xx = np.sum(Jet_px * Jet_px) / ptot_squared
        S_xy = np.sum(Jet_px * Jet_py) / ptot_squared
        S_xz = np.sum(Jet_px * Jet_pz) / ptot_squared
        S_yy = np.sum(Jet_py * Jet_py) / ptot_squared
        S_yz = np.sum(Jet_py * Jet_pz) / ptot_squared
        S_zz = np.sum(Jet_pz * Jet_pz) / ptot_squared

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
        
        # planar approx lambda1 = lambda2 >> lambda 3
        #planarity = 2 * (lambda2 - lambda3) / (lambda1 + lambda2 + lambda3)
        #features_.append(planarity)

        features_.append(lambda1)
        features_.append(lambda2)
        features_.append(lambda3)

        # thrust axis
        Jet_phiT = np.linspace(0, 3.14, 500)
        Jet_pt_extended = Jet_pt[:, np.newaxis]  # Shape (nJet, 1)
        Jet_phi_extended = Jet_phi[:, np.newaxis]  # Shape (nJet, 1)
        T_values = np.sum(Jet_pt_extended * np.abs(np.cos(Jet_phi_extended - Jet_phiT)), axis=0)


        # Find the maximum value
        T_max = np.max(T_values)
        phiT_max = Jet_phiT[np.argmax(T_values)]
        features_.append(T_max)
        features_.append(phiT_max)





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
        jet1.SetPtEtaPhiM(Jet_pt[selected1]*(1-Jet_rawFactor[selected1])*Jet_PNetRegPtRawCorr[selected1]*Jet_PNetRegPtRawCorrNeutrino[selected1],Jet_eta[selected1],Jet_phi[selected1],Jet_mass[selected1])
        jet2.SetPtEtaPhiM(Jet_pt[selected2]*(1-Jet_rawFactor[selected2])*Jet_PNetRegPtRawCorr[selected2]*Jet_PNetRegPtRawCorrNeutrino[selected2],Jet_eta[selected2],Jet_phi[selected2],Jet_mass[selected2])
        dijet_pnet = jet1+jet2
        features_.append(jet1.Pt())
        features_.append(jet2.Pt())
        features_.append(dijet_pnet.Pt())
        features_.append(dijet_pnet.Eta())
        features_.append(dijet_pnet.Phi())
        features_.append(dijet_pnet.M())
# Event variables
# nJets
        features_.append(int(nJet))
        features_.append(int(np.sum(Jet_pt>20)))
        features_.append(int(np.sum(Jet_pt>30)))
        features_.append(int(np.sum(Jet_pt>50)))
        ht = 0
        for idx in range(nJet):
            ht = ht+Jet_pt[idx]
        features_.append((np.float32(ht)))
        features_.append(nMuon)
        features_.append(np.sum(Muon_pfRelIso04_all<0.15))
        features_.append(nElectron)
        #features_.append(nProbeTracks)
        features_.append(np.sum((Jet_pt>20) & (Jet_btagDeepFlavB>0.0490)))
        features_.append(np.sum((Jet_pt>20) & (Jet_btagDeepFlavB>0.2783)))
        features_.append(np.sum((Jet_pt>20) & (Jet_btagDeepFlavB>0.7100)))


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
        if 'Data' in processName:
            features_.append(1)
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
            features_.append(genWeight)
        assert Muon_isTriggering[muonIdx1]
        file_.append(features_)
    
    return file_
def main(fileName, maxEntries, maxJet, isMC, process):
    print("FileName", fileName)
    print("Process", process)
    assert maxJet==4
    fileData = treeFlatten(fileName=fileName, maxEntries=maxEntries, maxJet=maxJet, isMC=isMC, processName=process)
    df=pd.DataFrame(fileData)
    
    featureNames = getFlatFeatureNames(mc=True if ((isMC!=0) & (isMC!=39) & (isMC!=57)) else False)

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
    if 'Data' in process:
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
    isMC        = int(sys.argv[4] )
    process     = sys.argv[5] 
    
    main(fileName, maxEntries, maxJet, isMC, process)