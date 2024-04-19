import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re
import pandas as pd
import ROOT
import random
'''
Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                 = sys.argv[2]   (z for Z boson, h for Higgs boson)
'''


# Now open the file and use the previous distribution
def saveMatchedJets(fileNames, path, prefix):

    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    
    for fileName in fileNames:
        fileData=[]        # to store mjj for the matched signals
        outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(outFolder +"/%s_GenMatched_%s.parquet"%(prefix, fileNumber)):
            # if you already saved this file skip
            print("%s_GenMatched_%s.parquet already present\n"%(prefix, fileNumber))
            continue
        
        f = uproot.open(fileName)
        tree = f['Events']
        print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName)+1, len(fileNames), fileName[len(path)+1:], tree.num_entries))
        branches = tree.arrays()
        maxEntries = tree.num_entries

        for ev in  range(maxEntries):
            features_ = []
            
            GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
            GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
            GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
        # Reco Jets
            nJet                        = branches["nJet"][ev]
            Jet_eta                     = branches["Jet_eta"][ev]
            Jet_pt                      = branches["Jet_pt"][ev]
            Jet_phi                     = branches["Jet_phi"][ev]
            Jet_mass                    = branches["Jet_mass"][ev]
            Jet_bReg2018                = branches["Jet_bReg2018"][ev]
            Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
            Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
            GenJet_pt                   = branches["GenJet_pt"][ev]
            GenJet_eta                  = branches["GenJet_eta"][ev]
            GenJet_phi                  = branches["GenJet_phi"][ev]
            GenJet_mass                 = branches["GenJet_mass"][ev]
            Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
            Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
            Muon_isTriggering           = branches["Muon_isTriggering"][ev]
            Jet_qgl                     = branches["Jet_qgl"][ev]
            
            # limit the data to events where 4 jets are gen matched to higgs daughers
            if prefix=='GluGluHToBB':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==25)
            elif prefix=='EWKZJets':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==23)
            if np.sum(m)==2:
                pass
                #matchedEvents=matchedEvents+1
            
            elif np.sum(m)==1:
                newM = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5)
                if np.sum(newM)==2:
                    m=newM
                    #matchedEvents=matchedEvents+1
                else:
                    continue
            else:
                continue
            
            
            features_.append(Jet_pt[m][0])
            features_.append(Jet_eta[m][0])
            features_.append(Jet_phi[m][0])
            features_.append(Jet_mass[m][0])

            features_.append(Jet_pt[m][1])
            features_.append(Jet_eta[m][1])
            features_.append(Jet_phi[m][1])
            features_.append(Jet_mass[m][1])

            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet1.SetPtEtaPhiM(Jet_pt[m][0], Jet_eta[m][0], Jet_phi[m][0], Jet_mass[m][0])
            jet2.SetPtEtaPhiM(Jet_pt[m][1], Jet_eta[m][1], Jet_phi[m][1], Jet_mass[m][1])
            dijet = jet1 + jet2
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())


            features_.append(Jet_pt[m][0]*Jet_bReg2018[m][0])
            features_.append(Jet_mass[m][0]*Jet_bReg2018[m][0])

            features_.append(Jet_pt[m][1]*Jet_bReg2018[m][1])
            features_.append(Jet_mass[m][1]*Jet_bReg2018[m][1])

            jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
            jet1.SetPtEtaPhiM(Jet_pt[m][0]*Jet_bReg2018[m][0], Jet_eta[m][0], Jet_phi[m][0], Jet_mass[m][0]*Jet_bReg2018[m][0])
            jet2.SetPtEtaPhiM(Jet_pt[m][1]*Jet_bReg2018[m][1], Jet_eta[m][1], Jet_phi[m][1], Jet_mass[m][1]*Jet_bReg2018[m][1])
            
            dijet = jet1 + jet2
            features_.append(dijet.Pt())
            features_.append(dijet.Eta())
            features_.append(dijet.Phi())
            features_.append(dijet.M())

            features_.append(GenJet_pt[Jet_genJetIdx[m][0]])
            features_.append(GenJet_pt[Jet_genJetIdx[m][1]])

            



            fileData.append(features_)
        
        fileData=pd.DataFrame(fileData, columns=[
            'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
            'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
            'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass',

            'jet1Corr_pt', 'jet1Corr_mass',
            'jet2Corr_pt', 'jet2Corr_mass',
            'dijetCorr_pt', 'dijetCorr_eta', 'dijetCorr_phi', 'dijetCorr_mass',
            'genJet1_pt', 'genJet2_pt',
        ])
        fileData.to_parquet(outFolder+"/%s_GenMatched_%s.parquet"%(prefix, fileNumber))

    return 0


def main(nFiles, particle):
    if particle == 'H':
        path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000"
        fileNames = glob.glob(path+'/GluGlu*.root')
        prefix="GluGluHToBB"
    elif particle=="Z":
        path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/EWKZJets2024Mar15/EWKZ2Jets_ZToQQ_TuneCP5_13TeV-madgraph-pythia8/crab_EWKZ2Jets_ZToQQ/240315_141326/0000"
        fileNames = glob.glob(path+'/EWKZ*.root')
        prefix="EWKZJets"
    if (nFiles > len(fileNames)) | (nFiles == -1):
        pass
    else:
        fileNames = fileNames[:nFiles]
        

    print("nFiles                : ", nFiles)
    saveMatchedJets(fileNames, path=path, prefix=prefix)

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                   = (sys.argv[2]).upper()
    main(nFiles=nFiles, particle=particle)