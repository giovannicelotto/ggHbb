# %%
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
sys.path.append("/t3home/gcelotto/ggHbb/flatter/")
from treeFlatter import jetsSelector
# %%
'''
Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                 = sys.argv[2]   (z for Z boson, h for Higgs boson)
'''


# Now open the file and use the previous distribution
def saveMatchedJets(fileNames, path, prefix):


    for fileName in fileNames:
        fileData=[]        # to store mjj for the matched signals
        outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures"
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(outFolder +"/%s_%s.parquet"%(prefix, fileNumber)):
            # if you already saved this file skip
            print("%s_%s.parquet already present\n"%(prefix, fileNumber))
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
            Jet_nMuons                = branches["Jet_nMuons"][ev]
            Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
            Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
            Muon_isTriggering           = branches["Muon_isTriggering"][ev]
            Jet_qgl                     = branches["Jet_qgl"][ev]

            GenPart_pt                  = branches["GenPart_pt"][ev]
            GenPart_eta                 = branches["GenPart_eta"][ev]
            GenPart_mass                = branches["GenPart_phi"][ev]
            GenPart_phi                 = branches["GenPart_mass"][ev]

            #Muons
            nMuon                       = branches["nMuon"][ev]
            
            # limit the data to events where 4 jets are gen matched to higgs daughers
            if nJet<2:
                continue
            if len(GenJet_pt)<2:
                continue
            if prefix=='GluGluHToBB':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==25)
            elif prefix=='EWKZJets':
                m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==23)
            if np.sum(m)==2:
                pass
                #matchedEvents=matchedEvents+1
            
            #elif np.sum(m)==1:
            #    newM = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5)
            #    if np.sum(newM)==2:
            #        m=newM
            #        #matchedEvents=matchedEvents+1
            #    else:
            #        continue
            else:
                continue
            
            
            for idx in range(4):
                if idx>=nJet:
                    for j in range(7):
                        features_.append(0)
                else:
                    features_.append(Jet_pt[idx])
                    features_.append(Jet_eta[idx])
                    features_.append(Jet_phi[idx])
                    features_.append(Jet_mass[idx])
                    features_.append(Jet_btagDeepFlavB[idx])
                    features_.append(Jet_qgl[idx])
                    if Jet_nMuons[idx]>0:
                        features_.append(Muon_isTriggering[Jet_muonIdx1[idx]] + Muon_isTriggering[Jet_muonIdx2[idx]])
                    else:
                        features_.append(0)

            indices = np.where(m)[0]

            if ((np.array(indices)>3).any()):
                continue
            features_.append(indices[0])
            features_.append(indices[1])

            


            fileData.append(features_)
        
        fileData=pd.DataFrame(fileData, columns=[
            'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_qgl', 'jet1_nTrigMuons',
            'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_qgl', 'jet2_nTrigMuons',
            'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_qgl', 'jet3_nTrigMuons',
            'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_qgl', 'jet4_nTrigMuons',
            'true1', 'true2'
        ])
        fileData.to_parquet(outFolder+"/%s_%s.parquet"%(prefix, fileNumber))

    return 0


def main(nFiles, particle):
    df=pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    if particle == 'H':
        path = df.nanoPath[1]#"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Sep10/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240910_141038/0000"
        fileNames = glob.glob(path+'/**/GluGlu*.root', recursive=True)
        print("Looking for files in ", path)
        prefix="GluGluHToBB"
    #elif particle=="Z":
    #    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/EWKZJets2024Mar15/EWKZ2Jets_ZToQQ_TuneCP5_13TeV-madgraph-pythia8/crab_EWKZ2Jets_ZToQQ/240315_141326/0000"
    #    fileNames = glob.glob(path+'/EWKZ*.root')
    #    prefix="EWKZJets"
    if (nFiles > len(fileNames)) | (nFiles == -1):
        nFiles=len(fileNames)
    else:
        fileNames = fileNames[:nFiles]
        

    print("nFiles                : ", nFiles)
    saveMatchedJets(fileNames, path=path, prefix=prefix)

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                   = (sys.argv[2]).upper()    #(z for Z boson, h for Higgs boson)
    main(nFiles=nFiles, particle=particle)