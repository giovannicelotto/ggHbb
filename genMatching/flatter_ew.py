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
from functions import getDfProcesses_v2
import random
sys.path.append("/t3home/gcelotto/ggHbb/flatter/")
from treeFlatter import jetsSelector
sys.path.append("/t3home/gcelotto/ggHbb/scripts/flatterScripts/")
from efficiencyJetSelection_muon_btag import getTrueJets
# %%
'''
Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                 = sys.argv[2]   (z for Z boson, h for Higgs boson)
'''
try:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    particle                 = (sys.argv[2]).upper()    
except:
    nFiles = 1
    particle = "H"

df=getDfProcesses_v2()[0]
if particle == 'H':
    path = df.nanoPath[0]
    fileNames = glob.glob(path+'/**/*.root', recursive=True)
    print("Looking for files in ", path)
    prefix="GluGluHToBB"
if (nFiles > len(fileNames)) | (nFiles == -1):
    nFiles=len(fileNames)
else:
    fileNames = fileNames[:nFiles]

nMatched = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
}

nWellChosen = {
    0:0,
    1:0,
    2:0,
}

notChosen = 0
jetsAreChosable = 0
dijetM = []
for fileName in fileNames:
    fileData=[]        # to store mjj for the matched signals
    outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched_new"
    fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
    
    if os.path.exists(outFolder +"/%s_GenMatched_%s.parquet"%(prefix, fileNumber)):
        print("%s_GenMatched_%s.parquet already present\n"%(prefix, fileNumber))
        continue
    
    f = uproot.open(fileName)
    tree = f['Events']
    print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName)+1, len(fileNames), fileName[len(path)+1:], tree.num_entries))
    branches = tree.arrays()
    maxEntries = tree.num_entries

    for ev in  range(10000):
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
        Jet_rawFactor               = branches["Jet_rawFactor"][ev]
        Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
        Jet_tagUParTAK4B            = branches["Jet_tagUParTAK4B"][ev]
        Jet_jetId                   =branches["Jet_jetId"][ev]
        Jet_puId                    =branches["Jet_puId"][ev]

        GenJet_pt                   = branches["GenJet_pt"][ev]
        GenJet_eta                  = branches["GenJet_eta"][ev]
        GenJet_phi                  = branches["GenJet_phi"][ev]
        GenJet_mass                 = branches["GenJet_mass"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]
        Jet_qgl                     = branches["Jet_qgl"][ev]
        GenJetNu_pt                 = branches["GenJetNu_pt"][ev]

        GenPart_pt                  = branches["GenPart_pt"][ev]
        GenPart_eta                 = branches["GenPart_eta"][ev]
        GenPart_mass                = branches["GenPart_phi"][ev]
        GenPart_phi                 = branches["GenPart_mass"][ev]
        GenPart_pdgId               =branches["GenPart_pdgId"][ev]

        #Muons
        nMuon                       = branches["nMuon"][ev]
        Muon_charge                 = branches["Muon_charge"][ev]
        
        # limit the data to events where 4 jets are gen matched to higgs daughers
        #if nJet<2:
        #    pass
        #if len(GenJet_pt)<2:
        #    pass

        m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==25)
        #idx1, idx2 = getTrueJets(nJet, Jet_genJetIdx, GenJet_partonMotherIdx, GenJet_partonFlavour, GenJet_partonMotherPdgId)
        #m = int(idx1>=0)  + int(idx2>=0)
        nMatched[np.sum(m)] = nMatched[np.sum(m)] + 1
        idx1, idx2 = np.arange(nJet)[m] if np.sum(m)==2 else [-31, -33]
        #nMatched[np.sum(m)] = nMatched[np.sum(m)] + 1
        #idx1, idx2 = np.arange(nJet)[m] if np.sum(m)==2 else [-31, -33]



        # 2 Jets Matched
        if ((idx1 >= 0) & (idx2>=0)):

            if (abs(Jet_eta[idx1])<=2.5) & (abs(Jet_eta[idx2])<=2.5):
                jetsAreChosable = jetsAreChosable + 1


                jetsToCheck = np.min([4, nJet])
                selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_jetId, Jet_puId)

                if (selected1 == 999) & (selected2==999):
                    notChosen = notChosen + 1
                    continue
                if ((selected1 == idx1) & (selected2 == idx2)) | ((selected1 == idx2) & (selected2 == idx1)):
                    nWellChosen[2] = nWellChosen[2] + 1
                elif ((selected1 == idx1) & (selected2 != idx2)) | ((selected1 == idx2) & (selected2 != idx1)) | ((selected1 != idx1) & (selected2 == idx2)) | ((selected1 != idx2) & (selected2 == idx1)):
                    nWellChosen[1] = nWellChosen[1] + 1
                elif ((selected1 != idx1) & (selected2 != idx2)) & ((selected1 != idx2) & (selected2 != idx1)):
                    nWellChosen[0] = nWellChosen[0] + 1
                else:
                    assert False


# %%
totalEntries = nMatched[0] + nMatched[1] + nMatched[2]
print("Matched ")
for key in nMatched.keys():
    if nMatched[key]<1:
        continue
    print(key, "%d / %d : %.3f "%(nMatched[key], totalEntries, nMatched[key]*100/totalEntries))

print("\n\n Right Jets within eta : %d /%d : %.1f"%(jetsAreChosable, totalEntries, jetsAreChosable*100/totalEntries))


print("\n\nWell Chosen ")
sum =0
for key in nWellChosen.keys():
    if nWellChosen[key]<1:
        continue
    print(key, "%d / %d : %.2f "%(nWellChosen[key], totalEntries, nWellChosen[key]/totalEntries ))


print("\n\nNot chosen ")
print("%d / %d : %.2f "%(notChosen, totalEntries, notChosen/totalEntries))

# %%
