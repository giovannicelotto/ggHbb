import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re
import random
'''



Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
'''


# Now open the file and use the previous distribution
def saveMatchedJets(fileNames, path):
    
    goodChoice = 0
    totalChoice = 0
    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    
    for fileName in fileNames:
        fileData=[ ]        # to store mjj for the matched signals
        outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/genOnly"
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1)
        if os.path.exists(outFolder +"/jjFeaturesTrue_%s.npy"%fileNumber):
            # if you already saved this file skip
            print("jjFeaturesTrue_%s.npy already present\n"%(fileNumber))
            continue
        
        f = uproot.open(fileName)
        tree = f['Events']
        print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName)+1, len(fileNames), fileName[len(path)+1:], tree.num_entries))
        branches = tree.arrays()
        maxEntries = tree.num_entries
    

        for ev in  range(maxEntries):
            features_ = []
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()
                pass
            GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
            GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
            GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
        # Reco Jets
            nJet                        = branches["nJet"][ev]
            
            Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
        
            idxJet1, idxJet2 = -1, -1       # index of the first jet satisfying requirements
            numberOfGoodJets=0              # number of jets satisfying requirements per event
            #ht = 0

            for i in range(nJet):
            # Find the jets from the signal
                if (Jet_genJetIdx[i]>-1):                                           # jet is matched to gen
                    if abs(GenJet_partonFlavour[Jet_genJetIdx[i]])==5:          # jet matched to genjet from b
                        if GenJet_partonMotherPdgId[Jet_genJetIdx[i]]==25:      # jet parton mother is higgs (b comes from h)
                            numberOfGoodJets=numberOfGoodJets+1
                            assert numberOfGoodJets<=2, "Error numberOfGoodJets = %d"%numberOfGoodJets                 # check there are no more than 2 jets from higgs
                            if idxJet1==-1:                                     # first match
                                idxJet1=i
                            elif GenJet_partonMotherIdx[Jet_genJetIdx[idxJet1]]==GenJet_partonMotherIdx[Jet_genJetIdx[i]]:  # second match. Also sisters
                                idxJet2=i    
            if ((idxJet1==-1) | (idxJet2==-1)):
                # take only events where there is the interested signal
                continue
            assert idxJet1>-0.01
            assert idxJet2>-0.01
            
            features_.append(idxJet1)
            features_.append(idxJet2)

            fileData.append(features_)
        
        fileData=np.array(fileData)
        np.save(outFolder+"/idxTwoTrueJets_%s.npy"%fileNumber, fileData)

    return 0


def main(nFiles):
    
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/0000"
    fileNames = glob.glob(path+'/ggH*.root')
    random.shuffle(fileNames)
    if (nFiles > len(fileNames)) | (nFiles == -1):
        pass
    else:
        fileNames = fileNames[:nFiles]
        

    print("nFiles                : ", nFiles)
    saveMatchedJets(fileNames, path=path)

if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    main(nFiles=nFiles)