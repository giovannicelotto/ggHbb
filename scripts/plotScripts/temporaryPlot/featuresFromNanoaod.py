import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import pickle
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/flatterScripts/')
from newCriterion import getTrueJets
sys.path.append("/t3home/gcelotto/ggHbb/flatter")
from treeFlatter import jetsSelector





def main(fileNames, process):
    data = []
    for fileName in fileNames:
        print(fileName)
        f = uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries 
        
        print("\nFile %d/%d :\nEntries : %d"%(fileNames.index(fileName), len(fileNames), tree.num_entries))
    
        for ev in  range(maxEntries):
            features=[]
            
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()
                pass
            nGenPart                    = branches["nGenPart"][ev]
            GenPart_pdgId               = branches['GenPart_pdgId'][ev]
            GenPart_genPartIdxMother    = branches["GenPart_genPartIdxMother"][ev]
            GenPart_Pt                  = branches["GenPart_pt"][ev]
            GenPart_Eta                 = branches["GenPart_eta"][ev]
            Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]
            GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
            GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
            GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
            Jet_eta                     = branches["Jet_eta"][ev]
            Jet_pt                      = branches["Jet_pt"][ev]
            Jet_phi                     = branches["Jet_phi"][ev]
            Jet_mass                    = branches["Jet_mass"][ev]
            nJet                        = branches["nJet"][ev]
            Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
            Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
            Muon_isTriggering           = branches["Muon_isTriggering"][ev]
            Jet_bReg2018                = branches["Jet_bReg2018"][ev]
            Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
            Jet_nMuons                  = branches["Jet_nMuons"][ev]
            #nGenJet                     = branches["nGenJet"][ev]

            idxJet1, idxJet2 = getTrueJets(nJet, Jet_genJetIdx, GenJet_partonMotherIdx, GenJet_partonFlavour, GenJet_partonMotherPdgId)
            if ((idxJet1==-123) | (idxJet2==-124)):
                continue

            jet1=ROOT.TLorentzVector(0. ,0. ,0., 0.)
            jet2=ROOT.TLorentzVector(0. ,0. ,0., 0.)
            jet1.SetPtEtaPhiM(Jet_pt[idxJet1], Jet_eta[idxJet1], Jet_phi[idxJet1], Jet_mass[idxJet1])
            jet2.SetPtEtaPhiM(Jet_pt[idxJet2], Jet_eta[idxJet2], Jet_phi[idxJet2], Jet_mass[idxJet2])

            features.append(jet1.Px())
            features.append(jet1.Py())
            features.append(jet1.Pz())
            features.append(jet2.Px())
            features.append(jet2.Py())
            features.append(jet2.Pz())
            
            data.append(features)

    return data


if __name__ == "__main__":
    #nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05"
    fileNames = glob.glob(path+'/**/*.root', recursive=True)
    fileNames = fileNames[:1]
    hbb = main(fileNames=fileNames, process='hbb')
    hbb = np.array(hbb)
    #np.save("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/trueJets/Hbb.npy", hbb)

    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data1A2024Mar05"
    fileNames = glob.glob(path+'/**/*.root', recursive=True)
    fileNames = fileNames[:1]
    data = main(fileNames=fileNames, process='bparking')
    data = np.array(data)
    #np.save("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/trueJets/bparking.npy", data)

    nrow, ncol = 2, 3
    fig, ax = plt.subplots(nrow, ncol)
    bins=np.linspace(0, 150, 30)

    for i in range(nrow):
        for j in range(ncol):
            print(i, j, i*nrow+j)
            c = np.histogram(np.clip(hbb[:,i*ncol+j], bins[0], bins[-1]), bins = bins)[0]
            #d = np.histogram(np.clip(data[:,i*ncol+j], bins[0], bins[-1]), bins = bins)[0]
            c= c/np.sum(c)
            #d= d/np.sum(d)
            ax[i, j].hist(bins[:-1], bins=bins, weights=c, color='blue', histtype=u'step')
            #ax[i, j].hist(bins[:-1], bins=bins, weights=d, color='red', histtype=u'step')

    fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/features/first4jetsFeatures.png", bbox_inches='tight')