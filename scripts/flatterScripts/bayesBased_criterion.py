import numpy as np
import uproot
import glob
import sys
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import random
from bayes_opt import BayesianOptimization
hep.style.use("CMS")
def blackBox( p0, p1, p2, p3):
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB_20UL18"
    fileNames = glob.glob(path+"/*.root")[:10]
    print("%d files to be used" %len(fileNames))
    correct = 0 #numerator = number of times the selected 4 jets corresponds to the 4 gen matched jets
    matchedEvents = 0 # denominator
    totalEntries = 0
    for fileName in fileNames:
        f = uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries 
        totalEntries = totalEntries + maxEntries
        print("Entries : %d" %maxEntries)

        for ev in  range(maxEntries):
            
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
            m = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5) & (GenJet_partonMotherPdgId[Jet_genJetIdx]==25)
            if np.sum(m)==2:
                pass
                matchedEvents=matchedEvents+1
            
            elif np.sum(m)==1:

                newM = (Jet_genJetIdx>-1) & (abs(GenJet_partonFlavour[Jet_genJetIdx])==5)
                
                if np.sum(newM)==2:
                    m=newM
                    matchedEvents=matchedEvents+1
                else:
                    continue
            else:
                continue
            
            

            # list of bools [nJet]
            IsTrigJet = []

            # list of scores based on pT (max), mass (max), |eta| (min), btag (max)
            scores = []
            jetsToCheck = np.min((4, nJet))
            for jetIdx in range(jetsToCheck):
                #jetTemp = ROOT.TLorentzVector(0.,0.,0.,0.)
                #jetTemp.SetPtEtaPhiM(Jet_pt[jetIdx], Jet_eta[jetIdx], Jet_phi[jetIdx], Jet_mass[jetIdx])
                #myRootJets.append(jetTemp)
                
                if Jet_muonIdx1[jetIdx]>-1:
                    if (Muon_isTriggering[Jet_muonIdx1[jetIdx]]):
                        IsTrigJet.append(100)
                        continue
                if Jet_muonIdx2[jetIdx]>-1:
                    if (Muon_isTriggering[Jet_muonIdx2[jetIdx]]):
                        IsTrigJet.append(100)
                    else:
                        IsTrigJet.append(0)
                else:
                    IsTrigJet.append(0)
            
            scores = p0*(Jet_pt[:jetsToCheck] - np.min(Jet_pt[:jetsToCheck])) / (np.max(Jet_pt[:jetsToCheck]) - np.min(Jet_pt[:jetsToCheck]))
            scores = scores + p1*(1- (abs(Jet_eta[:jetsToCheck]) - np.min(abs(Jet_eta[:jetsToCheck]))) / (np.max(abs(Jet_eta[:jetsToCheck])) - np.min(abs(Jet_eta[:jetsToCheck]))))
            scores = scores + p2*((Jet_qgl[:jetsToCheck] - np.min(Jet_qgl[:jetsToCheck])) / (np.max(Jet_qgl[:jetsToCheck]) - np.min(Jet_qgl[:jetsToCheck])))
            scores =  scores + p3*((Jet_btagDeepFlavB[:jetsToCheck] - np.min(Jet_btagDeepFlavB[:jetsToCheck])) / (np.max(Jet_btagDeepFlavB[:jetsToCheck]) - np.min(Jet_btagDeepFlavB[:jetsToCheck])))
            scores = scores + IsTrigJet
            taken = np.argsort(scores)[::-1][:2]
            #input("next\n")
            #print(taken, np.arange(nJet)[m])

            
            if np.array_equal(np.sort(taken), np.arange(nJet)[m]):
                correct = correct + 1
                #print("correct")
            else:
                #print("wrong")
                pass

    return correct/matchedEvents
                    
    
    

def main():
    print("main started")

    pbounds = {'p0': (0, 1),
               'p1': (0, 1),
               'p2': (0, 1),
               'p3': (0.5, 1)}
    optimizer = BayesianOptimization(
    f=blackBox,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
    optimizer.maximize(
    init_points=5,
    n_iter=50,
)
        
        #| 0.7988    | 0.2889    | 0.06528   | 0.03519   | 0.9019    |
        
        
    return

if __name__ == "__main__":
    main()