import numpy as np
import matplotlib.pyplot as plt
import uproot
import sys
import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import pickle

'''
Choose a criterio to select the candidates jets that are most likely to come from the Higgs
Select _nFiles_ files and look for matched jets with the Higgs
Once we know which are the real jets from the Higgs consider the fist _maxJet_ in order of pT and choose the dijet that maximize the chosen criterion
Returns the percentage of selected pairs that match the correct pairs only when a complete matching is done.

Args:
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    maxJet1                   = int(sys.argv[2]) if len(sys.argv) > 1 else 2
    maxJet2                   = int(sys.argv[2]) if len(sys.argv) > 1 else 8


'''


# Now open the file and use the previous distribution
def evaluateCriterion(maxJet, fileNames):
    
    goodChoice = 0
    wrongChoice = 0
    matched = 0
    nonMatched = 0
    outOfConsideredJets = 0
    totalEntriesVisited = 0
    outOfEta=0
    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000"
    
    for fileName in fileNames:
        f = uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries 
        totalEntriesVisited += maxEntries
        print("\nFile %d/%d : %s\nEntries : %d"%(fileNames.index(fileName), len(fileNames), fileName[len(path)+1:], tree.num_entries))
    
        for ev in  range(maxEntries):
            
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()
                pass
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
            Jet_bRegNN2                 = branches["Jet_bRegNN2"][ev]
            Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
            Jet_nMuons                  = branches["Jet_nMuons"][ev]
            nGenJet                     = branches["nGenJet"][ev]

            idxJet1, idxJet2 = -123, -124       # index of the first jet satisfying requirements
            numberOfGoodJets=0              # number of jets satisfying requirements per event
            #ht = 0

            for i in range(nJet):
            # Find the jets from the signal
                if (Jet_genJetIdx[i]>-1):               # jet is matched to gen
                    #if Jet_genJetIdx[i]<nGenJet:                               # some events have jetGenIdx > nGenJet
                    if abs(GenJet_partonFlavour[Jet_genJetIdx[i]])==5:          # jet matched to genjet from b

                        if GenJet_partonMotherPdgId[Jet_genJetIdx[i]]==25:      # jet parton mother is higgs (b comes from h)
                            numberOfGoodJets=numberOfGoodJets+1
                            assert numberOfGoodJets<=2, "Error numberOfGoodJets = %d"%numberOfGoodJets                 # check there are no more than 2 jets from higgs
                            if idxJet1==-123:                                     # first match
                                idxJet1=i
                            elif GenJet_partonMotherIdx[Jet_genJetIdx[idxJet1]]==GenJet_partonMotherIdx[Jet_genJetIdx[i]]:  # second match. Also sisters
                                idxJet2=i    
            if ((idxJet1==-123) | (idxJet2==-124)):
                # take only events where there is the interested signal
                nonMatched+=1
                continue
            else:
                matched=matched+1
            assert idxJet1>-0.01
            assert idxJet2>-0.01
            
            # if the reco jets are out of 2.5 no way that the choice will be correct
            if ((abs(Jet_eta[idxJet1])>2.5)|(abs(Jet_eta[idxJet2])>2.5)):
                outOfEta=outOfEta+1
                continue
                       

            jetsToCheck = np.min([maxJet, nJet])
            score=-999

            # criterion 1:
            selected1 = 999
            selected2 = 999
            jetsWithMuon = []
            for i in range(nJet): # exclude the last jet because we are looking for pairs
                if abs(Jet_eta[i])>2.5:
                    continue
                if (Jet_muonIdx1[i]>-1): #if there is a muon
                    if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                        jetsWithMuon.append(i)
                        continue
                if (Jet_muonIdx2[i]>-1):
                    if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                        jetsWithMuon.append(i)
                        continue
                
            for i in jetsWithMuon:
                for j in range(0, jetsToCheck):
                    if i==j:
                        continue
                    if abs(Jet_eta[j])>2.5:
                        continue

                    
                    jet1 = ROOT.TLorentzVector(0.,0.,0.,0.)
                    jet2 = ROOT.TLorentzVector(0.,0.,0.,0.)
                    jet1.SetPtEtaPhiM(Jet_pt[i]*Jet_bRegNN2[i], Jet_eta[i], Jet_phi[i], Jet_mass[i])
                    jet2.SetPtEtaPhiM(Jet_pt[j]*Jet_bRegNN2[j], Jet_eta[j], Jet_phi[j], Jet_mass[j])
                    # massDr criterion
                    #deltaPhi = jet1.Phi()-jet2.Phi()
                    #deltaPhi = deltaPhi - 2*np.pi*(deltaPhi > np.pi) + 2*np.pi*(deltaPhi< -np.pi)
                    #tau = np.arctan(abs(deltaPhi)/abs(jet1.Eta() - jet2.Eta() + 0.0000001))

                    currentScore = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[j]
                    if currentScore>score:
                        score=currentScore
                        selected1 = min(i, j)
                        selected2 = max(i, j)

            #print(ev, idxJet1, idxJet2, selected1, selected2)
            if (idxJet1==selected1) & (idxJet2==selected2):
                goodChoice=goodChoice+1
            elif (selected1==999) & (selected2==999):
                outOfConsideredJets+=1
            else:
                wrongChoice+=1
            
    print("\nTotal Entries visited                                      : %d  \t  %.2f" %(totalEntriesVisited, totalEntriesVisited/totalEntriesVisited*100))
    print("Non matched events or outOfAcceptance abs(eta)>2.5           : %d  \t  %.2f" %(nonMatched, nonMatched/totalEntriesVisited*100))
    print("Events matched and within acceptance                         : %d  \t  %.2f" %(matched, matched/totalEntriesVisited*100))
    print("Correct choice of jets                                       : %d  \t  %.2f" %(goodChoice, goodChoice/totalEntriesVisited*100))
    print("Wrong choice of jets                                         : %d  \t  %.2f" %(wrongChoice , wrongChoice/totalEntriesVisited*100))
    print("Wrong choice of jets                                         : %d  \t  %.2f" %(wrongChoice , wrongChoice/totalEntriesVisited*100))
    print("Out of Eta                                                   : %d  \t  %.2f" %(outOfEta , outOfEta/totalEntriesVisited*100))
    
    print("Consistency = nonMatched + correct + wrong + noDijetWithTrigger: %d  \t  %.1f" %(nonMatched+goodChoice+outOfConsideredJets+wrongChoice+outOfEta, (nonMatched+goodChoice+outOfConsideredJets+wrongChoice)/totalEntriesVisited))
    
    return nonMatched, matched, goodChoice, wrongChoice, outOfConsideredJets, outOfEta


def main(nFiles, maxJet1, maxJet2):
    '''take the first _nFiles_ NanoAOD files.
    If recomputeDistributions is True recompute the 2D distributions using the right jets coming from the Higgs
                                        Save an array with two features of the jets from higsg per event
    If _plot_ is true, replot the 2d distributions and save it
    if doEvaluate use __nFiles__  to evaluate the efficiency of the criterio using considering the _maxJet_ leading jets in pt
    _maxjet_ can be changed without saving new files
    '''
    

    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/0000"
    fileNames = glob.glob(path+'/Hbb_QCDBackground_Run2_mc_2023Nov01*.root')
    fileNames = fileNames[:nFiles]
        
    criterionSummary = {}
    for maxJet in range(maxJet1, maxJet2):
        print("nFiles                : ", nFiles)
        print("Max jet to check      : %d"%maxJet)
        nonMatched, matched, goodChoice, wrongChoice, outOfConsideredJets, outOfEta = evaluateCriterion(maxJet, fileNames)
        criterionSummary[maxJet] = [nonMatched, matched, goodChoice, wrongChoice, outOfConsideredJets, outOfEta]
    with open("/t3home/gcelotto/ggHbb/outputs/dict_criterionEfficiency.pkl", 'wb') as file:
        pickle.dump(criterionSummary, file)




if __name__ == "__main__":
    nFiles                   = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    maxJet1                   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    maxJet2                   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(nFiles=nFiles, maxJet1=maxJet1, maxJet2=maxJet2)