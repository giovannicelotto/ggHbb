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
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from criterionEfficiencySummary import plotCriterionEfficiency
sys.path.append("/t3home/gcelotto/ggHbb/flatter")
from treeFlatter import jetsSelector
#from bdtJetSelector import bdtJetSelector
import xgboost as xgb
import random

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
def getTrueJets(nJet, Jet_genJetIdx, GenJet_partonMotherIdx, GenJet_partonFlavour, GenJet_partonMotherPdgId):
    idxJet1, idxJet2 = -123, -124       # index of the first jet satisfying requirements
    numberOfGoodJets=0              # number of jets satisfying requirements per event
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
    return idxJet1, idxJet2

# Now open the file and use the previous distribution
def evaluateCriterion(maxJet, fileNames, tag): 
    firstJetIsGood = 0   
    secondJetIsGood = 0   
    goodChoice,wrongChoice, matched, nonMatched, noPossiblePair, totalEntriesVisited, outOfEta = 0, 0, 0, 0, 0, 0, 0
    within3Jets = 0
    recoJetsBelow20 =0
    # goodChoice       = events where criterion worked
    # wrongChoice      = events criterion did not work
    # mathced          = events in which Higgs daughters are matched
    # nonMathced       = events in which Higgs daughters are not matched
    # noPossiblePair   = events where no candidate dijet with trig muon inside, eta<2.5 and considering only the first N jets was found
    # totalEntryVisited= all the events visited for checking some final computations
    # outOfEta         = events that are matched but the reco jets are |eta| > 2.5 so btagging is not appropriate
    wrongJetsMass = []
    
    differenceJetsQuarkPtEtaNonMathced = []
    etaTracker = 2.5
    print("\n***********************************************************************\n* Computing efficiency of criterion based on two  selected features \n***********************************************************************")
    
    #path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000"
    assert len(fileNames)>0, "No Filenames found. Check the path"
    for fileName in fileNames:
        print("Opening ", fileName)
        f = uproot.open(fileName)
        tree = f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries 
        totalEntriesVisited += maxEntries
        print("\nFile %d/%d :\nEntries : %d"%(fileNames.index(fileName), len(fileNames), tree.num_entries))
        
        for ev in  range(maxEntries):
            
            if (ev%(int(maxEntries/100))==0):
                sys.stdout.write('\r')
                sys.stdout.write("%d%%"%(ev/maxEntries*100))
                sys.stdout.flush()
                pass
            # GenPart
            nGenPart                    = branches["nGenPart"][ev]
            GenPart_pdgId               = branches['GenPart_pdgId'][ev]
            GenPart_genPartIdxMother    = branches["GenPart_genPartIdxMother"][ev]
            GenPart_Pt                  = branches["GenPart_pt"][ev]
            GenPart_Eta                 = branches["GenPart_eta"][ev]
            
            nGenJet                     = branches["nGenJet"][ev]
            GenJet_partonFlavour        = branches["GenJet_partonFlavour"][ev]
            GenJet_partonMotherIdx      = branches["GenJet_partonMotherIdx"][ev]
            GenJet_partonMotherPdgId    = branches["GenJet_partonMotherPdgId"][ev]
            Jet_genJetIdx               = branches["Jet_genJetIdx"][ev]

            # GenJets with Neutrino
            nGenJetNu                   = branches["nGenJetNu"][ev]
            GenJetNu_eta                = branches["GenJetNu_eta"][ev]
            GenJetNu_phi                = branches["GenJetNu_phi"][ev]
            GenJetNu_partonMotherIdx    = branches["GenJetNu_partonMotherIdx"][ev]
            GenJetNu_partonMotherPdgId  = branches["GenJetNu_partonMotherPdgId"][ev]
            GenJetNu_partonFlavour      = branches["GenJetNu_partonFlavour"][ev]

            Jet_eta                     = branches["Jet_eta"][ev]
            Jet_pt                      = branches["Jet_pt"][ev]
            Jet_phi                     = branches["Jet_phi"][ev]
            Jet_mass                    = branches["Jet_mass"][ev]
            Jet_puId                    = branches["Jet_puId"][ev]
            Jet_jetId                    = branches["Jet_jetId"][ev]

            Jet_btagPNetB               = branches["Jet_btagPNetB"][ev]
            Jet_tagUParTAK4B            = branches["Jet_tagUParTAK4B"][ev]
            nJet                        = branches["nJet"][ev]
            Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
            Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
            Muon_isTriggering           = branches["Muon_isTriggering"][ev]
            Jet_bReg2018                = branches["Jet_bReg2018"][ev]
            Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]
            Jet_nMuons                  = branches["Jet_nMuons"][ev]
            Jet_qgl                     = branches["Jet_qgl"][ev]
            nMuon                       = branches["nMuon"][ev]
            

            #ht = 0

            # Try to match to Jets with Neutrinos
            #Jet_genJetNuIdx = []
            #Jet_genJetNu_dR = []
            #for j in range(nJet):
            #    minDeltaR = 999
            #    minDeltaR_idx = 999
            #    for gjnu in range(nGenJetNu):
            #        delta_eta = (Jet_eta[j] - GenJetNu_eta[gjnu])
            #        delta_phi = (Jet_phi[j] - GenJetNu_phi[gjnu])
            #        if delta_phi > np.pi:
            #            delta_phi -= 2 * np.pi
            #        elif delta_phi < -np.pi:
            #            delta_phi += 2 * np.pi
            #        deltaR = np.sqrt(delta_eta**2 + delta_phi**2)
            #        if (deltaR<0.1) & (deltaR<minDeltaR):
            #            minDeltaR = deltaR
            #            minDeltaR_idx = gjnu
            #    if minDeltaR == 999:
            #        Jet_genJetNuIdx.append(-1)
            #    else:
            #        Jet_genJetNuIdx.append(minDeltaR_idx)
#
            #    Jet_genJetNu_dR.append(minDeltaR)
            #Jet_genJetNuIdx = np.array(Jet_genJetNuIdx)
            #Jet_genJetNu_dR = np.array(Jet_genJetNu_dR)


            idxJet1, idxJet2 = getTrueJets(nJet, Jet_genJetIdx, GenJet_partonMotherIdx, GenJet_partonFlavour, GenJet_partonMotherPdgId)
# in case one jets was not identified as higgs daughter  
            if ((idxJet1==-123) | (idxJet2==-124)):
                # events of nonMatched
                nonMatched+=1
            
                # look for quarks higgs daughters
                #mask = (GenPart_pdgId[GenPart_genPartIdxMother]==25) & (abs(GenPart_pdgId)==5)
                #if np.sum(mask)==2: # two true bquarks
                #    
                #    selected1, selected2, muonIdx, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, np.min([maxJet, nJet]), Jet_btagDeepFlavB)
                #
                #    #if (selected1!=0) & (selected2!=0):
                #    #    selected1=0
                #    if (selected1!=999) & (selected2!=999):
                #        jet1 = ROOT.TLorentzVector(0., 0., 0., 0.)
                #        jet2 = ROOT.TLorentzVector(0., 0., 0., 0.)
                #        jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1],Jet_eta[selected1],Jet_phi[selected1],Jet_mass[selected1])
                #        jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2],Jet_eta[selected2],Jet_phi[selected2],Jet_mass[selected2])
                #        isQuark1Leading = True if GenPart_Pt[mask][0]>GenPart_Pt[mask][1] else False
                #        quark1Pt = GenPart_Pt[mask][0] if isQuark1Leading else GenPart_Pt[mask][1]
                #        quark1Eta = GenPart_Eta[mask][0] if isQuark1Leading else GenPart_Eta[mask][1]
                #        quark2Pt = GenPart_Pt[mask][1] if isQuark1Leading else GenPart_Pt[mask][0]
                #        quark2Eta = GenPart_Eta[mask][1] if isQuark1Leading else GenPart_Eta[mask][0]
                #        differenceJetsQuarkPtEtaNonMathced.append([(jet1.Pt() - quark1Pt)/quark1Pt, (jet1.Eta() - quark1Eta)/quark1Eta, (jet2.Pt() - quark2Pt)/quark2Pt, (jet2.Eta() - quark2Eta)/quark2Eta] )
                continue
            else:
                matched=matched+1
                if (Jet_pt[idxJet1]<20) | (Jet_pt[idxJet2]<20):
                    recoJetsBelow20 +=1
                    continue
                if ((abs(Jet_eta[idxJet1])>etaTracker)|(abs(Jet_eta[idxJet2])>etaTracker)):
                    outOfEta=outOfEta+1
                    continue
                
            assert idxJet1>-0.01
            assert idxJet2>-0.01

            # new selector
            #if nMuon>0:
            #    Jet_nTrigMuons = (Muon_isTriggering[Jet_muonIdx1]>0)+(Muon_isTriggering[Jet_muonIdx2]>0)
            #else:
            #    Jet_nTrigMuons = np.zeros(nJet)
            #bst_loaded = xgb.Booster()
            #bst_loaded.load_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')
            #selected1, selected2 = bdtJetSelector(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_btagDeepFlavB, Jet_qgl, Jet_nTrigMuons, bst_loaded, isMC=1)


            
            # if the reco jets are out of 2.5 no way that the choice will be correct
            jetsToCheck = np.min([nJet, maxJet])
                #selected1, selected2, muonIdx, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB)
                #if (selected1!=999) & (selected2!=999):
                #    jet1 = ROOT.TLorentzVector(0., 0., 0., 0.)
                #    jet2 = ROOT.TLorentzVector(0., 0., 0., 0.)
                #    jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1],Jet_eta[selected1],Jet_phi[selected1],Jet_mass[selected1]*Jet_bReg2018[selected1])
                #    jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2],Jet_eta[selected2],Jet_phi[selected2],Jet_mass[selected2]*Jet_bReg2018[selected2])
                #    wrongJetsMass.append([(jet1+jet2).Pt(), (jet1+jet2).M()])
            
                       
            taggers = [Jet_btagDeepFlavB, Jet_btagPNetB, Jet_tagUParTAK4B]
            tagLabel = ['DeepJet', 'PNet', 'PartT'][tag]
            selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=1, Jet_pt=Jet_pt)
            if nJet>2:
                for i in range(nJet):
                    if (i==selected1) | (i==selected2):
                        continue
                    else:
                        selected3 = i
                        break


            #print(ev, idxJet1, idxJet2, selected1, selected2)
            if ((idxJet1==selected1) | (idxJet2==selected1)):
                firstJetIsGood=firstJetIsGood+1
            if ((idxJet1==selected2) | (idxJet2==selected2)):
                secondJetIsGood=secondJetIsGood+1
            if ((idxJet1==selected1) & (idxJet2==selected2) | (idxJet1==selected2) & (idxJet2==selected1)): #choice we made is good
                goodChoice=goodChoice+1
            if ((idxJet1==selected1) & (idxJet2==selected2) |
                (idxJet1==selected2) & (idxJet2==selected1) |
                (idxJet1==selected1) & (idxJet2==selected3) |
                (idxJet1==selected3) & (idxJet2==selected1) |
                (idxJet1==selected2) & (idxJet2==selected3) |
                (idxJet1==selected3) & (idxJet2==selected2)): 
                
                within3Jets=within3Jets+1
            elif (selected1==999) & (selected2==999):       #no choice made (event skipped)
                noPossiblePair+=1
            else:
                wrongChoice+=1                              # choice made not good
                jet1 = ROOT.TLorentzVector(0., 0., 0., 0.)
                jet2 = ROOT.TLorentzVector(0., 0., 0., 0.)
                jet1.SetPtEtaPhiM(Jet_pt[selected1]*Jet_bReg2018[selected1],Jet_eta[selected1],Jet_phi[selected1],Jet_mass[selected1])
                jet2.SetPtEtaPhiM(Jet_pt[selected2]*Jet_bReg2018[selected2],Jet_eta[selected2],Jet_phi[selected2],Jet_mass[selected2])
                #wrongJetsMass.append([(jet1+jet2).Pt(), (jet1+jet2).M()])
            
    print("\nTriggering Jet is Good                                       : %d  \t  %.2f%%"%(firstJetIsGood, firstJetIsGood/totalEntriesVisited*100))
    print("Non-Triggering Jet is Good                                   : %d  \t  %.2f%%"%(secondJetIsGood, secondJetIsGood/totalEntriesVisited*100))
    
    print("\nTotal Entries visited                                      : %d  \t  %.2f%%" %(totalEntriesVisited, totalEntriesVisited/totalEntriesVisited*100))
    print("Non matched events                                           : %d  \t  %.2f%%" %(nonMatched, nonMatched/totalEntriesVisited*100))
    print("Events matched                                               : %d  \t  %.2f%%" %(matched, matched/totalEntriesVisited*100))
    print("Correct choice of jets                                       : %d  \t  %.2f%%" %(goodChoice, goodChoice/totalEntriesVisited*100))
    print("Wrong choice of jets                                         : %d  \t  %.2f%%" %(wrongChoice , wrongChoice/totalEntriesVisited*100))
    print("No Dijet with trigger inside within 2.5. No choice made      : %d  \t  %.2f%%" %(noPossiblePair , noPossiblePair/totalEntriesVisited*100))
    print("Out of Eta                                                   : %d  \t  %.2f%%" %(outOfEta , outOfEta/totalEntriesVisited*100))
    print("Below 20 GeV                                                 : %d  \t  %.2f%%" %(recoJetsBelow20 , recoJetsBelow20/totalEntriesVisited*100))
    print("Within 3 jets                                                : %d  \t  %.2f%%"%(within3Jets, within3Jets/totalEntriesVisited*100))
    print("Consistency = nonMatched + correct + wrong + noDijetWithTrigger + outOfEta: %d  \t  %.1f" %(nonMatched+goodChoice+noPossiblePair+wrongChoice+outOfEta, (nonMatched+goodChoice+noPossiblePair+wrongChoice)/totalEntriesVisited))
    print(maxJet, maxJet==4)
    if int(maxJet)==4:
        print("Only for maxJet == 4 saving the wrong masses and non matched ones")
        #np.save("/t3home/gcelotto/ggHbb/outputs/wrongJetsMassCriterion.npy", wrongJetsMass)
        #np.save("/t3home/gcelotto/ggHbb/outputs/nonMatchedQuarksPt.npy", differenceJetsQuarkPtEtaNonMathced)
    
        #print("Wrong jets mass saved length: ", len(wrongJetsMass))
        #print("Non matched jets saved length: ", len(differenceJetsQuarkPtEtaNonMathced))
    
    return nonMatched, matched, goodChoice, wrongChoice, noPossiblePair, outOfEta


def main(nFiles, maxJet1, maxJet2, tag):
    '''
    Take the first _nFiles_ NanoAOD files.
    If recomputeDistributions is True recompute the 2D distributions using the right jets coming from the Higgs
                                        Save an array with two features of the jets from higsg per event
    If _plot_ is true, replot the 2d distributions and save it
    if doEvaluate use __nFiles__  to evaluate the efficiency of the criterio using considering the _maxJet_ leading jets in pt
    _maxjet_ can be changed without saving new files
    '''
    

    #path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/allSamplesWW2025Jan20/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/250120_100810/0000"
    path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MCfiducial_corrections2025Mar10/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/250310_104448/0000/training"
    fileNames = glob.glob(path+'/**/*.root', recursive=True)
    #random.shuffle(fileNames)
    fileNames = fileNames[:nFiles]
        
    criterionSummary = {}
    for maxJet in range(maxJet1, maxJet2):
        print("nFiles                : ", nFiles)
        print("Max jet to check      : %d"%maxJet)
        nonMatched, matched, goodChoice, wrongChoice, noPossiblePair, outOfEta = evaluateCriterion(maxJet, fileNames, tag)
        criterionSummary[maxJet] = [nonMatched, matched, goodChoice, wrongChoice, noPossiblePair, outOfEta]
    tagLabel = ['DeepJet', 'PNet', 'ParT'][tag]
    with open("/t3home/gcelotto/ggHbb/outputs/dict_criterionEfficiency_%s.pkl"%tagLabel, 'wb') as file:
        pickle.dump(criterionSummary, file)
    plotCriterionEfficiency(tag=tagLabel)
    

if __name__ == "__main__":
    nFiles              = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    maxJet1             = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    maxJet2             = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tag                 = int(sys.argv[4])   # ['DeepJet', 'PNet', 'ParT'][tag]
    main(nFiles=nFiles, maxJet1=maxJet1, maxJet2=maxJet2, tag=tag)