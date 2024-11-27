# %%
import numpy as np
import uproot
import sys
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%

def findDaughters(myGenParts, motherIdx, GenPart_genPartIdxMother):
    '''Given a b quark idx (motherIdx) find all the daughters.'''
    daughters=[]
    for i in myGenParts:
        if GenPart_genPartIdxMother[i] == motherIdx:
            daughters.append(int(i))
    daughters = np.array(daughters)
    return daughters

def print_values_and_occurrences(arr, maxEntries):
    unique_values, counts = np.unique(arr, return_counts=True)

    for value, count in zip(unique_values, counts):
        print("%d: %.1f "%(value, count/maxEntries*100))
# %%

def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB):
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if abs(Jet_eta[i])>2.5:     # exclude jets>2.5 from the jets with  muon group
            continue
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                jetsWithMuon.append(i)
                muonIdxs.append(Jet_muonIdx1[i])
                continue
        if (Jet_muonIdx2[i]>-1):
            if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                jetsWithMuon.append(i)
                muonIdxs.append(Jet_muonIdx2[i])
                continue
    assert len(muonIdxs)==len(jetsWithMuon)
# Now loop over these jets as first element of the pair
    
    for i in jetsWithMuon:
        for j in range(0, jetsToCheck):
            #print(i, j, Jet_eta[j])
            if i==j:
                continue
            if abs(Jet_eta[j])>2.5:
                continue

            currentScore = Jet_btagDeepFlavB[i] + Jet_btagDeepFlavB[j]
            if currentScore>score:
                score=currentScore
                if j not in jetsWithMuon:  # if i has the muon only. jet1 is the jet with the muon.
                    selected1 = i
                    selected2 = j
                    muonIdx = muonIdxs[jetsWithMuon.index(i)]
                elif (i in jetsWithMuon) & (j in jetsWithMuon):
                    if muonIdxs[jetsWithMuon.index(i)] < muonIdxs[jetsWithMuon.index(j)]:
                        selected1 = i
                        selected2 = j
                        muonIdx = muonIdxs[jetsWithMuon.index(i)]
                    elif muonIdxs[jetsWithMuon.index(i)] > muonIdxs[jetsWithMuon.index(j)]:
                        selected1 = j
                        selected2 = i
                        muonIdx = muonIdxs[jetsWithMuon.index(j)]
                else:
                    assert False
    return selected1, selected2, muonIdx
# %%
filePath  = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000/GluGluHToBB_Run2_mc_2024Mar05_1.root"
#filePath  = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/ggH_noTrig_Run2_mc_124X.root"
f = uproot.open(filePath)
tree = f['Events']
branches = tree.arrays()
maxEntries = tree.num_entries
print("Entries : %d"%(maxEntries))

muonsSelected =0
muonsSelectedGenMatched =0
muonsSelectedGenMatchedFromHiggs =0
# %%
for ev in  range(maxEntries):
    #sys.stdout.write('\r')
    #sys.stdout.write("%.1f%%  %.1f%%  %d%%  %.1f%%  %.1f%%  %.1f%%  %.1f%%  "%(higgsEndingWithMuons/(ev+1)*100, muonsFromB/(totalGenMuons+1)*100, muonsFromD/(totalGenMuons+1)*100, muonsFromTau/(totalGenMuons+1)*100, muonsFromMuons/(totalGenMuons+1)*100, muonsFromPi/(totalGenMuons+1)*100, muonsFromKaons/(totalGenMuons+1)*100))
    #sys.stdout.flush()
    nGenPart                    = branches['nGenPart'][ev]
    GenPart_pdgId               = branches['GenPart_pdgId'][ev]
    GenPart_statusFlags         = branches['GenPart_statusFlags'][ev]
    Muon_isTriggering           = branches["Muon_isTriggering"][ev]
    Muon_genPartIdx             = branches["Muon_genPartIdx"][ev]
    nMuon                       = branches["nMuon"][ev]
    GenPart_genPartIdxMother    = branches["GenPart_genPartIdxMother"][ev]
    nJet                        = branches["nJet"][ev]
    Jet_eta                     = branches["Jet_eta"][ev]
    Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
    Jet_muonIdx2                = branches["Jet_muonIdx2"][ev]
    Jet_btagDeepFlavB           = branches["Jet_btagDeepFlavB"][ev]

    jet1Idx, jet2Idx, muonIdx = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, np.min([4, nJet])     , Jet_btagDeepFlavB)
    if muonIdx!=999:
        #muon is selected
        muonsSelected=muonsSelected+1
        if Muon_genPartIdx[muonIdx]!=-1:
            #muon is selected and genmatched
            muonsSelectedGenMatched=muonsSelectedGenMatched+1 
            genPart = Muon_genPartIdx[muonIdx]
            while (GenPart_genPartIdxMother[genPart]!=-1) & (GenPart_pdgId[GenPart_genPartIdxMother[genPart]]!=25):

                genPart = GenPart_genPartIdxMother[genPart]
            if GenPart_pdgId[GenPart_genPartIdxMother[genPart]]==25:
                muonsSelectedGenMatchedFromHiggs=muonsSelectedGenMatchedFromHiggs+1
            
            

    

print(muonsSelected)
print(muonsSelectedGenMatched)
print(muonsSelectedGenMatchedFromHiggs)


#print("Triggering", np.sum(isTriggering))
#print("matched muons", len(isTriggering))
#print("total reco ", totalRecoMuons)
#print("total gen ", totalGenMuons)
#print("mother muons", len(motherMuons))
#print("Events where at least one triggered muon comes from a matched one", triggeredByMatched)
#
#
#print_values_and_occurrences(motherMuons, len(motherMuons))
#unique_values, counts = np.unique(abs(motherMuons), return_counts=True)
#counts=counts/len(motherMuons)
#bins = np.arange(0, len(unique_values)+1)
#fig, ax = plt.subplots(1, 1, figsize=(30, 6))
#ax.hist(bins[:-1], bins=bins-0.5, weights=counts, color='blue', histtype=u'step')
#ax.set_xticks(bins[:-1])
#ax.set_xticklabels([str(i) for i in unique_values], rotation=90)
#ax.set_ylim(0, 0.5)
#ax.set_xlabel("PDG ID")
#ax.set_ylabel("Probability")
#ax.set_title("Pdg ID of mothers of muons")
#for i, count in enumerate(counts):
#    ax.text(bins[i], counts[i]*1.05, "%.1f%%"%(count*100), ha='center', va='bottom', color='black')
#
#fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/motherGenMuons.png", bbox_inches='tight')

